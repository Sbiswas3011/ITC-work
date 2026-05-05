import os
import json
import copy
import random
import pandas as pd

from google import genai
from ortools.sat.python import cp_model


# -----------------------------
# GEMINI API KEY SETUP
# -----------------------------
os.environ["GEMINI_API_KEY"] = "AIzaSyBCKflfH27M6sbrkP11zCk0EXAHBzaHON0"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# -----------------------------
# BASE CONFIG
# -----------------------------
BASE_CONFIG = {
    "precedence": [
        {"before": "Drive", "after": "DES", "lag": 1},
        {"before": "DES", "after": "PU", "lag": 3},
        {"before": "PU", "after": "FAT", "lag": 1}
    ],

    "capacity": {
        "L1": 6,
        "L2": 5
    },

    "spacing": [
        {"type": "PT", "min_gap": 2}
    ],

    "no_same_day": [],
    "no_overlap_projects": [],

    "project_limits": {
        "max_per_day": 2
    }
}

NUM_DAYS = 15
LINES = list(BASE_CONFIG["capacity"].keys())


# -----------------------------
# JOB GENERATION
# -----------------------------
def generate_jobs():
    jobs = []
    idx = 1

    types = ["Drive", "DES", "PU", "FAT"]

    for p in range(1, 6):
        for t in types:
            jobs.append({
                "id": f"J{idx}",
                "project": f"P{p}",
                "type": t,
                "line": random.choice(LINES),
                "weight": random.randint(1, 3),
                "earliest": random.randint(0, 3),
                "due": random.randint(8, 14)
            })
            idx += 1

    for _ in range(6):
        jobs.append({
            "id": f"J{idx}",
            "project": f"P{random.randint(1,5)}",
            "type": "PT",
            "line": "L1",
            "weight": 3,
            "earliest": 2,
            "due": 14
        })
        idx += 1

    return jobs


jobs_data = generate_jobs()


# -----------------------------
# GEMINI: JSON PATCH GENERATOR
# -----------------------------
def get_llm_patch(config, instruction):

    prompt = f"""
You are a scheduling constraint editor.

Return ONLY valid JSON.

Format:
{{
  "add": {{}},
  "remove": {{}},
  "modify": {{}}
}}

Allowed keys:
- precedence
- capacity
- spacing
- no_same_day
- no_overlap_projects
- project_limits

Current config:
{json.dumps(config, indent=2)}

User instruction:
{instruction}

Rules:
- Output ONLY JSON
- No markdown
- No explanation
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    return json.loads(text)


# -----------------------------
# APPLY PATCH
# -----------------------------
def apply_patch(config, patch):
    config = copy.deepcopy(config)

    for k, v in patch.get("add", {}).items():
        if k not in config:
            config[k] = v
        elif isinstance(config[k], list):
            config[k].extend(v)
        elif isinstance(config[k], dict):
            config[k].update(v)

    for k, v in patch.get("remove", {}).items():
        if k in config:
            if isinstance(config[k], list):
                config[k] = [x for x in config[k] if x not in v]
            else:
                config.pop(k, None)

    for k, v in patch.get("modify", {}).items():
        if "." in k:
            a, b = k.split(".")
            config[a][b] = v
        else:
            config[k] = v

    return config


# -----------------------------
# VALIDATION
# -----------------------------
def validate(config):
    allowed = {
        "precedence",
        "capacity",
        "spacing",
        "no_same_day",
        "no_overlap_projects",
        "project_limits"
    }

    for k in config:
        if k not in allowed:
            raise ValueError(f"Invalid constraint key: {k}")

    return True


# -----------------------------
# CP-SAT MODEL
# -----------------------------
def build_model(config, jobs):
    model = cp_model.CpModel()

    job_day = {
        j["id"]: model.NewIntVar(j["earliest"], NUM_DAYS - 1, j["id"])
        for j in jobs
    }

    is_on_day = {}

    for j in jobs:
        jid = j["id"]
        for d in range(NUM_DAYS):
            b = model.NewBoolVar(f"{jid}_{d}")
            is_on_day[(jid, d)] = b

            model.Add(job_day[jid] == d).OnlyEnforceIf(b)
            model.Add(job_day[jid] != d).OnlyEnforceIf(b.Not())

    # Precedence
    for rule in config.get("precedence", []):
        for proj in set(j["project"] for j in jobs):
            pj = [x for x in jobs if x["project"] == proj]

            a = next((x for x in pj if x["type"] == rule["before"]), None)
            b = next((x for x in pj if x["type"] == rule["after"]), None)

            if a and b:
                model.Add(job_day[b["id"]] >= job_day[a["id"]] + rule["lag"])

    # Capacity
    for d in range(NUM_DAYS):
        for line, cap in config["capacity"].items():
            model.Add(
                sum(is_on_day[(j["id"], d)] * j["weight"]
                    for j in jobs if j["line"] == line)
                <= cap
            )

    # Spacing
    for rule in config.get("spacing", []):
        ids = [j["id"] for j in jobs if j["type"] == rule["type"]]

        for i in range(len(ids)):
            for k in range(i + 1, len(ids)):
                diff = model.NewIntVar(-NUM_DAYS, NUM_DAYS, "")
                model.Add(diff == job_day[ids[i]] - job_day[ids[k]])

                absd = model.NewIntVar(0, NUM_DAYS, "")
                model.AddAbsEquality(absd, diff)

                model.Add(absd >= rule["min_gap"])

    # Same-day constraints
    for a, b in config.get("no_same_day", []):
        A = [j["id"] for j in jobs if j["type"] == a]
        B = [j["id"] for j in jobs if j["type"] == b]

        for d in range(NUM_DAYS):
            for i in A:
                for j in B:
                    model.Add(is_on_day[(i, d)] + is_on_day[(j, d)] <= 1)

    # Project overlap constraints
    for p1, p2 in config.get("no_overlap_projects", []):
        A = [j["id"] for j in jobs if j["project"] == p1]
        B = [j["id"] for j in jobs if j["project"] == p2]

        for d in range(NUM_DAYS):
            model.Add(
                sum(is_on_day[(i, d)] for i in A) +
                sum(is_on_day[(j, d)] for j in B)
                <= 1
            )

    return model, job_day


# -----------------------------
# RUN SYSTEM
# -----------------------------
# def run(instruction):
#     config = BASE_CONFIG

#     patch = get_llm_patch(config, instruction)

#     print("PATCH:")
#     print(json.dumps(patch, indent=2))

#     config = apply_patch(config, patch)
#     validate(config)

#     model, job_day = build_model(config, jobs_data)

#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = 10

#     status = solver.Solve(model)

#     if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
#         print("No solution found")
#         return

#     print("\nSCHEDULE:\n")

#     for j in jobs_data:
#         print(j["id"], j["project"], solver.Value(job_day[j["id"]]))


def run(instruction):
    config = BASE_CONFIG

    patch = get_llm_patch(config, instruction)

    print("PATCH:")
    print(json.dumps(patch, indent=2))

    config = apply_patch(config, patch)
    validate(config)

    model, job_day = build_model(config, jobs_data)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    status = solver.Solve(model)

    if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        print("No solution found")
        return

    # -----------------------------
    # BUILD FLAT TABLE
    # -----------------------------
    rows = []
    for j in jobs_data:
        rows.append({
            "JobID": j["id"],
            "Project": j["project"],
            "Type": j["type"],
            "Line": j["line"],
            "Day": solver.Value(job_day[j["id"]])
        })

    df = pd.DataFrame(rows)

    # -----------------------------
    # BUILD LINE-DAY GRID
    # -----------------------------
    schedule = {line: [""] * NUM_DAYS for line in LINES}

    for _, row in df.iterrows():
        line = row["Line"]
        day = row["Day"]
        job = row["JobID"]

        if schedule[line][day] == "":
            schedule[line][day] = job
        else:
            schedule[line][day] += f", {job}"

    final_df = pd.DataFrame.from_dict(schedule, orient="index")
    final_df.columns = [f"Day {d}" for d in range(NUM_DAYS)]
    final_df.index.name = "Line"

    # -----------------------------
    # WRITE EXCEL
    # -----------------------------
    file_name = "schedule_output.xlsx"

    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Plan")
        df.to_excel(writer, sheet_name="Raw", index=False)

    print(f"\nExcel generated: {file_name}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run("Make PT jobs conflict with FAT and increase L1 capacity to 7")