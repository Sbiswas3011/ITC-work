from ortools.sat.python import cp_model
import pandas as pd
import random
import json

# -----------------------------
# STEP 1: RULES (JSON)
# -----------------------------
rules_json = """
{
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

  "no_same_day": [
    ["PT", "FAT"],
    ["DES", "DES"]
  ],

  "no_overlap_projects": [
    ["P1", "P3"]
  ],

  "project_limits": {
    "max_per_day": 2
  },

  "objective": {
    "lateness_weight": 1,
    "spread_weight": 1
  }
}
"""

rules = json.loads(rules_json)

# -----------------------------
# STEP 2: CONFIG
# -----------------------------
NUM_DAYS = 15
LINES = list(rules["capacity"].keys())

# -----------------------------
# STEP 3: JOB GENERATION
# -----------------------------
types_sequence = ["Drive", "DES", "PU", "FAT"]

jobs_data = []
job_index = 1

for project_id in range(1, 6):
    project = f"P{project_id}"

    for t in types_sequence:
        jobs_data.append({
            "id": f"J{job_index}",
            "project": project,
            "type": t,
            "line": random.choice(LINES),
            "weight": random.randint(1, 3),
            "earliest": random.randint(0, 3),
            "due": random.randint(8, 14)
        })
        job_index += 1

# PT jobs
for i in range(6):
    jobs_data.append({
        "id": f"J{job_index}",
        "project": f"P{random.randint(1,5)}",
        "type": "PT",
        "line": "L1",
        "weight": 3,
        "earliest": random.randint(2, 5),
        "due": 14
    })
    job_index += 1

# -----------------------------
# STEP 4: MODEL
# -----------------------------
model = cp_model.CpModel()

job_day = {
    job["id"]: model.NewIntVar(job["earliest"], NUM_DAYS - 1, f'day_{job["id"]}')
    for job in jobs_data
}

# Boolean activation matrix
is_on_day = {}

for job in jobs_data:
    j = job["id"]
    for d in range(NUM_DAYS):
        b = model.NewBoolVar(f"{j}_on_{d}")
        is_on_day[(j, d)] = b

        model.Add(job_day[j] == d).OnlyEnforceIf(b)
        model.Add(job_day[j] != d).OnlyEnforceIf(b.Not())

# -----------------------------
# STEP 5: PRECEDENCE
# -----------------------------
for rule in rules["precedence"]:
    before, after, lag = rule["before"], rule["after"], rule["lag"]

    for project in set(j["project"] for j in jobs_data):
        proj_jobs = [j for j in jobs_data if j["project"] == project]

        j1 = next((j for j in proj_jobs if j["type"] == before), None)
        j2 = next((j for j in proj_jobs if j["type"] == after), None)

        if j1 and j2:
            model.Add(job_day[j2["id"]] >= job_day[j1["id"]] + lag)

# -----------------------------
# STEP 6: CAPACITY (line-based)
# -----------------------------
for d in range(NUM_DAYS):
    for line, cap in rules["capacity"].items():
        load = []

        for job in jobs_data:
            if job["line"] != line:
                continue

            j = job["id"]
            weight = job["weight"]

            load.append(is_on_day[(j, d)] * weight)

        model.Add(sum(load) <= cap)

# -----------------------------
# STEP 7: SPACING RULES
# -----------------------------
for rule in rules["spacing"]:
    job_type = rule["type"]
    gap = rule["min_gap"]

    filtered = [j["id"] for j in jobs_data if j["type"] == job_type]

    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            a, b = filtered[i], filtered[j]

            diff = model.NewIntVar(-NUM_DAYS, NUM_DAYS, f"diff_{a}_{b}")
            model.Add(diff == job_day[a] - job_day[b])

            abs_diff = model.NewIntVar(0, NUM_DAYS, f"abs_{a}_{b}")
            model.AddAbsEquality(abs_diff, diff)

            model.Add(abs_diff >= gap)

# -----------------------------
# STEP 8: SAME-DAY CONFLICTS
# -----------------------------
for a_type, b_type in rules.get("no_same_day", []):
    jobs_a = [j["id"] for j in jobs_data if j["type"] == a_type]
    jobs_b = [j["id"] for j in jobs_data if j["type"] == b_type]

    for d in range(NUM_DAYS):
        for ja in jobs_a:
            for jb in jobs_b:
                model.Add(
                    is_on_day[(ja, d)] + is_on_day[(jb, d)] <= 1
                )

# -----------------------------
# STEP 9: PROJECT CONFLICTS
# -----------------------------
for p1, p2 in rules.get("no_overlap_projects", []):
    jobs_p1 = [j["id"] for j in jobs_data if j["project"] == p1]
    jobs_p2 = [j["id"] for j in jobs_data if j["project"] == p2]

    for d in range(NUM_DAYS):
        model.Add(
            sum(is_on_day[(j, d)] for j in jobs_p1) +
            sum(is_on_day[(j, d)] for j in jobs_p2)
            <= 1
        )

# -----------------------------
# STEP 10: PROJECT DAILY LIMIT
# -----------------------------
max_per_day = rules["project_limits"]["max_per_day"]

for project in set(j["project"] for j in jobs_data):
    proj_jobs = [j["id"] for j in jobs_data if j["project"] == project]

    for d in range(NUM_DAYS):
        model.Add(
            sum(is_on_day[(j, d)] for j in proj_jobs)
            <= max_per_day
        )

# -----------------------------
# STEP 11: OBJECTIVE
# -----------------------------
lateness = []
spread_terms = []

for job in jobs_data:
    j = job["id"]

    late = model.NewIntVar(0, NUM_DAYS, f"late_{j}")
    model.Add(late >= job_day[j] - job["due"])
    model.Add(late >= 0)
    lateness.append(late)

for project in set(j["project"] for j in jobs_data):
    proj_jobs = [j["id"] for j in jobs_data if j["project"] == project]

    max_day = model.NewIntVar(0, NUM_DAYS, f"max_{project}")
    min_day = model.NewIntVar(0, NUM_DAYS, f"min_{project}")

    model.AddMaxEquality(max_day, [job_day[j] for j in proj_jobs])
    model.AddMinEquality(min_day, [job_day[j] for j in proj_jobs])

    spread = model.NewIntVar(0, NUM_DAYS, f"spread_{project}")
    model.Add(spread == max_day - min_day)

    spread_terms.append(spread)

model.Minimize(
    rules["objective"]["lateness_weight"] * sum(lateness) +
    rules["objective"]["spread_weight"] * sum(spread_terms)
)

# -----------------------------
# STEP 12: SOLVE
# -----------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10
solver.parameters.num_search_workers = 8

status = solver.Solve(model)

if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
    print("No solution found!")
    exit()

# -----------------------------
# STEP 13: EXPORT
# -----------------------------
rows = []

for job in jobs_data:
    rows.append({
        "JobID": job["id"],
        "Project": job["project"],
        "Type": job["type"],
        "Line": job["line"],
        "Day": solver.Value(job_day[job["id"]])
    })

df = pd.DataFrame(rows)

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

with pd.ExcelWriter("json_driven_schedule.xlsx", engine="openpyxl") as writer:
    final_df.to_excel(writer, sheet_name="Plan")
    df.to_excel(writer, sheet_name="Raw", index=False)

print("\n✅ Advanced constraint schedule generated: json_driven_schedule.xlsx")