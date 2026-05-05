from google import genai

client = genai.Client(api_key='AIzaSyBCKflfH27M6sbrkP11zCk0EXAHBzaHON0')

for model in client.models.list():
    print(model.name)