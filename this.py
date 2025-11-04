import google.generativeai as genai
genai.configure(api_key="AIzaSyCVzO-I8dk4B1JGj9J5MUzvG-jqNwTM-uo")
for m in genai.list_models():
    print(m.name)