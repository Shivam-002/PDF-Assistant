import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GENERATIVEAI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def text_generation(text):
    response = model.generate_content(text)
    return response.text
