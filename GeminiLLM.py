import google.generativeai as genai
from utils import template
from pdf_processor import query_pdf
import os
from fastapi import HTTPException

genai.configure(api_key=os.getenv("GENERATIVEAI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-1.5-flash")


def query_llm(filename, query):
    try:
        similar_chunks, similar_distances = query_pdf(filename, query)
        print("length : ", len(similar_chunks), "\nsimilar_chunks : ", similar_chunks)
        context = "\n".join(similar_chunks)
        prompt = template.format(question=query, context=context)
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Something went wrong. Please try again."
        )
