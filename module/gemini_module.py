import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

def get_gemini_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

st.title("🤖 Gemini Chat")

query = st.text_input("Enter your prompt:")
if st.button("Generate"):
    if query.strip():
        result = get_gemini_response(query)
        st.write("### ✨ Gemini Response")
        st.write(result)
    else:
        st.warning("Please enter a prompt.")
