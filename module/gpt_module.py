import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, context):
    prompt = f"""
User Query: {query}

Context:
"""
    for i, doc in enumerate(context):
        prompt += f"\n{i+1}. Title: {doc['title']}\n   Text: {doc['text'][:300]}..."

    prompt += "\n\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()