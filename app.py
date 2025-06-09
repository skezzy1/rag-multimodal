import os
import json
import numpy as np
import faiss
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image as PILImage
import google.generativeai as genai
from io import BytesIO
import base64
import time
from dotenv import load_dotenv
import re

load_dotenv()

ARTICLES_PATH = os.getenv("ARTICLES_PATH")
INDEX_DIR = os.getenv("INDEX_DIR")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
GENAI_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_DIR = os.getenv("IMAGE_DIR")  

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("📰 Multimodal RAG on The Batch")

if not GENAI_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=GENAI_KEY)

@st.cache_resource(show_spinner=False)
def load_model_and_index():
    model_txt_only = SentenceTransformer("all-MiniLM-L6-v2")
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Index files not found. Please run 'python create_index.py' first.")

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model_txt_only, model_clip, processor_clip, index, metadata

try:
    model_txt_only, model_clip, processor_clip, index, meta = load_model_and_index()
    st.success("Models and index loaded successfully.")
except Exception as e:
    st.error(f"Failed to load models/index: {e}")
    st.stop()

def get_local_image_base64(image_path):
    try:
        full_path = os.path.join(IMAGE_DIR, image_path) 
        img = PILImage.open(full_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"mime_type": "image/jpeg", "data": img_str}
    except Exception as e:
        st.warning(f"Не вдалося завантажити зображення {image_path}: {e}")
        return None

def query_embed_multimodal(q):
    text_only_dim = model_txt_only.get_sentence_embedding_dimension()
    clip_feature_dim = model_clip.config.projection_dim
    EMBEDDING_DIM_FINAL = text_only_dim + clip_feature_dim * 2  # clip_text + clip_image

    text_embed_minilm = model_txt_only.encode(q, normalize_embeddings=True)

    clip_text_inputs = processor_clip(text=q, return_tensors="pt", padding=True, truncation=True)
    clip_text_features = model_clip.get_text_features(**clip_text_inputs).detach().cpu().numpy().flatten()
    clip_text_features = clip_text_features / np.linalg.norm(clip_text_features)

    image_features = np.zeros(clip_feature_dim, dtype='float32')  

    combined_embedding = np.concatenate([text_embed_minilm, clip_text_features, image_features])

    if combined_embedding.shape[0] != EMBEDDING_DIM_FINAL:
        raise ValueError(f"Query embedding dimension mismatch: Expected {EMBEDDING_DIM_FINAL}, got {combined_embedding.shape[0]}")

    return combined_embedding.astype('float32')

def extract_relevant_sentences(text, query, max_sentences=5):
    sentences = re.split(r'(?<=[.!?]) +', text)
    query_words = set(query.lower().split())

    relevant_sentences = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        if query_words.intersection(sent_words):
            relevant_sentences.append(sent)
        if len(relevant_sentences) >= max_sentences:
            break

    if relevant_sentences:
        return ' '.join(relevant_sentences)
    else:
        return text[:500] + "..."

def query_gemini_multimodal(q, ctx):
    model_id = "models/gemini-1.5-flash"

    prompt_parts = [
        {"text": "You are a helpful assistant. Using the following articles and their images, answer the user's question clearly and concisely."},
        {"text": "\nContext articles:\n"},
    ]

    for i, c in enumerate(ctx):
        relevant_text = extract_relevant_sentences(c.get('text', ''), q)
        prompt_parts.append({"text": f"Article {i+1} Title: {c.get('title', 'No Title')}\nContent excerpt: {relevant_text}\n"})

        images = c.get("images", [])
        if not images and c.get("image_path"):
            images = [c.get("image_path")]

        for image_path in images:
            image_data = get_local_image_base64(image_path)
            if image_data:
                prompt_parts.append({"text": f"Image from Article {i+1}:"})
                prompt_parts.append(image_data)

    prompt_parts.append({"text": f"\nUser question: {q}\nPlease provide a direct and well-structured answer."})

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt_parts)

        # Очищуємо відповідь від зайвих переносів рядків
        answer = response.text.strip().replace("\n\n", "\n").replace("\n", " ").strip()
        return answer
    except Exception as e:
        st.error(f"Error calling Gemini: {e}")
        return "Failed to generate a response."

query = st.text_input("🔎 Enter your query:")

if "evaluation_logs" not in st.session_state:
    st.session_state.evaluation_logs = []

def display_article_text_collapsible(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    html_text = (
        "<details><summary style='cursor:pointer; font-weight:bold; color:#4A90E2;'>Показати текст статті</summary>"
    )
    for para in paragraphs:
        html_text += f'<p style="text-align: justify; margin-bottom: 1em;">{para}</p>'
    html_text += "</details>"
    st.markdown(html_text, unsafe_allow_html=True)

if st.button("🔍 Find") and query.strip():
    with st.spinner("Searching and generating response..."):
        qe = query_embed_multimodal(query)

        if index.ntotal == 0:
            st.error("FAISS index is empty. Please run 'python create_index.py'.")
            st.stop()

        _, I = index.search(np.array([qe]), 3)
        ctx_indices = I[0]
        ctx = [meta[int(i)] for i in ctx_indices if 0 <= int(i) < len(meta)]

        answer = query_gemini_multimodal(query, ctx)

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Articles")
        if not ctx:
            st.info("No articles found for your query.")
        else:
            for article in ctx:
                st.markdown(f"### 📰 {article.get('title', 'No Title')}")
                display_article_text_collapsible(article.get("text", "No content available."))

                images = article.get("images", [])
                if not images and article.get("image_path"):
                    images = [article.get("image_path")]

                if images:
                    cols = st.columns(min(len(images), 3))
                    for i, img_name in enumerate(images):
                        try:
                            img_path = os.path.join(IMAGE_DIR, img_name)
                            with cols[i % 3]:
                                st.image(img_path, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Не вдалося показати зображення: {img_path}, помилка: {e}")

                st.markdown("---")

        st.session_state.evaluation_logs.append({
            "query": query,
            "retrieved_articles": [a.get('title', '') for a in ctx],
            "gemini_answer": answer,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + " UTC"
        })
