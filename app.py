import os
import json
import numpy as np
import faiss
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image as PILImage
import google.generativeai as genai
from io import BytesIO
import base64
import time

ARTICLES_PATH = os.getenv("ARTICLES_PATH")
INDEX_DIR = os.getenv("INDEX_DIR")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
GENAI_KEY = os.getenv("GEMINI_API_KEY") 

st.set_page_config("Multimodal RAG", layout="wide")
st.title("📰 Multimodal RAG on The Batch")

@st.cache_resource
def load_model_and_index():
    st.write("Loading models and index...") 
    model_txt_only = SentenceTransformer("all-MiniLM-L6-v2")
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        st.error("Index files not found. Please run 'python create_index.py' first.")
        st.stop()

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    st.write("Models and index loaded successfully.") 
    return model_txt_only, model_clip, processor_clip, index, metadata

model_txt_only, model_clip, processor_clip, index, meta = load_model_and_index()

if not GENAI_KEY:
    st.error("Gemini API key not found. Please set GENAI_KEY in your .env file.")
genai.configure(api_key=GENAI_KEY)


def get_image_base64(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG") 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"mime_type": "image/jpeg", "data": img_str} 
    except Exception as e:
        st.warning(f"Failed to fetch or process image {image_url}: {e}") 
        return None

def query_embed_multimodal(q):
    text_only_dim = model_txt_only.get_sentence_embedding_dimension()
    clip_feature_dim = model_clip.config.projection_dim
    EMBEDDING_DIM_FINAL = text_only_dim + clip_feature_dim + clip_feature_dim

    text_embed_minilm = model_txt_only.encode(q, normalize_embeddings=True)

    clip_text_inputs = processor_clip(text=q, return_tensors="pt", padding=True, truncation=True)
    clip_text_features = model_clip.get_text_features(clip_text_inputs.input_ids).detach().cpu().numpy().flatten()
    clip_text_features = clip_text_features / np.linalg.norm(clip_text_features)

    image_features = np.zeros(clip_feature_dim, dtype='float32') 

    combined_embedding = np.concatenate([text_embed_minilm, clip_text_features, image_features])

    if combined_embedding.shape[0] != EMBEDDING_DIM_FINAL:
        raise ValueError(f"Query embedding dimension mismatch: Expected {EMBEDDING_DIM_FINAL}, got {combined_embedding.shape[0]}")

    return combined_embedding.astype('float32')

def query_gemini_multimodal(q, ctx):
    model_id = "models/gemini-1.5-flash"

    prompt_parts = [
        {"text": "You are a helpful assistant. Based on the provided articles and images, answer the user's question. "},
        {"text": "Context:\n"},
    ]

    for i, c in enumerate(ctx):
        prompt_parts.append({"text": f"### Article {i+1}: {c.get('title', 'No Title')}\n{c.get('text', '')[:500]}...\n"})
        if c.get("image_url"):
            image_data = get_image_base64(c.get("image_url"))
            if image_data:
                prompt_parts.append({"text": f"Image for Article {i+1}: "})
                prompt_parts.append(image_data)
        prompt_parts.append({"text": "---\n"})

    prompt_parts.append({"text": f"User question: {q}"})

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini: {e}")
        st.info("Please check your Gemini API key and ensure it has access to the 'gemini-1.5-flash' model.")
        return "Failed to generate a response."

query = st.text_input("🔎 Enter your query:")

if "evaluation_logs" not in st.session_state:
    st.session_state.evaluation_logs = []

if st.button("🔍 Find") and query.strip():
    with st.spinner("Searching and generating response..."):
        qe = query_embed_multimodal(query)

        if index.ntotal == 0:
            st.error("FAISS index is empty. Please ensure create_index.py ran successfully and created embeddings.")
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
                # st.write(article.get("text", "")[:300] + "...")
                st.write(article.get("text", "No content available."))

                if image_url := article.get("image_url"):
                    try:
                        st.image(image_url, width=300)
                    except Exception as e:
                        st.warning(f"Failed to load or display image from URL {image_url}: {e}")
                st.markdown("---")

        st.session_state.evaluation_logs.append({
            "query": query,
            "retrieved_articles": [a['title'] for a in ctx],
            "gemini_answer": answer,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + " UTC"
        })