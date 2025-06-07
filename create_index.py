import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

ARTICLES_PATH = os.getenv("ARTICLES_PATH")
INDEX_DIR = os.getenv("INDEX_DIR")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

os.makedirs(INDEX_DIR, exist_ok=True)

IMAGE_CACHE = {}

def get_image_from_url(url):
    if url in IMAGE_CACHE:
        return IMAGE_CACHE[url]
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content)).convert("RGB")
        IMAGE_CACHE[url] = img
        return img
    except Exception as e:
        return None

def create_faiss_index_multimodal():
    try:
        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {ARTICLES_PATH} not found. Ensure you ran scrape_articles.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: File {ARTICLES_PATH} contains invalid JSON. Please check its content.")
        return

    if not articles:
        print("No articles to index. Ensure scrape_articles.py successfully extracted data.")
        return

    print("Loading SentenceTransformer (text) model...")
    model_text_only = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading CLIP (multimodal) model and processor...")
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Models loaded.")

    embeddings = []
    metadata = []

    text_only_dim = model_text_only.get_sentence_embedding_dimension() 
    clip_feature_dim = model_clip.config.projection_dim 

    EMBEDDING_DIM_FINAL = text_only_dim + clip_feature_dim + clip_feature_dim 

    print(f"Creating multimodal embeddings for {len(articles)} articles (dimension: {EMBEDDING_DIM_FINAL})...")
    for i, article in enumerate(tqdm(articles)):
        text_content = article.get("content", "")
        title = article.get("title", "")
        image_url = article.get("images")[0] if article.get("images") else None

        text_embed_minilm = model_text_only.encode(text_content, normalize_embeddings=True)

        clip_text_input = title + " " + text_content[:200]
        clip_text_inputs = processor_clip(text=clip_text_input, return_tensors="pt", padding=True, truncation=True)
        clip_text_features = model_clip.get_text_features(clip_text_inputs.input_ids).detach().cpu().numpy().flatten()
        clip_text_features = clip_text_features / np.linalg.norm(clip_text_features) 

        image_features = np.zeros(clip_feature_dim, dtype='float32')
        if image_url:
            img = get_image_from_url(image_url)
            if img:
                clip_image_inputs = processor_clip(images=img, return_tensors="pt")
                image_features = model_clip.get_image_features(clip_image_inputs.pixel_values).detach().cpu().numpy().flatten()
                image_features = image_features / np.linalg.norm(image_features) 
            
        combined_embedding = np.concatenate([text_embed_minilm, clip_text_features, image_features])
        
        if combined_embedding.shape[0] != EMBEDDING_DIM_FINAL:
             print(f"Warning: Incorrect embedding dimension for article {i}. Expected {EMBEDDING_DIM_FINAL}, got {combined_embedding.shape[0]}. Attempting to adjust.")
             if combined_embedding.shape[0] < EMBEDDING_DIM_FINAL:
                 combined_embedding = np.pad(combined_embedding, (0, EMBEDDING_DIM_FINAL - combined_embedding.shape[0]), 'constant')
             else:
                 combined_embedding = combined_embedding[:EMBEDDING_DIM_FINAL]

        embeddings.append(combined_embedding)
        
        metadata.append({
            "url": article.get("url"),
            "title": article.get("title"),
            "text": text_content,
            "date": article.get("date"),
            "reading_time": article.get("reading_time"),
            "image_url": image_url,
        })

    if not embeddings:
        print("Failed to create embeddings for any articles.")
        return

    embeddings = np.array(embeddings).astype('float32')

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM_FINAL)
    index.add(embeddings)
    print("FAISS index created.")

    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Index and metadata successfully saved.")

if __name__ == "__main__":
    create_faiss_index_multimodal()