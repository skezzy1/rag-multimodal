import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

ARTICLES_PATH = os.getenv("ARTICLES_PATH")
INDEX_DIR = os.getenv("INDEX_DIR")
IMAGES_DIR = "Images"

INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

os.makedirs(INDEX_DIR, exist_ok=True)

IMAGE_CACHE = {}

def get_image_from_local(image_filename):
    if image_filename in IMAGE_CACHE:
        return IMAGE_CACHE[image_filename]
    
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    try:
        img = PILImage.open(image_path).convert("RGB")
        IMAGE_CACHE[image_filename] = img
        return img
    except Exception as e:
        print(f"Warning: Failed to load image from {image_path}: {e}")
        return None

def create_faiss_index_multimodal():
    try:
        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {ARTICLES_PATH} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File {ARTICLES_PATH} contains invalid JSON.")
        return

    if not articles:
        print("No articles to index.")
        return

    print("Loading models...")
    model_text_only = SentenceTransformer("all-MiniLM-L6-v2")
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Models loaded.")

    embeddings = []
    metadata = []

    text_only_dim = model_text_only.get_sentence_embedding_dimension()
    clip_feature_dim = model_clip.config.projection_dim
    EMBEDDING_DIM_FINAL = text_only_dim + clip_feature_dim + clip_feature_dim

    print(f"Creating multimodal embeddings for {len(articles)} articles (dim: {EMBEDDING_DIM_FINAL})...")

    for i, article in enumerate(tqdm(articles)):
        title = article.get("title", "")
        text_content = article.get("content", "")
        if not text_content.strip():
            print(f"Skipping article {i} due to empty content.")
            continue

        # Текстові ембеддинги
        full_text = (title + " " + text_content).strip()
        text_embed_minilm = model_text_only.encode(full_text, normalize_embeddings=True)

        clip_text_input = (title + " " + text_content[:500]).strip()
        clip_text_inputs = processor_clip(text=clip_text_input, return_tensors="pt", padding=True, truncation=True)
        clip_text_features = model_clip.get_text_features(**clip_text_inputs).detach().cpu().numpy().flatten()
        clip_text_features = clip_text_features / np.linalg.norm(clip_text_features)

        # Обробка всіх зображень
        image_features_list = []
        images = article.get("images", [])
        if images and isinstance(images, list):
            for img_filename in images:
                if isinstance(img_filename, str) and img_filename.strip():
                    img = get_image_from_local(img_filename.strip())
                    if img:
                        try:
                            clip_image_inputs = processor_clip(images=img, return_tensors="pt")
                            img_features = model_clip.get_image_features(**clip_image_inputs).detach().cpu().numpy().flatten()
                            img_features = img_features / np.linalg.norm(img_features)
                            image_features_list.append(img_features)
                        except Exception as e:
                            print(f"Error processing image {img_filename}: {e}")
                    else:
                        print(f"Warning: Could not load image {img_filename}.")

        # Усереднення або 0-вектор
        if image_features_list:
            image_features = np.mean(image_features_list, axis=0)
        else:
            image_features = np.zeros(clip_feature_dim, dtype='float32')

        # Комбіноване ембеддинг
        combined_embedding = np.concatenate([
            text_embed_minilm * 2.0,
            clip_text_features * 1.0,
            image_features * 0.5
        ])
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        if combined_embedding.shape[0] != EMBEDDING_DIM_FINAL:
            print(f"Warning: Embedding dimension mismatch at article {i}. Fixing...")
            combined_embedding = np.resize(combined_embedding, EMBEDDING_DIM_FINAL)

        embeddings.append(combined_embedding)

        metadata.append({
            "url": article.get("url"),
            "title": title,
            "text": text_content,
            "date": article.get("date"),
            "reading_time": article.get("reading_time"),
            "images": images if images else [],
        })

    if not embeddings:
        print("No valid embeddings created.")
        return

    embeddings = np.array(embeddings).astype('float32')

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM_FINAL)
    index.add(embeddings)

    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ Indexing complete.")

if __name__ == "__main__":
    create_faiss_index_multimodal()
