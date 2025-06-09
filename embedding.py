# embedding.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List

# --- Конфігурація моделей ---
TEXT_MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim
IMAGE_MODEL_NAME = 'openai/clip-vit-base-patch32' # 512-dim

# Розмірності ембедингів
TEXT_EMBEDDING_DIM = 384
IMAGE_EMBEDDING_DIM = 512
COMBINED_EMBEDDING_DIM = TEXT_EMBEDDING_DIM + IMAGE_EMBEDDING_DIM # 896

# --- Завантаження моделей ---
# Використовуйте кешування на рівні модуля, щоб не завантажувати моделі при кожному виклику
try:
    print("Завантаження текстової моделі...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    print("Завантаження моделі зображень (CLIP)...")
    clip_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    print("✅ Моделі для ембедингів успішно завантажено.")
except Exception as e:
    print(f"❌ Помилка завантаження моделей: {e}")
    text_model = None
    clip_model = None
    clip_processor = None


def get_text_embedding(text: str) -> np.ndarray:
    """Генерує векторний ембединг для тексту."""
    if not text_model or not text:
        return np.zeros(TEXT_EMBEDDING_DIM, dtype=np.float32)
    
    embedding = text_model.encode(text, convert_to_tensor=False)
    return embedding.astype(np.float32)

def get_image_embedding(image_paths: List[str]) -> np.ndarray:
    """Генерує векторний ембединг для списку шляхів до зображень."""
    if not clip_model or not clip_processor or not image_paths:
        return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)

    images = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                images.append(img)
        except Exception as e:
            print(f"⚠️ Не вдалося завантажити зображення: {path}, помилка: {e}")

    if not images:
        return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)
    
    try:
        inputs = clip_processor(text=None, images=images, return_tensors="pt", padding=True)
        image_features = clip_model.get_image_features(**inputs)
        
        # Усереднюємо ембединги, якщо зображень кілька
        avg_embedding = image_features.mean(dim=0).detach().numpy()
        return avg_embedding.astype(np.float32)
    except Exception as e:
        print(f"❌ Помилка при створенні ембедингу зображення: {e}")
        return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)

def get_multimodal_embedding(text: str, image_paths: List[str] = []) -> np.ndarray:
    """
    Створює комбінований мультимодальний ембединг з тексту та зображень.
    Для запитів користувача `image_paths` буде порожнім.
    """
    text_emb = get_text_embedding(text)
    image_emb = get_image_embedding(image_paths)

    # Об'єднуємо вектори
    combined_embedding = np.concatenate([text_emb, image_emb])
    
    # Нормалізація (за бажанням, але часто покращує результати для L2 відстані)
    norm = np.linalg.norm(combined_embedding)
    if norm != 0:
        combined_embedding = combined_embedding / norm

    return combined_embedding.astype(np.float32)

if __name__ == "__main__":
    # Тестовий запуск
    print("--- Тестування модуля ембедингів ---")
    
    # Створюємо фейкове зображення для тесту
    if not os.path.exists("images"):
        os.makedirs("images")
    fake_image_path = "images/test_image.jpg"
    if not os.path.exists(fake_image_path):
        Image.new('RGB', (100, 100), color = 'red').save(fake_image_path)
    
    sample_text = "A cat sitting on a table"
    sample_image_paths = [fake_image_path]

    embedding = get_multimodal_embedding(sample_text, sample_image_paths)
    
    if embedding is not None:
        print(f"✅ Тест успішний!")
        print(f"Розмірність фінального вектора: {embedding.shape[0]}")
        assert embedding.shape[0] == COMBINED_EMBEDDING_DIM, "Розмірність не відповідає очікуваній!"
    else:
        print("❌ Тест не вдався.")