import faiss
import numpy as np
import json
import os
from embedding import get_multimodal_embedding 
from typing import List, Dict

INDEX_DIR = os.getenv("INDEX_DIR", "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.json')

class MultimodalRetriever:
    """
    Клас для пошуку релевантних статей у FAISS індексі.
    """
    def __init__(self):
        self.index = None
        self.metadata = []
        self._load_index_and_metadata()

    def _load_index_and_metadata(self):
        """Завантажує FAISS індекс та метадані з файлів."""
        try:
            if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
                print("⚠️ Попередження: Файли індексу не знайдено. Запустіть `create_index.py` спочатку.")
                return

            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"✅ Завантажено FAISS індекс з {self.index.ntotal} векторами.")
            print(f"✅ Завантажено метадані для {len(self.metadata)} статей.")
        except Exception as e:
            print(f"❌ Помилка завантаження індексу або метаданих: {e}")
            self.index = None
            self.metadata = []

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self.index:
            print("❌ Індекс не завантажено. Пошук неможливий.")
            return []
            
        query_embedding = get_multimodal_embedding(query_text, image_paths=[])

        if query_embedding is None:
            print("❌ Не вдалося створити ембединг для запиту.")
            return []

        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_articles = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                article_info = self.metadata[idx]
                retrieved_articles.append({
                    'id': f"article-{idx}",
                    'distance': float(distances[0][i]),
                    'title': article_info.get('title'),
                    'content': article_info.get('content'),
                    'url': article_info.get('url'),
                    'images': article_info.get('images', [])
                })
        
        return retrieved_articles

if __name__ == '__main__':
    print("--- Тестування ретривера ---")
    retriever = MultimodalRetriever()
    if retriever.index:
        test_query = "latest advancements in large language models"
        results = retriever.retrieve(test_query, top_k=3)
        if results:
            print(f"\nЗнайдено {len(results)} результатів для запиту: '{test_query}'")
            for i, res in enumerate(results):
                print(f"{i+1}. {res['title']} (Distance: {res['distance']:.4f})")
        else:
            print("Нічого не знайдено.")