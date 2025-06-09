import os
import json
import requests
from tqdm import tqdm
from urllib.parse import urlparse

ARTICLES_PATH = os.getenv("ARTICLES_PATH", "articles.json")
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False

print(f"📥 Downloading images for {len(articles)} articles...")
image_index = {}

for article in tqdm(articles):
    article_images = []
    for img_url in article.get("images", []):
        parsed_url = urlparse(img_url)
        filename = os.path.basename(parsed_url.path)
        filename = filename.split("?")[0]  

        save_path = os.path.join(IMAGE_DIR, filename)

        if not os.path.exists(save_path):
            if not download_image(img_url, save_path):
                continue  

        article_images.append({
            "filename": filename,
            "original_url": img_url
        })

    article["local_images"] = article_images

with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"✅ Image download complete. Articles with image paths updated in {ARTICLES_PATH}")
