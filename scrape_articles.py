import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
import re 
from dotenv import load_dotenv
load_dotenv()

ROOT_DOMAIN = os.getenv("ROOT_DOMAIN")
OUTPUT_PATH = os.getenv("ARTICLES_PATH")
IMAGE_DIR = os.getenv("IMAGE_DIR")

os.makedirs(IMAGE_DIR, exist_ok=True)

def get_all_issue_urls():
    issue_urls = set()
    page_num = 1

    while True:
        paginated_url = f"{ROOT_DOMAIN}/the-batch/page/{page_num}/"
        print(f"📄 Завантаження сторінки: {paginated_url}")
        response = requests.get(paginated_url)
        if response.status_code != 200:
            print(f"❌ Сторінка {paginated_url} не знайдена або більше немає сторінок.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        issues = soup.select("a[href*='/the-batch/issue-'][data-sentry-component='Link']")
        if not issues:
            issues = soup.select("a[href*='/the-batch/issue-']")

        if not issues:
            print(f"⚠️ Не знайдено жодного випуску на сторінці {page_num}. Завершення.")
            break

        for a in issues:
            href = a["href"]
            full_url = ROOT_DOMAIN + href if href.startswith("/") else href
            issue_urls.add(full_url)

        page_num += 1

    return list(issue_urls)

def parse_issue(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Unable to access {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    print(f"\n--- Parse the output: {url} ---")

    all_articles = []
    
    article_blocks = soup.select("article, div.card, div[class*='article'], div[class*='prose-styled']")

    if not article_blocks:
        print(f"⚠️ No article blocks found on the page: {url}")
        return []

    for idx, block in enumerate(article_blocks):
        article_data = {
            "url": url,
            "title": "",
            "date": "",
            "reading_time": "",
            "content": "",
            "images": []
        }

        title_element = block.select_one("h1, h2, h3")
        if title_element:
            article_data["title"] = title_element.get_text(strip=True)

        date_element = block.find(text=re.compile(r"(Published|Issue date|Batch Date|[A-Z][a-z]{2,8} \d{1,2}, \d{4})"))
        if date_element:
            parent = date_element.find_parent()
            date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}", parent.get_text() if parent else date_element)
            if date_match:
                article_data["date"] = date_match.group(0)

        time_element = block.find(text=re.compile(r"\d+\s+min read"))
        if time_element:
            time_match = re.search(r"\d+\s+min read", time_element)
            if time_match:
                article_data["reading_time"] = time_match.group(0)

        paragraphs = block.find_all("p")
        content_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and not text.lower().startswith(("dear", "loading the elevenlabs")):
                content_paragraphs.append(text)
        article_data["content"] = "\n\n".join(content_paragraphs)

        for img in block.find_all("img"):
            img_src = img.get("src")
            if img_src and img_src.startswith("http"):
                article_data["images"].append(img_src)

        if article_data["title"] or article_data["content"]:
            all_articles.append(article_data)

    print(f"✅ The issue contains {len(all_articles)} articles.")
    return all_articles

def main():
    all_articles = []
    issue_urls = get_all_issue_urls()

    print(f"🔎 Issues found {len(issue_urls)}. Processing...")
    for url in tqdm(issue_urls):
        articles = parse_issue(url)
        all_articles.extend(articles)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved {len(all_articles)} articles in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
