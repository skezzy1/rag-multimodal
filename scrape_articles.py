import requests
from bs4 import BeautifulSoup
import json
import os
import re
from urllib.parse import urljoin, urlparse
from PIL import Image 
import io
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv
import concurrent.futures 

load_dotenv()

ROOT_DOMAIN = os.getenv("ROOT_DOMAIN", "https://www.deeplearning.ai") 
OUTPUT_PATH = os.getenv("ARTICLES_PATH", "data/articles.json")
IMAGE_DIR = os.getenv("IMAGE_DIR", "images") 

if os.path.dirname(OUTPUT_PATH):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def download_image(image_url, image_dir, article_slug):
    """
    Завантажує зображення за URL та зберігає його локально.
    Генерує унікальне ім'я файлу, використовуючи slug статті та оригінальне ім'я зображення.
    Обробляє різні режими зображення (наприклад, прозорість).
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        image_name_raw = os.path.basename(urlparse(image_url).path)
        image_name_sanitized = re.sub(r'[^\w.-]', '_', image_name_raw)
        url_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()[:8]
        final_image_name = f"{article_slug}_{url_hash}_{image_name_sanitized}"

        if not '.' in final_image_name or final_image_name.endswith(('.', '_')):
            content_type = response.headers.get('Content-Type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                final_image_name += '.jpg'
            elif 'png' in content_type:
                final_image_name += '.png'
            elif 'gif' in content_type:
                final_image_name += '.gif'
            elif 'webp' in content_type:
                final_image_name += '.webp'
            else:
                final_image_name += '.unknown'

        image_path = os.path.join(image_dir, final_image_name)

        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return final_image_name

        img = Image.open(io.BytesIO(response.content))
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')

        save_format = final_image_name.split('.')[-1].upper()
        if save_format == 'JPG':
            save_format = 'JPEG'
        img.save(image_path, format=save_format) 
        print(f"🖼️ Завантажено зображення: {final_image_name}")
        return final_image_name

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Помилка завантаження зображення {image_url}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Помилка обробки зображення {image_url}: {e}")
        return None

def get_all_issue_urls():
    """
    Виявляє всі унікальні URL-адреси випусків The Batch, ітеруючи сторінки пагінації.
    """
    issue_urls = set()
    page_num = 1

    while True:
        paginated_url = f"{ROOT_DOMAIN}/the-batch/page/{page_num}/"
        print(f"📄 Завантаження сторінки пагінації: {paginated_url}")
        try:
            response = requests.get(paginated_url, timeout=15)
            response.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"❌ Сторінка {paginated_url} не знайдена або більше немає сторінок: {e}")
            break 

        soup = BeautifulSoup(response.text, "html.parser")
        
        issues = soup.select("a[href*='/the-batch/issue-'][data-sentry-component='Link']")
        if not issues:
            issues = soup.select("a[href*='/the-batch/issue-']")

        if not issues and page_num > 1:
            print(f"⚠️ Не знайдено жодного випуску на сторінці {paginated_url}. Завершення.")
            break

        found_on_page = False
        for a in issues:
            href = a["href"]
            full_url = urljoin(ROOT_DOMAIN, href)
            if full_url.startswith(f"{ROOT_DOMAIN}/the-batch/issue-") and full_url not in issue_urls:
                issue_urls.add(full_url)
                found_on_page = True
        
        if not found_on_page and page_num > 1:
            print(f"⚠️ На сторінці {paginated_url} не знайдено нових випусків. Завершення.")
            break

        page_num += 1

    print(f"🔎 Виявляємо {len(issue_urls)} унікальних URL-адрес випусків.")
    return list(issue_urls)

def parse_single_issue_page(url):
    """
    Парсить одну сторінку випуску The Batch (яка, як правило, містить одну велику статтю/інформаційний бюлетень).
    """
    print(f"\n--- Парсинг сторінки випуску: {url} ---")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Не вдалося отримати {url}: {e}")
        return None 

    soup = BeautifulSoup(response.text, "html.parser")

    article_data = {
        "url": url, 
        "title": "",
        "date": "",
        "reading_time": "",
        "content": "", 
        "images": [] 
    }
    
    title_element = soup.select_one("h1.post-full-title") or \
                    soup.select_one("h1.article-title") or \
                    soup.select_one("h1") or \
                    soup.select_one("meta[property='og:title']")
    if title_element:
        article_data["title"] = title_element['content'] if title_element.name == 'meta' else title_element.get_text(strip=True)
        print(f"✅ Знайдено заголовок: '{article_data['title']}'")
    else:
        print(f"⚠️ Заголовок не знайдено для {url}")

    date_time_elements = soup.select("time.post-full-meta-date, span.byline-date, meta[property='article:published_time'], p, div, span, li")
    
    for elem in date_time_elements:
        text = elem.get_text(strip=True)
        
        date_match = re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}", text)
        if date_match and not article_data["date"]:
            article_data["date"] = date_match.group(0)
            print(f"DEBUG: Знайдено дату: '{article_data['date']}' з тексту: '{text}'")
        
        time_match = re.search(r"(\d+)\s+min read", text, re.IGNORECASE)
        if time_match and not article_data["reading_time"]:
            article_data["reading_time"] = time_match.group(0)
            print(f"DEBUG: Знайдено час читання: '{article_data['reading_time']}' з тексту: '{text}'")
        
        if article_data["date"] and article_data["reading_time"]:
            break 

    main_article_content_div = soup.find('div', class_='prose--styled justify-self-center post_postContent__wGZtc')
    
    if not main_article_content_div:
        content_div_selectors = [
            'div.gh-content',               
            'div.kg-card-markdown',         
            'section.post-full-content',    
            'div.entry-content',            
            'div.post-content',             
            'div.article-body',             
            'div.blog-post-content',        
            'div[itemprop="articleBody"]',  
            'div#article-content',          
            'div.article-content',          
            'div.content-main',             
            'article',                      
            'main'                          
        ]
        for selector in content_div_selectors:
            main_article_content_div = soup.select_one(selector)
            if main_article_content_div:
                print(f"✅ Знайдено основний div контенту за запасним селектором: '{selector}'")
                break

    if not main_article_content_div:
        print(f"🚫 Не вдалося знайти основний div контенту для {url}. Пропускаємо статтю.")
        return None

    article_content_parts = []
    images_in_content = []
    article_slug = re.sub(r'[^\w-]', '', article_data["title"].lower()[:50] or hashlib.md5(url.encode('utf-8')).hexdigest()[:8])

    for child in main_article_content_div.children:
        if child.name == 'h2' and child.get('id') == 'news':
            print("INFO: Знайдено заголовок 'News'. Припиняємо парсинг основного контенту.")
            break 
        
        if child.name == 'p' and not child.get('class'): 
            text = child.get_text(separator='\n', strip=True)
            if text:
                article_content_parts.append(text)
        
        elif child.name == 'figure' and 'kg-card' in child.get('class', []) and 'kg-image-card' in child.get('class', []):
            img_tag = child.find('img')
            if img_tag and img_tag.get('src'):
                src = img_tag.get('src')
                if not src or src.startswith('data:'):
                    continue
                full_img_url = urljoin(url, src)
                
                if any(x in full_img_url.lower() for x in ['icons', 'logo', 'avatar', '.svg', 'ads']):
                    continue
                
                downloaded_name = download_image(full_img_url, IMAGE_DIR, article_slug)
                if downloaded_name:
                    images_in_content.append(downloaded_name)
                    caption_tag = child.find('figcaption')
                    if caption_tag:
                        caption_text = caption_tag.get_text(strip=True)
                        if caption_text:
                            article_content_parts.append(f"Image Caption: {caption_text}")
        
        elif child.name in ['h1', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'div', 'blockquote']:
            text = child.get_text(separator='\n', strip=True)
            if text:
                article_content_parts.append(text)


    article_data["content"] = "\n\n".join(article_content_parts)
    article_data["images"] = images_in_content
    print(f"🖼️ Завантажено {len(images_in_content)} зображень.")

    if not article_data["content"].strip():
        print(f"🚫 Не знайдено значущого текстового вмісту для {url}. Це може бути порожня сторінка або проблема з селекторами тексту/фільтрацією.")
    else:
        print(f"✅ Зібрано текстовий вміст (довжина: {len(article_data['content'])} символів). Початок тексту: {article_data['content'][:500]}...")

    if article_data["title"].strip() or article_data["content"].strip():
        return article_data
    else:
        print(f"⚠️ Пропущено сторінку випуску {url} через відсутність заголовка та значного вмісту.")
        return None

def main():
    all_articles = []

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"Видалено існуючий {OUTPUT_PATH} для чистого скрапінгу.")

    if os.path.exists(IMAGE_DIR):
        import shutil
        shutil.rmtree(IMAGE_DIR)
        print(f"Видалено існуючий вміст {IMAGE_DIR} для чистого скрапінгу зображень.")
    os.makedirs(IMAGE_DIR, exist_ok=True) 

    issue_urls = get_all_issue_urls()
    if not issue_urls:
        print("Не знайдено жодних URL-адрес випусків. Перевірте ROOT_DOMAIN або підключення до Інтернету.")
        return

    print(f"🔎 Знайдено {len(issue_urls)} випусків. Починаємо обробку...")
    successful_articles_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor: 
        futures = {executor.submit(parse_single_issue_page, url): url for url in issue_urls}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обробка випусків The Batch"):
            article_from_issue = future.result()
            if article_from_issue:
                all_articles.append(article_from_issue)
                successful_articles_count += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    
    print(f"\n--- Підсумок скрапінгу ---")
    print(f"Загальна кількість успішно оброблених статей: {successful_articles_count}")
    print(f"Дані збережено у: {OUTPUT_PATH}")
    print(f"Зображення збережено у: {IMAGE_DIR}")

if __name__ == "__main__":
    main()
