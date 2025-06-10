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
import time
import random

load_dotenv()

ROOT_DOMAIN = os.getenv("ROOT_DOMAIN")
OUTPUT_PATH = os.getenv("ARTICLES_PATH")
IMAGE_DIR = os.getenv("IMAGE_DIR")

if os.path.dirname(OUTPUT_PATH):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def download_image(image_url, image_dir, article_slug):
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
        print(f"Image uploaded: {final_image_name}")
        return final_image_name

    except requests.exceptions.RequestException as e:
        print(f"Image upload error {image_url}: {e}")
        return None
    except Exception as e:
        print(f"Image processing error {image_url}: {e}")
        return None

def get_all_issue_urls():
    issue_urls = set()
    page_num = 1

    while True:
        paginated_url = f"{ROOT_DOMAIN}/the-batch/page/{page_num}/"
        print(f"📄 Loading pagination page: {paginated_url}")
        try:
            sleep_time = random.uniform(0.5, 1.5) 
            print(f"Delay {sleep_time:.2f} seconds before requesting pagination {paginated_url}")
            time.sleep(sleep_time)

            response = requests.get(paginated_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f" Page {paginated_url} not found or no more pages: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        issues = soup.select("a[href*='/the-batch/issue-'][data-sentry-component='Link']")
        if not issues:
            issues = soup.select("a[href*='/the-batch/issue-']")

        if not issues and page_num > 1:
            print(f"⚠️ No issue found on page {paginated_url}. Ending.")
            break

        found_on_page = False
        for a in issues:
            href = a["href"]
            full_url = urljoin(ROOT_DOMAIN, href)
            if full_url.startswith(f"{ROOT_DOMAIN}/the-batch/issue-") and full_url not in issue_urls:
                issue_urls.add(full_url)
                found_on_page = True

        if not found_on_page and page_num > 1:
            print(f"⚠️ Не знайдено жодного випуску на сторінці {paginated_url}. Завершення.")
            break

        page_num += 1

    print(f"🔎 Виявляємо {len(issue_urls)} унікальних URL-адрес випусків.")
    return list(issue_urls)


def parse_single_issue_page(url):
    print(f"\n--- Parsing the release page: {url} ---")
    try:
        sleep_time = random.uniform(1, 3) 
        print(f"😴 Delay {sleep_time:.2f} seconds before querying {url}")
        time.sleep(sleep_time)

        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to retrieve {url}: {e}")
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
        print(f"✅ Title found:'{article_data['title']}'")
    else:
        print(f" ⚠️Title not found for {url}")
    date_time_elements = soup.select("time.post-full-meta-date, span.byline-date, meta[property='article:published_time'], p, div, span, li")
    
    date_text_to_remove = ""
    reading_time_text_to_remove = ""

    for elem in date_time_elements:
        text = elem.get_text(strip=True)

        date_match = re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}", text)
        if date_match and not article_data["date"]:
            article_data["date"] = date_match.group(0)
            date_text_to_remove = "Published " + article_data["date"] 
            print(f"DEBUG: Found date: '{article_data['date']}' from text: '{text}'")

        time_match = re.search(r"(\d+)\s+min read", text, re.IGNORECASE)
        if time_match and not article_data["reading_time"]:
            article_data["reading_time"] = time_match.group(0)
            reading_time_text_to_remove = article_data["reading_time"] 
            print(f"DEBUG: Found reading time: '{article_data['reading_time']}' from text: '{text}'")

        if article_data["date"] and article_data["reading_time"]:
            break

    main_article_content_div = soup.find('div', class_='prose--styled justify-self-center post_postContent__wGZtc')

    if not main_article_content_div:
        print(f"🚫 Could not find main content div 'prose--styled justify-self-center post_postContent__wGZtc' for {url}. Skipping article.")
        return None

    article_content_parts = []
    images_in_content = []
    article_slug = re.sub(r'[^\w-]', '', article_data["title"].lower()[:50] or hashlib.md5(url.encode('utf-8')).hexdigest()[:8])

    found_first_h2 = False

    image_downloaded = False

    for element in main_article_content_div.descendants:
        if element.name: 
            if element.name == 'h2':
                found_first_h2 = True
                print(f"INFO: First <h2> tag reached. Stopping parsing content and images.")
                break 

            if not image_downloaded and element.name == 'figure' and 'kg-card' in element.get('class', []) and 'kg-image-card' in element.get('class', []):
                img_tag = element.find('img')
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
                        image_downloaded = True 
                        caption_tag = element.find('figcaption')
                        if caption_tag:
                            caption_text = caption_tag.get_text(strip=True)
                            if caption_text:
                                article_content_parts.append(f"Image Caption: {caption_text}")
                continue 
            element_text = element.get_text(separator='\n', strip=True)
            
            if not element_text:
                continue 

            element_text = re.sub(r'\bDear friends,?\b', '', element_text, flags=re.IGNORECASE)
            element_text = re.sub(r'\bAndrew\b', '', element_text, flags=re.IGNORECASE)
            element_text = re.sub(r'\bKeep learning!\b', '', element_text, flags=re.IGNORECASE)
            element_text = re.sub(r'\bShare\b', '', element_text, flags=re.IGNORECASE) 
            
            if date_text_to_remove:
                element_text = element_text.replace(date_text_to_remove, "").strip()
            if reading_time_text_to_remove:
                element_text = element_text.replace(reading_time_text_to_remove, "").strip()

            element_text = re.sub(r'Published\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}', '', element_text).strip()
            element_text = re.sub(r'Reading time\s+\d+\s+min read', '', element_text, flags=re.IGNORECASE).strip()


            if element_text.strip():
                article_content_parts.append(element_text.strip())
    
    article_content_elements = [] 
    for elem in main_article_content_div.children: 
        if elem.name == 'h2':
            break
        article_content_elements.append(elem)

    for element in article_content_elements:
        if not image_downloaded and element.name == 'figure' and 'kg-card' in element.get('class', []) and 'kg-image-card' in element.get('class', []):
            img_tag = element.find('img')
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
                    image_downloaded = True
                    caption_tag = element.find('figcaption')
                    if caption_tag:
                        caption_text = caption_tag.get_text(strip=True)
                        if caption_text:
                            article_content_parts.append(f"Image Caption: {caption_text}")
            continue 

        if element.name:
            element_text = element.get_text(separator='\n', strip=True)
        elif isinstance(element, str): 
            element_text = element.strip()
        else:
            continue 
        
        if not element_text:
            continue

        element_text = re.sub(r'\bDear friends,?\b', '', element_text, flags=re.IGNORECASE)
        element_text = re.sub(r'\bAndrew\b', '', element_text, flags=re.IGNORECASE)
        element_text = re.sub(r'\bKeep Learning,\b', '', element_text, flags=re.IGNORECASE)
        element_text = re.sub(r'\bShare\b', '', element_text, flags=re.IGNORECASE)
        element_text = re.sub(r'\bKeep Learning!\b', '', element_text, flags=re.IGNORECASE)

        if date_text_to_remove:
            element_text = element_text.replace(date_text_to_remove, "").strip()
        if reading_time_text_to_remove:
            element_text = element_text.replace(reading_time_text_to_remove, "").strip()

        element_text = re.sub(r'Published\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}', '', element_text).strip()
        element_text = re.sub(r'Reading time\s+\d+\s+min read', '', element_text, flags=re.IGNORECASE).strip()

        if element_text.strip():
            article_content_parts.append(element_text.strip())


    article_data["content"] = "\n\n".join([part for part in article_content_parts if part.strip()])
    article_data["images"] = images_in_content

    if not article_data["content"].strip():
        print(f"🚫 No meaningful text content was found for {url} after filtering. This could be a blank page or a problem with text selectors/filtering.")
    else:
        print(f"✅ Text content collected. Total length: {len(article_data['content'])} characters.")
        print(f"🖼️ Loaded {len(images_in_content)} images (only one if found before h2).")


    if article_data["title"].strip() or article_data["content"].strip():
        return article_data
    else:
        print(f"⚠️ Skipped release page {url} due to lack of title and significant content.")
        return None

def main():
    all_articles = []

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"Removed existing {OUTPUT_PATH} for clean scraping.")

    if os.path.exists(IMAGE_DIR):
        import shutil
        shutil.rmtree(IMAGE_DIR)
        print(f"Removed existing contents of {IMAGE_DIR} for clean image scraping.")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    issue_urls = get_all_issue_urls()
    if not issue_urls:
        print("No release URLs found. Please check your ROOT_DOMAIN or internet connection.")
        return

    successful_articles_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_single_issue_page, url): url for url in issue_urls}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing of The Batch releases"):
            article_from_issue = future.result()
            if article_from_issue:
                all_articles.append(article_from_issue)
                successful_articles_count += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"\n--- Scraping Summary ---")
    print(f"Total number of successfully processed articles: {successful_articles_count}")
    print(f"Data saved in: {OUTPUT_PATH}")
    print(f"Image saved in: {IMAGE_DIR}")

if __name__ == "__main__":
    main()