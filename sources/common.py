import time
import json
import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]      
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20            
DELAY = 1.5             
RETRIES = 3             
BACKOFF = 3             


def get_soup(url):
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            if resp.status_code in (429, 500, 502, 503, 504):   
                wait = BACKOFF * attempt
                print(f"  . {resp.status_code} (attempt {attempt}/{RETRIES}), waiting {wait}s")
                time.sleep(wait)
                continue
            print(f"  ! status {resp.status_code}: {url}")        
            return None
        except requests.RequestException as exc:
            wait = BACKOFF * attempt
            print(f"  . {exc.__class__.__name__} (attempt {attempt}/{RETRIES}), waiting {wait}s")
            time.sleep(wait)
    print(f"  ! gave up after {RETRIES} attempts: {url}")
    return None


def make_id(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def save_jsonl(records, filename):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records -> {path}")


def collect(source_name, source_type, find_urls, parse_article, out_file):
    print(f"\n=== Collecting: {source_name} ===")
    urls = find_urls()
    print(f"Found {len(urls)} URLs")

    records = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        soup = get_soup(url)
        if soup is None:
            continue

        data = parse_article(soup, url)
        if data and data.get("text"):
            records.append({
                "id": make_id(url),
                "source": source_name,
                "source_type": source_type,
                "title": data.get("title", ""),
                "url": url,
                "published": data.get("published"),   
                "text": data["text"],
            })
        time.sleep(DELAY)

    save_jsonl(records, out_file)
    return records