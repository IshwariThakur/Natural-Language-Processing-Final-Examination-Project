from urllib.parse import urljoin
from common import get_soup, collect

BASE = "https://boeing.mediaroom.com"
LISTING = f"{BASE}/news-releases-statements"


def find_urls():
    urls = set()
    for offset in range(0, 125, 25):
        page = LISTING if offset == 0 else f"{LISTING}?o={offset}"
        soup = get_soup(page)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "item=" in href and "#assets_" not in href:
                urls.add(urljoin(BASE, href))
    return sorted(urls)


def parse_article(soup, url):
    title_tag = soup.find("div", class_="wd_title")
    body_tag = soup.find("div", class_="wd_news_body")
    if not body_tag:
        return None
    return {
        "title": title_tag.get_text(" ", strip=True) if title_tag else "",
        "text": body_tag.get_text(" ", strip=True),
    }


def run():
    return collect("Boeing", "competitor", find_urls, parse_article, "boeing_news.jsonl")


if __name__ == "__main__":
    run()