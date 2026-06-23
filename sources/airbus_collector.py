from urllib.parse import urljoin
from common import get_soup, collect

BASE = "https://www.airbus.com"


def find_urls():
    urls = set()
    for page in range(5):
        soup = get_soup(f"{BASE}/en/newsroom/press-releases?langcode[en]=en&page={page}")
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            if "/en/newsroom/press-releases/" in a["href"]:
                urls.add(urljoin(BASE, a["href"]))
    return sorted(urls)


def parse_article(soup, url):
    main = soup.find("main")
    if not main:
        return None
    title = soup.title.get_text(strip=True) if soup.title else ""
    return {"title": title, "text": main.get_text(" ", strip=True)}


def run():
    return collect("Airbus", "company", find_urls, parse_article, "airbus_news.jsonl")


if __name__ == "__main__":
    run()