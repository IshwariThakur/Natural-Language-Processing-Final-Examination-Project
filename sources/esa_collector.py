from urllib.parse import urljoin
from common import get_soup, collect

BASE = "https://www.esa.int"
ARCHIVE = f"{BASE}/Newsroom/Press_Releases/(lang)/en/(year)/2026"


def find_urls():
    soup = get_soup(ARCHIVE)
    if not soup:
        return []
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.startswith("/Newsroom/Press_Releases/")
            and "(lang)" not in href
            and "(year)" not in href
        ):
            urls.add(urljoin(BASE, href))
    return sorted(urls)


def parse_article(soup, url):
    title = soup.title.get_text(strip=True) if soup.title else ""
    article = soup.find("article")
    text = (article or soup).get_text(" ", strip=True)
    return {"title": title, "text": text}


def run():
    return collect("ESA", "agency", find_urls, parse_article, "esa_news.jsonl")


if __name__ == "__main__":
    run()