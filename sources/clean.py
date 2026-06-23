import re
import glob
import json
import hashlib

from common import RAW_DIR, CLEAN_DIR

OUT_FILE = "cleaned.jsonl"

DROP_TITLE_PATTERNS = [
    "media invitation",
    "media accreditation",
    "media programme",
    "media advisory",
    "call for interest",
    "call for media",
    "share buyback",
]

MIN_TEXT_LENGTH = 200

BOILERPLATE_PHRASES = [
    # "Accept all cookies",
    # "Back to top",
]

URL_PATTERN = re.compile(r"http\S+|www\.\S+")        
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")            


def remove_noise(text):
    text = text or ""
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    for phrase in BOILERPLATE_PHRASES:
        text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()         
    return text


def is_low_value(title):
    title = (title or "").lower()
    return any(pattern in title for pattern in DROP_TITLE_PATTERNS)


def strip_airbus_boilerplate(text):
    text = re.sub(r"^.*?Share Airbus X Airbus Facebook Airbus Linkedin Airbus Mail",
                  "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*(Downloads\s*)?(Contacts\s*)?", "", text)
    text = re.sub(r"^Breadcrumb Home Newsroom Latest Airbus global press releases\s*", "", text)
    text = re.sub(r"(Related keywords:|Register to receive Airbus|Related Assets).*$",
                  "", text, flags=re.DOTALL)
    text = re.sub(r"(\s*@\w+)+\s*$", "", text)
    return text.strip()


SOURCE_CLEANERS = {
    "Airbus": strip_airbus_boilerplate,
}


def clean():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    kept = []
    dropped = {"low_value": 0, "too_short": 0, "duplicate_text": 0}
    seen_text = set()

    for path in sorted(glob.glob(str(RAW_DIR / "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                doc["title"] = remove_noise(doc.get("title"))
                doc["text"] = remove_noise(doc.get("text"))

                cleaner = SOURCE_CLEANERS.get(doc.get("source"))
                if cleaner:
                    doc["text"] = cleaner(doc["text"])

                if is_low_value(doc["title"]):
                    dropped["low_value"] += 1
                    continue
                if len(doc["text"]) < MIN_TEXT_LENGTH:
                    dropped["too_short"] += 1
                    continue

                text_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()
                if text_hash in seen_text:
                    dropped["duplicate_text"] += 1
                    continue
                seen_text.add(text_hash)

                kept.append(doc)

    out_path = CLEAN_DIR / OUT_FILE
    with open(out_path, "w", encoding="utf-8") as fh:
        for doc in kept:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Kept {len(kept)} documents")
    print(f"Dropped -> low value: {dropped['low_value']}, "
          f"too short: {dropped['too_short']}, "
          f"duplicate text: {dropped['duplicate_text']}")
    print(f"Saved -> {out_path}")
    return kept


if __name__ == "__main__":
    clean()