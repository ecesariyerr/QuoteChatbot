"""import json
import os
import re

INPUT_FILE = "quotes.json"
OUTPUT_DIR = "texts"

def safe_filename(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s]', '', s)
    return s[:50]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


for idx, entry in enumerate(data, start=1):
    # Burada sadece 'text' alanını alıyoruz; istersen author/tags de ekleyebilirsin
    content = entry.get("text", "").strip()
    if not content:
        continue

    # Dosya adı: quote_1.txt, ya da içeriğe göre safe name
    base = entry.get("text", "")
    filename = f"{idx:03d}_{safe_filename(base[:30])}.txt"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as out:
        out.write(content + "\n")

    print(f"Wrote {path}")"""

import json
import os
import re

INPUT_FILE = "quotes.json"      # your JSON file
OUTPUT_DIR = "texts"            # where the .txt files go

def safe_filename(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '_', s)
    return s[:50]

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, entry in enumerate(data, start=1):
    text   = entry.get("text", "").strip()
    author = entry.get("author", "").strip()
    tags   = entry.get("tags", [])

    if not text:
        continue

    filename = f"{idx:03d}_{safe_filename(text[:30])}.txt"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as out:
        out.write(f"TEXT: {text}\n")
        out.write(f"AUTHOR: {author}\n")
        out.write("TAGS: " + ", ".join(tags) + "\n")

    print(f"Wrote {path}")


