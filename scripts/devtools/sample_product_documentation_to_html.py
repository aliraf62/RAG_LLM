from pathlib import Path
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def clean_html_fragment(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "html.parser")
    # remove style/script
    for tag in soup(["style", "script"]):
        tag.decompose()
    # strip inline styles
    for tag in soup(True):
        tag.attrs.pop("style", None)
    # ensure <meta charset>
    head = soup.head or soup.new_tag("head")
    if not head.find("meta", {"charset": True}):
        head.insert(0, soup.new_tag("meta", charset="utf-8"))
    return "<!DOCTYPE html>\n" + str(soup)

def slugify_url(link: str, default_name: str) -> str:
    if link:
        parts = urlparse(link)
        path = parts.path.strip("/").replace("/", "_")
        if path:
            return f"{path}.html"
    return f"{default_name}.html"

def clean_parquet_html(input_file: str, out_dir: str):
    input_path = Path(input_file)
    output_path = Path(out_dir)
    output_path.mkdir(exist_ok=True)
    for i, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_html = rec.get("content", "")
        if not raw_html:
            continue
        cleaned = clean_html_fragment(raw_html)
        filename = slugify_url(rec.get("link", ""), f"doc_{i}")
        (output_path / filename).write_text(cleaned, encoding="utf-8")
        print(f"✓ Wrote {filename}")

# --- RUN IT: ---
# make sure you're in the folder where `sample_product_documentation` lives, or give full path. the sample file i made using parquet app in terminal ie. parquet head product_do....parquet
clean_parquet_html("../../data/coupa_datasets/sample_product_documentation", "../../outputs/sample_product_documentation_parquet_cleaned_html")
