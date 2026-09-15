"""Download the Labor Code of the Russian Federation (TK RF) into a plain text file.

Source: legalacts.ru (one page per article). The official text of a law is not
protected by copyright (Civil Code of RF, art. 1259, p. 6).

Output format (this is what app/ingest/parser.py expects):

    # comment lines start with '#'
    Глава 13. Прекращение трудового договора

    Статья 81. Расторжение трудового договора по инициативе работодателя
    paragraph 1
    paragraph 2

Usage:
    python scripts/download_tk.py --out data/raw/tk_rf.txt
"""

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE = "https://legalacts.ru"
TOC_URL = f"{BASE}/kodeks/TK-RF/"
ARTICLE_HREF = re.compile(r"^/kodeks/TK-RF/[^\"]*/statja-[\d\-]+/$")
CHAPTER_HREF = re.compile(r"/glava-[\d\-]+/$")


def fetch(client: httpx.Client, url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {url}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def article_links(toc_html: str) -> list[str]:
    soup = BeautifulSoup(toc_html, "html.parser")
    seen: dict[str, None] = {}
    for a in soup.find_all("a", href=ARTICLE_HREF):
        seen.setdefault(a["href"], None)
    return list(seen)


def parse_article_page(html: str) -> tuple[str, str, list[str]]:
    """Return (chapter_header, article_header, paragraphs)."""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 is None:
        raise ValueError("no <h1> on article page")
    article_header = " ".join(h1.get_text(" ", strip=True).split())

    chapter_link = soup.find("a", href=CHAPTER_HREF, string=re.compile(r"^\s*Глава"))
    chapter_header = " ".join(chapter_link.get_text(" ", strip=True).split()) if chapter_link else ""

    body = soup.find("div", class_="main-center-block-article-text")
    paragraphs: list[str] = []
    if body is not None:
        for p in body.find_all("p"):
            text = " ".join(p.get_text(" ", strip=True).split())
            if text:
                paragraphs.append(text)
    return chapter_header, article_header, paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/tk_rf.txt")
    parser.add_argument("--delay", type=float, default=0.3, help="pause between requests, seconds")
    parser.add_argument("--limit", type=int, default=0, help="download only first N articles (debug)")
    args = parser.parse_args()

    headers = {"User-Agent": "Mozilla/5.0 (tk-rf-rag pet project)"}
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        links = article_links(fetch(client, TOC_URL))
        if args.limit:
            links = links[: args.limit]
        print(f"articles in TOC: {len(links)}")

        lines = [
            "# Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ",
            f"# Источник: {TOC_URL}",
            f"# Скачано: {date.today().isoformat()}",
            "",
        ]
        current_chapter = None
        for i, href in enumerate(links, 1):
            chapter, article, paragraphs = parse_article_page(fetch(client, BASE + href))
            if chapter and chapter != current_chapter:
                lines += [chapter, ""]
                current_chapter = chapter
            lines.append(article)
            lines += paragraphs
            lines.append("")
            if i % 50 == 0:
                print(f"  {i}/{len(links)}")
            time.sleep(args.delay)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
