import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("download_tk", Path(__file__).parents[1] / "scripts" / "download_tk.py")
download_tk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_tk)

TOC = """
<a href="/kodeks/TK-RF/chast-i/razdel-i/glava-2/">Глава 2</a>
<a href="/kodeks/TK-RF/chast-i/razdel-i/glava-2/statja-22/">Статья 22</a>
<a href="/kodeks/TK-RF/chast-i/razdel-i/glava-2/statja-22.1/">Статья 22.1</a>
<a href="/kodeks/TK-RF/chast-i/razdel-i/glava-2/statja-22/">duplicate</a>
<a href="/kodeks/TK-RF/chast-iii/razdel-x/glava-36.1/statja-226/">Статья 226</a>
<a href="/doc/other/">other</a>
"""

ARTICLE_PAGE = """
<a href="/kodeks/TK-RF/chast-iii/razdel-x/glava-36.1/">Глава 36.1. Специальная оценка условий труда</a>
<h1 class="main-center-block-article-header">Статья 226.  Заголовок статьи</h1>
<div class="main-center-block-article-text">
  <p class="pBoth"><a name="1"></a></p>
  <p class="pBoth">Первый   абзац <a href="/x">со ссылкой</a>.</p>
  <p class="pBoth">Второй абзац.</p>
</div>
"""


def test_article_links_include_dotted_numbers_and_are_unique():
    links = download_tk.article_links(TOC)
    assert links == [
        "/kodeks/TK-RF/chast-i/razdel-i/glava-2/statja-22/",
        "/kodeks/TK-RF/chast-i/razdel-i/glava-2/statja-22.1/",
        "/kodeks/TK-RF/chast-iii/razdel-x/glava-36.1/statja-226/",
    ]


def test_parse_article_page_with_dotted_chapter():
    chapter, article, paragraphs = download_tk.parse_article_page(ARTICLE_PAGE)
    assert chapter == "Глава 36.1. Специальная оценка условий труда"
    assert article == "Статья 226. Заголовок статьи"
    assert paragraphs == ["Первый абзац со ссылкой.", "Второй абзац."]
