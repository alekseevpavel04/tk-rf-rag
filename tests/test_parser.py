from app.ingest.parser import parse_articles

RAW = """# Трудовой кодекс (фрагмент для теста)

Глава 13. Прекращение трудового договора

Статья 80. Расторжение трудового договора по инициативе работника (по собственному желанию)
Работник имеет право расторгнуть трудовой договор, предупредив об этом работодателя в письменной форме.
По соглашению между работником и работодателем трудовой договор может быть расторгнут и до истечения срока предупреждения.

Статья 81. Расторжение трудового договора по инициативе работодателя
Трудовой договор может быть расторгнут работодателем в случаях:
1) ликвидации организации;

Статья 81.1. Утратила силу. - Федеральный закон от 30.06.2006 N 90-ФЗ

Глава 49.1. Особенности регулирования труда дистанционных работников

Статья 312.1. Общие положения
Дистанционной работой является выполнение определенной трудовым договором трудовой функции вне места нахождения работодателя.
"""


def test_parses_articles_with_numbers_and_titles():
    articles = parse_articles(RAW)
    assert [a.number for a in articles] == ["80", "81", "312.1"]
    assert articles[1].title == "Расторжение трудового договора по инициативе работодателя"
    assert articles[1].header.startswith("Статья 81. ")


def test_assigns_chapter_metadata():
    articles = parse_articles(RAW)
    assert (articles[0].chapter, articles[0].chapter_title) == ("13", "Прекращение трудового договора")
    assert articles[2].chapter == "49.1"


def test_article_text_keeps_paragraphs_and_excludes_headers():
    art80 = parse_articles(RAW)[0]
    assert art80.text.count("\n") == 1
    assert "Статья" not in art80.text
    assert "Глава" not in art80.text


def test_repealed_articles_are_skipped_by_default_and_kept_on_request():
    assert "81.1" not in [a.number for a in parse_articles(RAW)]
    kept = parse_articles(RAW, skip_repealed=False)
    assert "81.1" in [a.number for a in kept]


def test_article_number_prefix_is_not_confused():
    # "Статья 3" must not match a line like "Статьями 3 и 4 ..." inside the text
    raw = "Глава 1. Общие\n\nСтатья 3. Запрещение дискриминации\nСтатьями 3 и 4 установлено ...\n"
    articles = parse_articles(raw)
    assert len(articles) == 1
    assert "Статьями 3 и 4" in articles[0].text
