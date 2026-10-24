from assistant_rag.extraction import (
    Page,
    clean_pages,
    detect_boilerplate,
    extract_pages,
    normalize_text,
    split_articles,
)


def test_extract_pages_numbers_pages_from_one(regulation_pdf):
    pages = extract_pages(regulation_pdf, "test")

    assert [page.number for page in pages] == [1, 2, 3]
    assert all(page.document == "test" for page in pages)
    assert "Article premier" in pages[0].text


def test_detect_boilerplate_finds_headers_and_footers(regulation_pdf):
    boilerplate = detect_boilerplate(extract_pages(regulation_pdf, "test"))

    assert "#.#.# FR Journal officiel de l'Union européenne L #/#" in boilerplate
    assert "Page # sur #" in boilerplate
    assert not any(line.startswith("Article") for line in boilerplate)


def test_clean_pages_removes_boilerplate_only(regulation_pdf):
    pages = clean_pages(extract_pages(regulation_pdf, "test"))
    text = "\n".join(page.text for page in pages)

    assert "Journal officiel" not in text
    assert "Page 1 sur 3" not in text
    assert "Le présent règlement établit des règles." in text


def test_split_articles_follows_numbering(regulation_pdf):
    articles = split_articles(clean_pages(extract_pages(regulation_pdf, "test")))

    assert [article.number for article in articles] == [1, 2, 3]
    assert [article.title for article in articles] == [
        "Objet et objectifs",
        "Champ d'application matériel",
        "Modification du règlement (CE) no 300/2008",
    ]


def test_split_articles_tracks_pages_and_chapters(regulation_pdf):
    first, second, third = split_articles(clean_pages(extract_pages(regulation_pdf, "test")))

    assert (first.start_page, first.end_page, first.chapter) == (1, 2, "CHAPITRE I")
    assert (second.start_page, second.chapter) == (2, "CHAPITRE I")
    assert (third.start_page, third.end_page, third.chapter) == (2, 3, "CHAPITRE II")


def test_split_articles_excludes_chapter_titles_and_annexes(regulation_pdf):
    articles = split_articles(clean_pages(extract_pages(regulation_pdf, "test")))
    text = "\n".join(article.text for article in articles)

    assert "Principes" not in text
    assert "Fait à Bruxelles" not in text
    assert "Liste des législations." not in text
    assert "Texte cité d'un autre règlement." in articles[2].text


def test_split_articles_accepts_numeric_first_article():
    pages = [Page("test", 1, "Article 1er\nDéfinitions\nTexte de l'article.\nArticle 2\nEntrée en vigueur")]

    articles = split_articles(pages)

    assert [(article.number, article.title) for article in articles] == [(1, "Définitions"), (2, "Entrée en vigueur")]
    assert articles[0].text == "Texte de l'article."


def test_normalize_text_joins_words_split_by_soft_hyphen():
    assert normalize_text("indépen\u00ad\ndamment de") == "indépendamment de"


def test_normalize_text_replaces_non_breaking_spaces():
    assert normalize_text("article\u00a017\u202f:") == "article 17 :"


def test_clean_pages_collapses_multiple_spaces():
    pages = [Page("test", 1, "ou  le   droit d'un État")]

    assert clean_pages(pages, boilerplate=set())[0].text == "ou le droit d'un État"


def test_split_articles_joins_title_written_on_two_lines():
    text = "Article premier\nInformations à fournir lorsque les données\nn'ont pas été collectées\n1. Texte."

    (article,) = split_articles([Page("test", 1, text)])

    assert article.title == "Informations à fournir lorsque les données n'ont pas été collectées"
    assert article.text == "1. Texte."


def test_split_articles_removes_footnotes():
    text = "\n".join(
        [
            "Article premier",
            "Principes",
            "d) exactes;",
            "(1) Directive (UE) 2015/1535 du Parlement européen et du Conseil prévoyant une procédure dans le",
            "domaine des réglementations techniques (JO L 241 du 17.9.2015, p. 1).",
            "e) conservées.",
        ]
    )

    (article,) = split_articles([Page("test", 1, text)])

    assert article.text == "d) exactes;\ne) conservées."
