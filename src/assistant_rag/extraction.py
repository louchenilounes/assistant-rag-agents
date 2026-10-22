"""Extraction et nettoyage du texte des PDF, découpage en articles."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from assistant_rag.config import DOCUMENTS, SourceDocument

ARTICLE_HEADING = re.compile(r"^Article\s+(premier|1er|\d+)$")
CHAPTER_HEADING = re.compile(r"^CHAPITRE\s+[IVXLC]+$")
SECTION_HEADING = re.compile(r"^Section\s+\d+$")
END_OF_ARTICLES = re.compile(r"^(ANNEXE\b|Fait à )")
SOFT_HYPHEN_BREAK = re.compile(r"\u00ad[ \t]*\n[ \t]*")
MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
EDGE_LINES = 3


@dataclass(frozen=True)
class Page:
    """Texte d'une page d'un document (numérotation à partir de 1)."""

    document: str
    number: int
    text: str


@dataclass(frozen=True)
class Article:
    """Article d'un règlement, avec sa position dans le PDF."""

    document: str
    number: int
    title: str
    chapter: str | None
    start_page: int
    end_page: int
    text: str


def extract_pages(path: Path, document: str) -> list[Page]:
    """Lit un PDF et renvoie le texte brut de chaque page."""
    with pymupdf.open(path) as pdf:
        return [
            Page(document=document, number=index + 1, text=normalize_text(page.get_text("text", sort=True)))
            for index, page in enumerate(pdf)
        ]


def detect_boilerplate(pages: list[Page], min_ratio: float = 0.5) -> set[str]:
    """Repère les en-têtes et pieds de page répétés.

    Une ligne est considérée comme répétée si, une fois les chiffres masqués,
    elle apparaît en haut ou en bas d'au moins `min_ratio` des pages.
    """
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_mask_digits(line) for line in _edge_lines(page.text) if not ARTICLE_HEADING.match(line)})
    threshold = max(2, min_ratio * len(pages))
    return {line for line, count in counts.items() if count >= threshold}


def clean_pages(pages: list[Page], boilerplate: set[str] | None = None) -> list[Page]:
    """Retire les en-têtes et pieds de page de chaque page."""
    if boilerplate is None:
        boilerplate = detect_boilerplate(pages)
    return [Page(page.document, page.number, _remove_boilerplate(page.text, boilerplate)) for page in pages]


def split_articles(pages: list[Page]) -> list[Article]:
    """Découpe le texte nettoyé en articles numérotés.

    Un titre « Article N » n'est retenu que s'il suit le numéro attendu, ce qui
    écarte les articles cités dans le texte (par exemple dans les dispositions
    modifiant d'autres règlements).
    """
    articles: list[Article] = []
    current: dict | None = None
    chapter: str | None = None
    expected = 1
    collecting = False

    for page in pages:
        for line in _lines(page.text):
            heading = ARTICLE_HEADING.match(line)
            if heading and _article_number(heading.group(1)) == expected:
                if current:
                    articles.append(_build_article(current))
                current = {
                    "document": page.document,
                    "number": expected,
                    "title": None,
                    "chapter": chapter,
                    "start_page": page.number,
                    "end_page": page.number,
                    "lines": [],
                }
                expected += 1
                collecting = True
            elif CHAPTER_HEADING.match(line):
                chapter = line
                collecting = False
            elif SECTION_HEADING.match(line):
                collecting = False
            elif END_OF_ARTICLES.match(line) and current:
                collecting = False
            elif current and collecting:
                if current["title"] is None:
                    current["title"] = line
                else:
                    current["lines"].append(line)
                current["end_page"] = page.number

    if current:
        articles.append(_build_article(current))
    return articles


def normalize_text(text: str) -> str:
    """Recolle les mots coupés par un trait d'union conditionnel et uniformise les espaces."""
    text = SOFT_HYPHEN_BREAK.sub("", text).replace("\u00ad", "")
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def load_document(source: SourceDocument) -> list[Page]:
    """Charge un document du corpus et renvoie ses pages nettoyées."""
    if not source.path.exists():
        raise FileNotFoundError(f"{source.path} introuvable. Téléchargez-le depuis {source.url}")
    return clean_pages(extract_pages(source.path, source.key))


def load_corpus() -> dict[str, list[Page]]:
    """Charge tous les documents déclarés dans la configuration."""
    return {key: load_document(source) for key, source in DOCUMENTS.items()}


def _build_article(data: dict) -> Article:
    return Article(
        document=data["document"],
        number=data["number"],
        title=data["title"] or "",
        chapter=data["chapter"],
        start_page=data["start_page"],
        end_page=data["end_page"],
        text="\n".join(data["lines"]),
    )


def _article_number(value: str) -> int:
    return 1 if value in {"premier", "1er"} else int(value)


def _mask_digits(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip()


def _lines(text: str) -> list[str]:
    return [MULTIPLE_SPACES.sub(" ", line).strip() for line in text.splitlines() if line.strip()]


def _edge_lines(text: str) -> list[str]:
    lines = _lines(text)
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def _remove_boilerplate(text: str, boilerplate: set[str]) -> str:
    lines = _lines(text)
    last = len(lines) - EDGE_LINES
    kept = [
        line
        for index, line in enumerate(lines)
        if not ((index < EDGE_LINES or index >= last) and _mask_digits(line) in boilerplate)
    ]
    return "\n".join(kept)
