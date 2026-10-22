"""Vérifie que les documents du corpus sont présents et lisibles."""

import sys

import pymupdf

from assistant_rag.config import DOCUMENTS
from assistant_rag.extraction import load_document, split_articles


def main() -> int:
    errors = 0
    print(f"{'Document':<10}{'Taille (Mo)':>13}{'Pages':>8}{'Articles':>11}  Statut")

    for source in DOCUMENTS.values():
        if not source.path.exists():
            print(f"{source.name:<10}{'-':>13}{'-':>8}{'-':>11}  absent : {source.url}")
            errors += 1
            continue

        size_mb = source.path.stat().st_size / 1_000_000
        try:
            with pymupdf.open(source.path) as pdf:
                page_count = pdf.page_count
        except Exception:
            print(f"{source.name:<10}{size_mb:>13.2f}{'-':>8}{'-':>11}  PDF illisible")
            errors += 1
            continue

        articles = split_articles(load_document(source))
        ok = len(articles) == source.expected_articles
        status = "OK" if ok else f"{source.expected_articles} articles attendus"
        print(f"{source.name:<10}{size_mb:>13.2f}{page_count:>8}{len(articles):>11}  {status}")
        errors += not ok

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
