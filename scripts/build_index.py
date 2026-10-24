"""Construit l'index vectoriel du corpus et le sauvegarde dans storage/."""

import argparse
import time

from assistant_rag.config import Settings
from assistant_rag.indexing import build_embed_model, build_index, load_corpus_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconstruit l'index même s'il existe déjà")
    args = parser.parse_args()

    settings = Settings()
    if settings.index_dir.exists() and not args.force:
        print(f"Index déjà présent : {settings.index_dir}")
        return

    nodes = load_corpus_chunks(settings)
    print(f"{len(nodes)} chunks à indexer avec {settings.embed_model}")
    start = time.perf_counter()
    build_index(nodes, build_embed_model(settings), settings.index_dir)
    print(f"Index sauvegardé dans {settings.index_dir} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
