"""Pose une question au RAG simple et affiche la réponse avec ses sources."""

import argparse

from assistant_rag.rag import SimpleRAG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="question en français")
    parser.add_argument("--show-context", action="store_true", help="affiche aussi les passages retrouvés")
    args = parser.parse_args()

    answer = SimpleRAG.from_settings().answer(args.question)
    print(answer.format())
    if args.show_context:
        print("\nPassages retrouvés :")
        for source in answer.sources:
            print(f"[{source.number}] {source.label} (score {source.score:.3f})")
    print(f"\nTemps de réponse : {answer.latency:.1f} s")


if __name__ == "__main__":
    main()
