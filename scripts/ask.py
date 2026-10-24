"""Pose une question à l'assistant et affiche la réponse avec ses sources."""

import argparse

from assistant_rag.agents.graph import AgentsRAG
from assistant_rag.rag import SimpleRAG

SYSTEMS = {"simple": SimpleRAG, "agents": AgentsRAG}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="question en français")
    parser.add_argument("--system", choices=SYSTEMS, default="simple")
    parser.add_argument("--show-context", action="store_true", help="affiche aussi les passages retrouvés")
    parser.add_argument("--show-trace", action="store_true", help="affiche les étapes des agents")
    args = parser.parse_args()

    answer = SYSTEMS[args.system].from_settings().answer(args.question)
    print(answer.format())
    if args.show_context:
        print("\nPassages retrouvés :")
        for source in answer.sources:
            print(f"[{source.number}] {source.label} (score {source.score:.3f})")
    if args.show_trace:
        print("\nÉtapes :")
        for step in answer.trace:
            details = {key: value for key, value in step.items() if key != "agent"}
            print(f"- {step['agent']} : {details}")
    print(f"\nTemps de réponse : {answer.latency:.1f} s ({answer.llm_calls} appel(s) au LLM)")


if __name__ == "__main__":
    main()
