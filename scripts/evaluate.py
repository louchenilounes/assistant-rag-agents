"""Évalue un système sur le jeu de questions et enregistre les réponses et les métriques dans results/."""

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from assistant_rag.config import RESULTS_DIR, Settings
from assistant_rag.evaluation.dataset import load_questions
from assistant_rag.evaluation.judge import Judge, build_judge_llm
from assistant_rag.evaluation.metrics import summarize
from assistant_rag.evaluation.runner import run_evaluation
from assistant_rag.prompts import PROMPT_VERSION
from assistant_rag.rag import SimpleRAG

SYSTEMS = {"simple": SimpleRAG}


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, default="simple")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--run-name", help="nom du dossier de résultats (par défaut : <système>-<split>)")
    parser.add_argument("--limit", type=int, help="nombre maximal de questions, pour un essai rapide")
    parser.add_argument("--no-judge", action="store_true", help="n'évalue pas l'exactitude avec le LLM-juge")
    args = parser.parse_args()

    settings = Settings()
    questions = load_questions(split=args.split)[: args.limit]
    output_dir = RESULTS_DIR / (args.run_name or f"{args.system}-{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    if not config_path.exists():
        config = {
            "system": args.system,
            "split": args.split,
            "settings": asdict(settings),
            "prompt_version": PROMPT_VERSION,
            "judge": None if args.no_judge else settings.judge_model,
            "git_commit": git_commit(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    system = SYSTEMS[args.system].from_settings(settings)
    judge = None if args.no_judge else Judge(build_judge_llm(settings))
    records = run_evaluation(system, questions, output_dir, judge)

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 2):
        print(summarize(records))
    print(f"Résultats enregistrés dans {output_dir}")


if __name__ == "__main__":
    main()
