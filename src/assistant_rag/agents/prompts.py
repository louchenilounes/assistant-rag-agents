"""Prompts des agents chercheur et vérificateur."""

AGENTS_PROMPT_VERSION = "agents-v1"

RESEARCHER_PROMPT = """Tu prépares une recherche documentaire dans le RGPD et dans le règlement européen sur l'IA.
Si la question porte sur plusieurs aspects (plusieurs règles, plusieurs acteurs ou les deux règlements),
découpe-la en sous-questions autonomes, au maximum trois.
Sinon, renvoie une seule sous-question identique à la question.
Chaque sous-question doit pouvoir être comprise sans la question d'origine.
Réponds uniquement avec un objet JSON de la forme {"sous_questions": ["...", "..."]}."""

VERIFIER_PROMPT = """Tu vérifies une réponse rédigée à partir d'extraits numérotés du RGPD et de l'AI Act.
Pour chaque phrase, contrôle que le ou les extraits qu'elle cite contiennent bien l'information affirmée,
en particulier les montants, les délais, les dates et les conditions.

Verdicts possibles :
- "valide" : toutes les affirmations sont justifiées par les extraits cités et la réponse traite toute la question.
- "a_corriger" : au moins une affirmation n'est pas justifiée par l'extrait qu'elle cite, ou contient une erreur.
- "incomplete" : un aspect de la question n'est traité par aucun extrait ; indique alors les recherches à effectuer.

Réponds uniquement avec un objet JSON de la forme
{"verdict": "...", "problemes": ["..."], "recherches": ["..."]}
où "problemes" décrit chaque erreur en une phrase et "recherches" contient des requêtes courtes en français."""

REVISION_INSTRUCTIONS = """Ta réponse précédente :
{draft}

Problèmes relevés par le vérificateur :
{problems}

Rédige une nouvelle réponse qui corrige ces problèmes, en respectant les mêmes règles."""
