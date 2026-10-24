"""Prompts des agents chercheur, rédacteur et vérificateur."""

AGENTS_PROMPT_VERSION = "agents-v2"

RESEARCHER_PROMPT = """Tu prépares une recherche documentaire dans le RGPD et dans le règlement européen sur l'IA.
Découpe la question en sous-questions uniquement si elle demande explicitement plusieurs informations distinctes,
par exemple deux obligations, deux acteurs ou une règle de chacun des deux règlements. Au maximum trois sous-questions.
Dans tous les autres cas, renvoie une seule sous-question identique à la question.
N'ajoute jamais d'aspect absent de la question, en particulier un règlement que la question ne mentionne pas.

Exemples :
Question : Dans quel délai une violation doit-elle être notifiée à l'autorité de contrôle ?
{"sous_questions": ["Dans quel délai une violation doit-elle être notifiée à l'autorité de contrôle ?"]}
Question : Quelles obligations s'appliquent envers l'autorité de contrôle et envers les personnes concernées ?
{"sous_questions": ["Quelles obligations s'appliquent envers l'autorité de contrôle ?",
                    "Quelles obligations s'appliquent envers les personnes concernées ?"]}

Réponds uniquement avec un objet JSON de la forme {"sous_questions": ["..."]}."""

COVERAGE_INSTRUCTIONS = """Aspects à traiter :
{aspects}
Traite chaque aspect. Si un aspect n'est couvert par aucun extrait, indique-le en une phrase."""

VERIFIER_PROMPT = """Tu vérifies une réponse rédigée à partir d'extraits numérotés du RGPD et de l'AI Act.
Tu ne signales que des erreurs certaines et vérifiables :
- un montant, un pourcentage, un délai, une date ou un seuil absent de l'extrait cité ou différent de celui-ci ;
- une affirmation contredite par l'extrait qu'elle cite, ou attribuée à un extrait qui n'en parle pas ;
- un aspect de la question qui n'est traité par aucun extrait.
Une reformulation, un résumé ou une omission de détail ne sont pas des erreurs.

Verdicts possibles :
- "valide" : aucune erreur certaine.
- "a_corriger" : au moins une affirmation erronée ou attribuée au mauvais extrait.
- "incomplete" : un aspect de la question n'est couvert par aucun extrait ; indique alors les recherches à effectuer.

Réponds uniquement avec un objet JSON de la forme
{"verdict": "...", "problemes": ["..."], "recherches": ["..."]}
où "problemes" décrit chaque erreur en une phrase et "recherches" contient des requêtes courtes en français."""

REVISION_INSTRUCTIONS = """Ta réponse précédente :
{draft}

Problèmes relevés par le vérificateur :
{problems}

Rédige une nouvelle réponse qui corrige ces problèmes, en respectant les mêmes règles."""
