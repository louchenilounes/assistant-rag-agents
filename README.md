# Assistant RAG multi-agents

Assistant de questions-réponses sur des textes réglementaires européens (RGPD et AI Act).
Chaque réponse cite ses sources : document, article et page.

Deux approches sont développées et comparées sur un même jeu de questions :

- un **RAG simple** : recherche des passages pertinents, puis génération de la réponse ;
- un **système multi-agents** construit avec LangGraph : un chercheur sélectionne les passages, un rédacteur écrit la réponse, un vérificateur contrôle que chaque affirmation est soutenue par une source.

Le modèle de langage tourne en local avec Ollama : aucune donnée n'est envoyée à un service externe.

## Données

| Document | Référence | Articles | Source |
| --- | --- | --- | --- |
| RGPD | Règlement (UE) 2016/679 | 99 | [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679) |
| AI Act | Règlement (UE) 2024/1689 | 113 | [EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32024R1689) |

Textes en version française. © Union européenne, https://eur-lex.europa.eu. Réutilisation autorisée avec mention de la source.

Les PDF sont lus avec PyMuPDF. Les en-têtes et pieds de page du Journal officiel sont retirés automatiquement, puis chaque règlement est découpé en articles. Chaque article conserve son numéro, son titre, son chapitre et ses pages, ce qui permet de citer les sources avec précision.

## Structure du projet

```
assistant-rag-agents/
├── data/raw/                  Documents sources (PDF)
├── notebooks/
│   └── 01_exploration_documents.ipynb
├── scripts/
│   └── check_documents.py     Contrôle de la présence et du découpage des documents
├── src/assistant_rag/
│   ├── config.py              Chemins et description du corpus
│   └── extraction.py          Extraction du texte, nettoyage, découpage en articles
└── tests/                     Tests unitaires (pytest)
```

## Installation

Prérequis : Python 3.11 ou plus récent.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements-dev.txt
pip install -e .
```

Vérification du corpus et des tests :

```bash
python scripts/check_documents.py
pytest
```

## Stack technique

Python · PyMuPDF · pandas · LlamaIndex · LangGraph · Ollama · MLflow · FastAPI · Streamlit · Docker · GitHub Actions

## Feuille de route

- [x] Corpus, extraction du texte et découpage en articles
- [ ] RAG simple avec citation des sources
- [ ] Jeu de questions de test et évaluation de référence
- [ ] Système multi-agents (chercheur, rédacteur, vérificateur)
- [ ] Comparaison RAG simple / multi-agents
- [ ] Suivi des expériences avec MLflow
- [ ] API FastAPI et interface Streamlit
- [ ] Conteneurisation Docker et intégration continue
