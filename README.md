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

## Fonctionnement du RAG simple

1. **Découpage** : chaque article forme un chunk ; les articles de plus de 512 tokens sont redécoupés par phrases, avec un chevauchement de 64 tokens. Chaque chunk commence par le titre de son article et conserve le document, le numéro d'article et les pages.
2. **Indexation** : les chunks sont vectorisés avec le modèle d'embeddings multilingue `bge-m3`, puis l'index est sauvegardé dans `storage/`.
3. **Recherche** : les 5 chunks les plus proches de la question sont retrouvés.
4. **Génération** : `qwen3:4b-instruct` rédige la réponse à partir de ces seuls passages et cite chaque affirmation par un numéro [n]. Si les passages ne contiennent pas la réponse, le système l'indique au lieu d'inventer.

Les modèles tournent en local avec Ollama, sur processeur, sans carte graphique.

## Structure du projet

```
assistant-rag-agents/
├── data/raw/                  Documents sources (PDF)
├── notebooks/
│   ├── 01_exploration_documents.ipynb
│   └── 02_rag_simple.ipynb
├── scripts/
│   ├── check_documents.py     Contrôle de la présence et du découpage des documents
│   ├── build_index.py         Construction de l'index vectoriel
│   └── ask.py                 Question en ligne de commande
├── src/assistant_rag/
│   ├── config.py              Chemins, corpus et paramètres
│   ├── extraction.py          Extraction du texte, nettoyage, découpage en articles
│   ├── chunking.py            Découpage des articles en chunks avec métadonnées
│   ├── indexing.py            Construction et chargement de l'index
│   ├── prompts.py             Prompts de génération
│   ├── citations.py           Lecture des citations [n]
│   └── rag.py                 RAG simple
└── tests/                     Tests unitaires (pytest)
```

## Installation

Prérequis : Python 3.11 ou plus récent et [Ollama](https://ollama.com/download).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements-dev.txt
pip install -e .
ollama pull qwen3:4b-instruct
ollama pull bge-m3
```

## Utilisation

```bash
python scripts/check_documents.py
python scripts/build_index.py
python scripts/ask.py "Quelles sont les conditions du droit à l'effacement ?" --show-context
pytest
```

Les paramètres se modifient par variables d'environnement : `LLM_MODEL`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `OLLAMA_BASE_URL`.

## Stack technique

Python · PyMuPDF · pandas · LlamaIndex · LangGraph · Ollama · MLflow · FastAPI · Streamlit · Docker · GitHub Actions

## Feuille de route

- [x] Corpus, extraction du texte et découpage en articles
- [x] RAG simple avec citation des sources
- [ ] Jeu de questions de test et évaluation de référence
- [ ] Système multi-agents (chercheur, rédacteur, vérificateur)
- [ ] Comparaison RAG simple / multi-agents
- [ ] Suivi des expériences avec MLflow
- [ ] API FastAPI et interface Streamlit
- [ ] Conteneurisation Docker et intégration continue
