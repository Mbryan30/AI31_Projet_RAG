# AI31_Projet_RAG — Système RAG sur le RGPD

Projet réalisé dans le cadre de l'UV **LO17 / AI31 — Indexation et Recherche d'Information** à l'UTC (Printemps 2026).

## Description

Système de **Retrieval Augmented Generation (RAG)** sur le corpus du **RGPD**. On pose une question en français, le système récupère les articles pertinents dans une base vectorielle, puis un LLM rédige une réponse structurée et **fondée uniquement sur les articles récupérés** (anti-hallucination).

Le projet est découpé en trois briques indépendantes :

| Brique | Rôle | Techno |
|---|---|---|
| **`backend/`** | API REST qui orchestre le pipeline RAG | FastAPI + LangChain + ChromaDB + Mistral |
| **`rag-frontend/`** | Interface de chat | React 18 + Vite + Redux + Tailwind |
| **`evaluation/`** | Mesure objective de la qualité du RAG | RAGAS (juge = Mistral) |

## Architecture du dépôt

```
AI31_Projet_RAG/
├── .env                       # Clé API (MISTRAL_API_KEY) — NON versionné
├── .gitignore
├── README.md                  # Documentation du projet (ce fichier)
├── requirements.txt
│
├── backend/                   # ── API FastAPI ──────────────────────────
│   ├── run.py                 # Point d'entrée : uvicorn sur :8000
│   ├── scripts/
│   │   └── build_index.py     # (Re)construit l'index ChromaDB depuis le HTML
│   ├── data/
│   │   ├── L_2016119FR...html  # Texte officiel du RGPD (source)
│   │   └── chroma_db/          # Index vectoriel persisté (généré)
│   └── app/
│       ├── main.py            # create_app() + lifespan (init au démarrage)
│       ├── core/
│       │   ├── config.py      # Settings (modèles, chemins, params retrieval)
│       │   ├── container.py   # Injection de dépendances (singletons)
│       │   └── logging.py
│       ├── api/
│       │   ├── query.py       # POST /query, GET /query/stream, POST /query/filter
│       │   └── health.py      # GET /health, GET /ready
│       ├── schemas/
│       │   └── query.py       # Modèles Pydantic (Request/Response)
│       └── services/
│           ├── indexing.py    # HTML → documents parent/enfant
│           ├── vectorstore.py # Embeddings HF + ChromaDB
│           ├── llm.py         # Client Mistral
│           ├── retriever.py   # Multi-query + parent recovery + rerank
│           └── pipeline.py    # smart_rag (routeur) + full_rag
│
├── rag-frontend/              # ── Interface React ──────────────────────
│   └── src/                   # (voir rag-frontend/README.md pour le détail)
│
├── evaluation/                # ── Évaluation RAGAS ─────────────────────
│   ├── evaluation_ragas.py    # Script : charge le RAG, génère, score
│   ├── gold_set.json          # Jeu de questions principal (8 conceptuelles + 2 pointues)
│   ├── gold_set2.json         # Jeu de questions plus faciles (définitions)
│   └── resultats_*.csv        # Scores par question (généré, un CSV par gold set)
│
└── venv/                      # Environnement virtuel (non versionné)
```

## Pipeline RAG (cœur du système)

```
        Texte RGPD (HTML)
              │
      indexing.py  ── parse → articles (parents) + petits chunks (enfants)
              │
     vectorstore.py ── embeddings multilingues → ChromaDB (persisté)
              │
  ┌───────────┴─────────── Question ──────────────────────┐
  │                                                         │
  retriever.py                                        pipeline.py
  1. multi-query   : Mistral reformule la Q en 4 variantes   smart_rag() route :
  2. recherche     : top-k chunks par variante              • « article N » → lookup direct (sans LLM)
  3. parent recovery : on remonte aux articles complets      • titre de chapitre → lookup chapitre
  4. rerank        : cross-encoder garde les 3 meilleurs     • sinon → full_rag (RAG sémantique)
              │                                                         │
              └─────────────► Mistral + prompt RGPD structuré ◄─────────┘
                                       │
                          Réponse en 5 sections + sources + métriques
```

**Idée clé — parent/child retrieval** : on embarque de *petits* chunks (précis pour la recherche sémantique) mais on renvoie au LLM l'*article entier* (contexte complet). On combine précision de recherche et richesse du contexte.

## Prérequis

- **Python 3.11 ou 3.12** recommandé (⚠️ Python 3.14 pose des soucis de compilation sur certaines dépendances — voir Notes)
- **Node.js 18+** (testé avec v24)
- Une **clé API Mistral** gratuite : https://console.mistral.ai → API Keys

## Installation & lancement

### 1. Configuration

Créer un fichier `.env` à la racine :

```
MISTRAL_API_KEY="votre_clé_mistral"
```

### 2. Backend (API FastAPI)

```bash
python -m venv venv
venv\Scripts\activate            # Windows (PowerShell)
pip install -r requirements.txt

# (Première fois) construire l'index vectoriel :
cd backend
python scripts/build_index.py

# Lancer l'API :
python run.py                    # → http://localhost:8000
```

L'API expose :
- `POST /query` — réponse JSON standard
- `GET  /query/stream` — réponse en streaming (SSE)
- `GET  /health` / `GET /ready` — health checks
- Docs interactives : http://localhost:8000/docs

### 3. Frontend (React)

```bash
cd rag-frontend
npm install
cp .env.example .env             # VITE_API_URL=http://localhost:8000
npm run dev                      # → http://localhost:5173
```

### 4. Évaluation (RAGAS)

```bash
# Pas besoin de lancer le backend : le script charge le RAG en mémoire.
python evaluation/evaluation_ragas.py                 # gold_set.json (par défaut)
python evaluation/evaluation_ragas.py gold_set2.json  # autre jeu de questions
```

Produit `evaluation/resultats_<nom-du-gold-set>.csv` et affiche les 4 scores moyens.

## Évaluation : les 4 métriques RAGAS

On évalue le RAG sur 10 questions RGPD (8 conceptuelles + 2 pointues : montant des amendes, responsable vs sous-traitant). Le « juge » est **Mistral lui-même** (limite assumée : biais d'auto-évaluation — voir rapport).

| Phase | Métrique | Question posée |
|---|---|---|
| **Récupération** | Context Precision | Les bons articles sont-ils en haut du classement ? |
| **Récupération** | Context Recall | A-t-on récupéré tous les articles nécessaires ? |
| **Génération** | Faithfulness | La réponse est-elle fondée sur les articles (pas d'hallucination) ? |
| **Génération** | Response Relevancy | La réponse colle-t-elle à la question ? |

## Notes techniques (pièges rencontrés)

- **IDs déterministes** : `indexing.py` utilise `chapitre::article` comme `parent_id` (et non un UUID aléatoire), sinon les chunks de ChromaDB pointent vers des parents introuvables après redémarrage → 0 contexte récupéré.
- **Scoring séquentiel** : l'executor parallèle de RAGAS (`ragas.evaluate`) est instable sur Python 3.14 + Windows. `evaluation_ragas.py` score donc chaque métrique une par une via `asyncio.run()`.
- **Juge ≠ LLM applicatif** : l'éval utilise une instance Mistral dédiée à `max_tokens=4096` (le LLM du RAG est à 512, trop court pour le JSON structuré que RAGAS demande au juge), à température 0 pour des sorties déterministes.
- **Compat `langchain_mistralai` 1.1.5** : sa fusion des `token_usage` plante (`dict += dict`) quand RAGAS demande plusieurs générations → patchée par une fusion récursive dans `evaluation_ragas.py`.
- **Console UTF-8** : `sys.stdout.reconfigure(encoding="utf-8")` en tête de script, sinon les emojis des `print` plantent sur la console Windows (cp1252).

## Technologies

- **LangChain** — orchestration du pipeline
- **Mistral AI** (`mistral-small-latest`) — LLM de génération et juge d'évaluation
- **ChromaDB** — base vectorielle
- **sentence-transformers** — embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`) + reranker cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **FastAPI** — API REST
- **React + Vite + Redux + Tailwind** — frontend
- **RAGAS** — évaluation

## Équipe

- Bryan MIGUEU
- Membre 2
- Membre 3
- Membre 4

## Licence

Projet universitaire — UTC, Printemps 2026.
