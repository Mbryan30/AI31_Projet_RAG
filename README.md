# AI31_Projet_RAG — Système RAG sur le RGPD

Projet réalisé dans le cadre de l'UV **LO17 / AI31 — Indexation et Recherche d'Information** à l'UTC (Printemps 2026).

## Description

Système de **Retrieval Augmented Generation (RAG)** sur le corpus du **RGPD**. On pose une question en français, le système récupère les articles pertinents dans une base vectorielle, puis un LLM rédige une réponse structurée et **fondée uniquement sur les articles récupérés** (anti-hallucination).

Deux architectures cohabitent et sont **comparées objectivement avec RAGAS** :

1. **RAG avancé** (baseline) : multi-query → parent/child → reranking → génération.
2. **RAG agentique multi-agents** (LangGraph) : planner/routeur → researcher (recherche hybride + outils) → grader CRAG → rewriter → writer → critic/reviser (**Reflection / Reflexion**).

Tout tourne **en local** avec des modèles **Ollama** (aucune clé API nécessaire). Détails : **[RAG_AGENTIQUE.md](RAG_AGENTIQUE.md)**.

Le projet est découpé en trois briques indépendantes :

| Brique | Rôle | Techno |
|---|---|---|
| **`backend/`** | API REST qui orchestre le pipeline RAG (baseline + agentique) | FastAPI + LangChain + LangGraph + ChromaDB + Ollama |
| **`rag-frontend/`** | Interface de chat | React 18 + Vite + Redux + Tailwind |
| **`evaluation/`** | Mesure objective et comparative de la qualité du RAG | RAGAS (juge = LLM local Ollama) |
| **`scripts_windows/`** | Installation des modèles, évaluation, publication | scripts `.bat` |

## Architecture du dépôt

```
AI31_Projet_RAG/
├── .env                       # Configuration locale (modèles Ollama, clés optionnelles) — NON versionné
├── RAG_AGENTIQUE.md           # Analyse, conception et évaluation du RAG agentique
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
│           ├── llm.py         # Fabrique LLM : Ollama (défaut), Mistral, Gemini + juge RAGAS
│           ├── retriever.py   # Multi-query + parent recovery + rerank
│           ├── pipeline.py    # smart_rag (routeur) + full_rag ; mode="agentic" → graphe
│           └── agentic/       # RAG agentique multi-agents (LangGraph)
│               ├── graph.py   # graphe : planner, researcher, grader, rewriter, writer, critic, reviser
│               ├── prompts.py # prompts + schémas Pydantic de chaque agent
│               └── tools.py   # outils : recherche hybride BM25+dense (RRF), get_article, rerank
│
├── rag-frontend/              # ── Interface React ──────────────────────
│   └── src/                   # (voir rag-frontend/README.md pour le détail)
│
├── evaluation/                # ── Évaluation RAGAS ─────────────────────
│   ├── evaluation_ragas.py    # Script : charge le RAG, génère, score (baseline seule)
│   ├── evaluation_comparative.py # Baseline vs agentique, même LLM, même juge (reprenable)
│   ├── results/               # Comparaisons (md, png, csv) générées
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
  1. multi-query   : le LLM reformule la Q en 4 variantes     smart_rag() route :
  2. recherche     : top-k chunks par variante              • « article N » → lookup direct (sans LLM)
  3. parent recovery : on remonte aux articles complets      • titre de chapitre → lookup chapitre
  4. rerank        : cross-encoder garde les 3 meilleurs     • sinon → full_rag (RAG sémantique)
              │                                                         │
              └─────────────► LLM (Ollama) + prompt RGPD structuré ◄────┘
                                       │
                          Réponse en 5 sections + sources + métriques
```

**Idée clé — parent/child retrieval** : on embarque de *petits* chunks (précis pour la recherche sémantique) mais on renvoie au LLM l'*article entier* (contexte complet). On combine précision de recherche et richesse du contexte.

## Prérequis

- **Python 3.11 ou 3.12** recommandé (⚠️ Python 3.14 pose des soucis de compilation sur certaines dépendances — voir Notes)
- **Node.js 18+** (testé avec v24)
- **Ollama** : https://ollama.com/download, puis `ollama pull qwen2.5:7b` (RAG) et `ollama pull qwen2.5:14b` (juge RAGAS) — ou simplement `scripts_windows\1_installer.bat`
- (Optionnel) une clé API Mistral pour utiliser le fournisseur cloud historique (`LLM_PROVIDER=mistral`)

## Installation & lancement

### 1. Configuration

Aucun réglage obligatoire : par défaut le RAG utilise Ollama en local. Pour changer de modèle, créer/compléter `.env` à la racine :

```
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
JUDGE_MODEL=qwen2.5:14b
# MISTRAL_API_KEY=...   (seulement si LLM_PROVIDER=mistral)
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
- `POST /query` — réponse JSON standard (`"mode": "agentic"` → RAG agentique, avec la `trace` des agents)
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
# Pas besoin de lancer le backend : les scripts chargent le RAG en mémoire.
# Comparaison baseline vs agentique (même LLM, même juge) :
python evaluation/evaluation_comparative.py --gold gold_set.json gold_set2.json
python evaluation/evaluation_comparative.py --limit 3        # test rapide

# Baseline seule (script historique) :
python evaluation/evaluation_ragas.py gold_set2.json
```

Sous Windows : `scripts_windows\TOUT_LANCER.bat` enchaîne installation, évaluation et publication.
Résultats : `evaluation/results/comparaison_<gold>.md` (+ `.png`, `.csv`).

## Évaluation : les 4 métriques RAGAS

On évalue le RAG sur 10 questions RGPD (8 conceptuelles + 2 pointues : montant des amendes, responsable vs sous-traitant). Le « juge » est un **LLM local plus grand** (`qwen2.5:14b` via Ollama, température 0). Il note **les deux systèmes de la même façon**, ce qui rend la comparaison équitable.

| Phase | Métrique | Question posée |
|---|---|---|
| **Récupération** | Context Precision | Les bons articles sont-ils en haut du classement ? |
| **Récupération** | Context Recall | A-t-on récupéré tous les articles nécessaires ? |
| **Génération** | Faithfulness | La réponse est-elle fondée sur les articles (pas d'hallucination) ? |
| **Génération** | Response Relevancy | La réponse colle-t-elle à la question ? |

## Notes techniques (pièges rencontrés)

- **IDs déterministes** : `indexing.py` utilise `chapitre::article` comme `parent_id` (et non un UUID aléatoire), sinon les chunks de ChromaDB pointent vers des parents introuvables après redémarrage → 0 contexte récupéré.
- **Scoring séquentiel** : l'executor parallèle de RAGAS (`ragas.evaluate`) est instable sur Python 3.14 + Windows. `evaluation_ragas.py` score donc chaque métrique une par une via `asyncio.run()`.
- **Juge ≠ LLM applicatif** : l'éval utilise un modèle juge dédié (`qwen2.5:14b`), à `num_predict=4096` (JSON structuré long demandé par RAGAS) et température 0 pour des sorties déterministes.
- **Modèles locaux lents** : `evaluation_comparative.py` sauvegarde générations et notes au fil de l'eau (`evaluation/results/*.json`) ; relancer la commande reprend le calcul là où il s'était arrêté.
- **Sorties structurées des agents** : chaque agent déclare un schéma Pydantic, converti en JSON Schema et imposé par Ollama (`format`). En cas d'échec, l'agent réessaie puis applique un repli déterministe.
- **Console UTF-8** : `sys.stdout.reconfigure(encoding="utf-8")` en tête de script, sinon les emojis des `print` plantent sur la console Windows (cp1252).

## Technologies

- **LangChain** — orchestration du pipeline
- **Ollama** (`qwen2.5:7b` pour le RAG, `qwen2.5:14b` pour le juge) — LLM locaux ; Mistral AI reste disponible en option
- **LangGraph** — orchestration du RAG agentique multi-agents (état partagé, boucles CRAG et Reflexion)
- **rank-bm25** — recherche lexicale pour le retrieval hybride
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
