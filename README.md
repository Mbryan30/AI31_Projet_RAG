# AI31_Projet_RAG

Projet réalisé dans le cadre de l'UV **LO17 / AI31 — Indexation et Recherche d'Information** à l'UTC (Printemps 2026).

## Description

Ce projet implémente un système de **Retrieval Augmented Generation (RAG)** sur le thème du Seigneur des Anneaux. Le RAG permet de poser des questions en français et d'obtenir des réponses précises basées sur un corpus de documents, en combinant la recherche d'information et la génération de texte via un LLM (Gemini).

## Technologies utilisées

- **LangChain** — orchestration du pipeline RAG
- **Google Gemini** — modèle de langage (LLM) et modèle d'embedding
- **ChromaDB** — base de données vectorielle
- **Streamlit** — interface utilisateur
- **Python 3.11+**

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/AI31_Projet_RAG.git
cd AI31_Projet_RAG
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

Activation :

- **Windows (PowerShell)** : `venv\Scripts\activate`
- **Windows (Git Bash)** : `source venv/Scripts/activate`
- **macOS / Linux** : `source venv/bin/activate`

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

Créer un fichier `.env` à la racine du projet :

```
GOOGLE_API_KEY=votre_clé_api_ici
```

Pour obtenir une clé API gratuite : [Google AI Studio](https://aistudio.google.com) → Get API Key → Create API key.

> ⚠️ Ne partagez jamais votre clé API. Le fichier `.env` est exclu du dépôt via `.gitignore`.

## Utilisation

### Notebook

Le fichier `Gemini_LangChain_QA_Chroma_WebLoad.ipynb` contient le pipeline RAG complet. Ouvrir avec VS Code ou Jupyter Notebook et exécuter les cellules dans l'ordre.

### Application Streamlit

```bash
streamlit run app.py
```

## Structure du projet

```
AI31_Projet_RAG/
├── chroma_db/              # Base de données vectorielle (générée automatiquement)
├── documents/              # Documents sources du corpus
├── venv/                   # Environnement virtuel (non versionné)
├── .env                    # Clé API (non versionné)
├── .gitignore
├── Gemini_LangChain_QA_Chroma_WebLoad.ipynb   # Notebook RAG
├── requirements.txt
└── README.md
```

## Équipe

- Membre Bryan MIGUEU
- Membre 2
- Membre 3
- Membre 4

## Licence

Projet universitaire — UTC, Printemps 2026.