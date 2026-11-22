# ANTISÈCHE — Projet RAG RGPD (pour l'oral)

> Une page à avoir sous les yeux pendant la présentation. Détails complets : `GUIDE_PROJET.md`.

---

## Le projet en 10 phrases

1. On a construit un **système RAG** (Retrieval-Augmented Generation) qui répond à des **questions sur le RGPD** en français.
2. **Problème** : un LLM seul peut **inventer** et ne cite pas ses sources — inacceptable sur un sujet **juridique**.
3. **Solution** : on **récupère d'abord les vrais articles** du RGPD pertinents, puis le LLM répond **uniquement à partir d'eux**.
4. Le **corpus** est le texte officiel du RGPD (Règlement UE 2016/679), soit ~99 articles.
5. **Indexation** : on parse le HTML (BeautifulSoup), on découpe chaque article en petits **chunks**, qu'on **vectorise** (embeddings multilingues) dans **ChromaDB**.
6. **Stratégie parent/child** : on cherche sur de **petits chunks** (précis) mais on renvoie au LLM **l'article entier** (contexte complet).
7. À chaque question : **multi-query** (4 reformulations) → **recherche vectorielle** → **parent recovery** → **reranking** cross-encoder (top 3).
8. Un **« smart router »** traite les questions sur un article/chapitre précis **sans LLM** (rapide, zéro hallucination).
9. Le LLM **Mistral** génère une réponse **structurée en 5 sections**, avec une **règle anti-hallucination** explicite dans le prompt.
10. On **évalue** la qualité avec **RAGAS** (4 métriques) : faithfulness ≈ 0,87, context precision ≈ 0,88, recall ≈ 0,85.

---

## Le pitch en 30 secondes

> « Notre projet répond à des questions juridiques sur le RGPD. Plutôt que de faire confiance à la mémoire d'un LLM — qui hallucine — on **va d'abord chercher les vrais articles** du règlement dans une base vectorielle, puis on demande au modèle de répondre **en s'appuyant uniquement dessus**. On combine recherche sémantique, récupération de l'article complet et re-classement précis pour donner les meilleurs articles au modèle. Résultat : des réponses **fondées, structurées et citées**, qu'on a **mesurées objectivement avec RAGAS**. »

---

## Chiffres clés (à citer)

| Élément | Valeur |
|---|---|
| Articles du RGPD | ~99 |
| Reformulations multi-query | 4 |
| Chunks récupérés / variante (k) | 6 |
| Articles candidats (parents) | ≤ 8 |
| Articles finaux envoyés au LLM | 3 |
| Taille des chunks / chevauchement | 300 / 50 |
| Latence (question sémantique) | ~5 s |
| Faithfulness / Precision / Recall (RAGAS) | ~0,87 / ~0,88 / ~0,85 |

---

## RAG agentique en 5 phrases

1. Un **graphe LangGraph** fait collaborer 7 agents spécialisés via un **état partagé** : planner, researcher, grader, rewriter, writer, critic, reviser.
2. Le **planner** route la question et la **décompose** ; le **researcher** utilise des **outils** (recherche hybride BM25 + dense fusionnée par RRF, accès direct à un article, reranker multilingue).
3. Le **grader** applique **CRAG** : il ne garde que les articles utiles et, si le contexte est insuffisant, le **rewriter** reformule et on recherche à nouveau.
4. Le **critic** applique **Reflection/Self-RAG** (affirmations non fondées, manques, digressions) ; le **reviser** applique **Reflexion** (recherche ciblée + réécriture), boucles bornées.
5. Même LLM local (Ollama `qwen2.5:7b`), même index, même juge (`qwen2.5:14b`) que la baseline → l'écart RAGAS mesure l'apport de l'**orchestration agentique**.

---

## Réponses éclair (si on te demande…)

- **« C'est quoi un RAG ? »** → Chercher d'abord, générer ensuite : on récupère les documents pertinents, puis le LLM répond à partir d'eux.
- **« Hallucinations ? »** → Prompt anti-invention + on ne donne que les articles récupérés + on mesure la fidélité (Faithfulness).
- **« Embedding vs reranker ? »** → L'embedding est rapide mais approximatif (toute la base) ; le reranker est précis mais lent (lit question+doc ensemble) → on cherche vite, on re-classe précis.
- **« Pourquoi Mistral ? »** → Bon en français, clé gratuite, bon compromis qualité/vitesse/coût ; architecture agnostique.
- **« Pourquoi ChromaDB ? »** → Léger, local, sans serveur ; idéal pour ce volume.
- **« Le juge est Mistral, c'est biaisé ? »** → Oui, limite assumée ; atténuée car vérifications factuelles + comparaison à une référence.

---

## ⚠️ À NE PAS dire (pièges)

- ❌ Les modes **CRAG / HyDE / Self-RAG** de l'interface restent des **étiquettes** (même algorithme) ; en revanche le mode **« RAG Agentique »** est réellement implémenté (LangGraph : planner, grader CRAG, rewriter, critic/reviser Reflexion) — voir `RAG_AGENTIQUE.md`.
- ❌ Ne pas dire « **vrai streaming token par token** » → le streaming de l'UI est **simulé** (réponse calculée puis renvoyée mot à mot).
- ❌ Ne pas survendre la couverture → le corpus est **le texte du RGPD seul** (pas la jurisprudence ni les lignes directrices CNIL).
