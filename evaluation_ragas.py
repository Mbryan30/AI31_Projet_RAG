"""
================================================================================
  evaluation_ragas.py  —  Évaluation du RAG RGPD avec RAGAS (100% local)
================================================================================

Objectif : mesurer objectivement la qualité du RAG (consigne « Évaluation du
RAG + Gestion des hallucinations ») avec 4 métriques qui se rangent dans les
3 phases du cours :

    RÉCUPÉRATION   ->  Context Precision   (le reranker classe-t-il bien ?)
                       Context Recall      (a-t-on récupéré le nécessaire ?)
    GÉNÉRATION     ->  Faithfulness        (hallucinations : réponse fondée ?)
                       Response Relevancy  (la réponse colle-t-elle à la Q ?)

Particularité : tout tourne en local. Le « juge » LLM est Qwen, et les
métriques à base d'embeddings utilisent le modèle HF. Aucune clé OpenAI.

Installation (dans ton venv) :
    pip install ragas datasets
  ⚠ Si Python 3.14 pose problème (pydantic / datasets), bascule sur un venv
    Python 3.11 ou 3.12 — c'est le même risque de compat que pour torch/chromadb.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

# nest_asyncio : RAGAS tourne en asynchrone ; sur Windows ça évite les
# conflits de boucle d'événements.
import nest_asyncio
nest_asyncio.apply()

import pandas as pd

# --- On réutilise TON pipeline déjà construit dans reponse.py ----------------
# Importer reponse.py charge le modèle Qwen + les embeddings (c'est voulu : on
# en a besoin). On récupère aussi model_hf/tokenizer pour fabriquer un juge.
from reponse import (
    full_rag_pipeline,     # le chemin RAG sémantique complet
    embedding_model,       # HuggingFaceEmbeddings (paraphrase-multilingual...)
    model_hf,              # le modèle Qwen déjà chargé en mémoire
    tokenizer,            # son tokenizer
    LocalLLMWrapper,       # ta classe wrapper LangChain
)

# =============================================================================
# 1. LE JUGE LLM  (décodage glouton = plus fiable pour produire du JSON)
# =============================================================================
# RAGAS demande au juge d'extraire des "claims" et de répondre en JSON structuré.
# Avec do_sample=False (greedy, déterministe) le modèle produit un JSON beaucoup
# plus stable qu'avec temperature=0.3. On REUTILISE les poids déjà en mémoire,
# donc pas de rechargement coûteux.
from transformers import pipeline

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

judge_pipe = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer,
    max_new_tokens=1024,        # extraction de claims = sorties parfois longues
    do_sample=False,            # GLOUTON : déterministe -> JSON plus fiable
    repetition_penalty=1.1,
    eos_token_id=[tokenizer.eos_token_id, im_end_id],
    pad_token_id=tokenizer.eos_token_id,
)
judge_llm = LocalLLMWrapper(hf_pipeline=judge_pipe, hf_tokenizer=tokenizer)

# --- Wrappers RAGAS : on emballe TON llm et TES embeddings -------------------
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm        = LangchainLLMWrapper(judge_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

print("✅  Juge RAGAS prêt (Qwen local, greedy)")

# =============================================================================
# 2. LE JEU DE TEST  (gold set)
# =============================================================================
# Questions CONCEPTUELLES -> elles passent par full_rag_pipeline (génération).
# La 'reference' est la vérité terrain : VÉRIFIE-LA contre tes articles réels
# et ajuste le texte pour qu'il colle à ton corpus. C'est ce qui rend
# Context Recall et la justesse factuelle crédibles.
gold_set = [
    {
        "question": "Quels sont les principes relatifs au traitement des données personnelles ?",
        "reference": "L'article 5 du RGPD pose : licéité, loyauté et transparence ; "
                     "limitation des finalités ; minimisation des données ; exactitude ; "
                     "limitation de la conservation ; intégrité et confidentialité ; "
                     "et responsabilité (accountability).",
    },
    {
        "question": "Qu'est-ce que le consentement au sens du RGPD ?",
        "reference": "Le consentement est une manifestation de volonté libre, spécifique, "
                     "éclairée et univoque (art. 4-11). L'article 7 ajoute qu'il doit pouvoir "
                     "être retiré aussi facilement qu'il a été donné.",
    },
    {
        "question": "Quels sont les droits de la personne concernée ?",
        "reference": "Le chapitre III prévoit notamment : droit d'accès (art. 15), de "
                     "rectification (16), à l'effacement / droit à l'oubli (17), à la "
                     "limitation (18), à la portabilité (20) et d'opposition (21).",
    },
    {
        "question": "Quelles sont les obligations en cas de violation de données personnelles ?",
        "reference": "En cas de violation (art. 4-12), le responsable de traitement notifie "
                     "l'autorité de contrôle dans les 72 heures (art. 33) et, si le risque "
                     "pour les personnes est élevé, les en informe (art. 34).",
    },
    {
        "question": "Quand une analyse d'impact relative à la protection des données est-elle obligatoire ?",
        "reference": "L'article 35 impose une analyse d'impact (AIPD) lorsqu'un traitement "
                     "est susceptible d'engendrer un risque élevé pour les droits et "
                     "libertés des personnes physiques.",
    },
    {
        "question": "Quelles sont les bases légales possibles d'un traitement ?",
        "reference": "L'article 6 prévoit six bases : consentement, exécution d'un contrat, "
                     "obligation légale, sauvegarde des intérêts vitaux, mission d'intérêt "
                     "public, et intérêts légitimes du responsable.",
    },
    {
        "question": "Quand la désignation d'un délégué à la protection des données est-elle obligatoire ?",
        "reference": "Les articles 37 à 39 imposent un DPO pour les autorités publiques, en "
                     "cas de suivi régulier et systématique à grande échelle, ou de traitement "
                     "à grande échelle de catégories particulières de données.",
    },
    {
        "question": "À quelles conditions peut-on transférer des données hors de l'Union européenne ?",
        "reference": "Le chapitre V (art. 44 à 49) autorise les transferts sur décision "
                     "d'adéquation, ou avec des garanties appropriées (clauses contractuelles "
                     "types, BCR), ou par dérogations pour situations spécifiques.",
    },
]

# =============================================================================
# 3. EXÉCUTION DU RAG  ->  on collecte les 4 champs attendus par RAGAS
# =============================================================================
print(f"\n⏳  Génération des réponses sur {len(gold_set)} questions...")

dataset = []
for i, item in enumerate(gold_set, 1):
    print(f"   {i}/{len(gold_set)}  {item['question'][:55]}...")
    out = full_rag_pipeline(item["question"])   # <-- ton pipeline RAG complet
    dataset.append({
        "user_input":         item["question"],
        "retrieved_contexts": out["contexts"],   
        "response":           out["reponse"],
        "reference":          item["reference"],
    })

# =============================================================================
# 4. ÉVALUATION RAGAS
# =============================================================================
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,                      # GÉNÉRATION : hallucinations
    ResponseRelevancy,                 # GÉNÉRATION : pertinence (embeddings)
    LLMContextPrecisionWithReference,  # RÉCUPÉRATION : qualité du classement
    LLMContextRecall,                  # RÉCUPÉRATION : couverture
)

evaluation_dataset = EvaluationDataset.from_list(dataset)

metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

# max_workers BAS : un modèle local sur CPU ne supporte pas 16 requêtes en
# parallèle. timeout généreux car le juge est lent.
run_config = RunConfig(max_workers=2, timeout=300, max_retries=3)

print("\n⏳  Évaluation RAGAS en cours (long sur CPU local)...")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
)

# =============================================================================
# 5. RÉSULTATS
# =============================================================================
print("\n" + "=" * 60)
print("  SCORES GLOBAUX (moyenne)")
print("=" * 60)
print(result)

df = result.to_pandas()
# Détail par question (utile pour le rapport : on voit où ça pèche)
df.to_csv("resultats_ragas.csv", index=False, encoding="utf-8-sig")
print("\n📄  Détail par question sauvegardé -> resultats_ragas.csv")

# Moyennes robustes (on ignore les NaN si le juge a parfois échoué à parser)
print("\n  Moyennes (NaN ignorés) :")
for col in df.columns:
    if df[col].dtype.kind in "fc":   # colonnes numériques
        print(f"    {col:<35} {df[col].mean(skipna=True):.3f}")