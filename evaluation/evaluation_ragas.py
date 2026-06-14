"""
Évaluation RAGAS du RAG RGPD.

Charge le RAG (backend), génère ses réponses sur le gold set (gold_set.json),
puis score chaque réponse sur 4 métriques RAGAS (juge = Mistral).

Pipeline :
  1. Charger le gold set (gold_set.json)
  2. Générer les réponses du RAG (run_rag)
  3. Construire le juge (Mistral + embeddings HF)
  4. Scorer chaque question sur 4 métriques
  5. Exporter resultats_ragas.csv + afficher les moyennes

Lancement :  python evaluation/evaluation_ragas.py
"""
import os
import sys
import json
import time
import asyncio
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")          # masque les warnings de dépréciation LangChain
sys.stdout.reconfigure(encoding="utf-8")   # console Windows en UTF-8 (emojis)
sys.stderr.reconfigure(encoding="utf-8")

# ── Chemins & accès au backend ──────────────────────────────────────
# Ce fichier vit dans evaluation/ ; la racine projet est un cran au-dessus.
# On bascule le cwd vers backend/ pour que les chemins relatifs de config.py
# (data/...) résolvent et que `app...` soit importable.
ROOT    = Path(__file__).parent.parent
BACKEND = ROOT / "backend"

# Gold set à évaluer : 1er argument CLI, sinon gold_set.json par défaut.
#   python evaluation/evaluation_ragas.py                 → gold_set.json
#   python evaluation/evaluation_ragas.py gold_set2.json  → gold_set2.json
# Le CSV de sortie est nommé d'après le gold set (résultats séparés par jeu).
_gold_name    = sys.argv[1] if len(sys.argv) > 1 else "gold_set.json"
GOLD_SET_PATH = str(Path(__file__).parent / _gold_name)
CSV_PATH      = str(Path(__file__).parent / f"resultats_{Path(_gold_name).stem}.csv")

load_dotenv(ROOT / ".env")          # MISTRAL_API_KEY chargée AVANT le chdir
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.container       import init_container, get_pipeline, get_retriever
from app.services.pipeline    import RGPD_PROMPT, _format_parents
from app.services.vectorstore import build_embedding_model
from app.core.config          import get_settings


# ── Chargement du RAG ───────────────────────────────────────────────
print("⏳  Initialisation du RAG (HTML + Chroma + embeddings + reranker)...")
t0 = time.perf_counter()
init_container()
pipeline  = get_pipeline()
retriever = get_retriever()
print(f"✅  RAG prêt en {time.perf_counter() - t0:.1f}s")


def run_rag(question: str) -> tuple[str, list[str]]:
    """Une traversée du pipeline RAG → (réponse, contextes bruts récupérés)."""
    parents  = retriever.retrieve_parents(question)
    reranked = retriever.rerank(question, parents)
    contexts = [d.page_content for d in reranked]

    prompt = RGPD_PROMPT.format(context=_format_parents(reranked), question=question)
    result = pipeline.llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)

    marker = "Réponse :"
    if marker in answer:
        answer = answer[answer.rfind(marker) + len(marker):].strip()
    return answer, contexts


# ── 1. GOLD SET (chargé depuis gold_set.json) ───────────────────────
with open(GOLD_SET_PATH, encoding="utf-8") as f:
    gold_set = json.load(f)
print(f"📋  Gold set : {len(gold_set)} questions chargées ({Path(GOLD_SET_PATH).name})")


# ── 2. Génération des réponses du RAG ───────────────────────────────
print(f"\n{'='*70}\n  Génération des réponses RAG sur {len(gold_set)} questions\n{'='*70}")
dataset = []
for i, item in enumerate(gold_set, 1):
    t0 = time.perf_counter()
    answer, contexts = run_rag(item["question"])
    dt = time.perf_counter() - t0
    print(f"   {i:>2}/{len(gold_set)}  ({dt:>4.1f}s, {len(contexts)} ctx)  {item['question'][:55]}...")
    dataset.append({
        "user_input":         item["question"],
        "retrieved_contexts": contexts,
        "response":           answer,
        "reference":          item["reference"],
    })


# ── 3. Construction du juge (Mistral + embeddings) ──────────────────
from ragas.llms       import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI

# Instance dédiée au juge : 4096 tokens (le JSON structuré que RAGAS demande
# au juge est long) et température nulle (sorties déterministes = JSON parsable).
judge_llm = ChatMistralAI(
    mistral_api_key=get_settings().MISTRAL_API_KEY,
    model="mistral-small-latest",
    max_tokens=4096,
    temperature=0.0,
)

# Compat langchain_mistralai 1.1.5 : sa fusion des token_usage (multi-générations
# de ResponseRelevancy) fait `dict += dict` et lève une TypeError. On la remplace
# par une fusion récursive qui sait additionner des champs imbriqués.
def _safe_combine_llm_outputs(self, llm_outputs):
    overall: dict = {}
    def _add(acc, d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                acc[k] = acc.get(k, {})
                _add(acc[k], v)
            elif isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + v
            else:
                acc[k] = v
    for output in llm_outputs:
        _add(overall, (output or {}).get("token_usage", {}))
    return {"token_usage": overall, "model_name": getattr(self, "model", None)}

import types
judge_llm._combine_llm_outputs = types.MethodType(_safe_combine_llm_outputs, judge_llm)

evaluator_llm        = LangchainLLMWrapper(judge_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    build_embedding_model(get_settings().embedding_model)
)
print("\n✅  Juge RAGAS prêt (Mistral + embeddings HF)")


# ── 4. Scoring des 4 métriques ──────────────────────────────────────
from ragas         import SingleTurnSample
from ragas.metrics import (
    Faithfulness,                       # GÉNÉRATION : réponse fondée sur les contextes ?
    ResponseRelevancy,                  # GÉNÉRATION : réponse alignée sur la question ?
    LLMContextPrecisionWithReference,   # RÉCUPÉRATION : bons docs en tête de liste ?
    LLMContextRecall,                   # RÉCUPÉRATION : tous les docs utiles récupérés ?
)

metrics = [
    ("faithfulness",                         Faithfulness(llm=evaluator_llm)),
    ("answer_relevancy",                     ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)),
    ("llm_context_precision_with_reference", LLMContextPrecisionWithReference(llm=evaluator_llm)),
    ("context_recall",                       LLMContextRecall(llm=evaluator_llm)),
]


def score(metric, sample: SingleTurnSample) -> float | None:
    """Score d'une métrique sur un échantillon.

    Appel séquentiel via asyncio.run() (contexte sync → vraie Task asyncio,
    requis par les métriques RAGAS). On évite ragas.evaluate() dont l'executor
    parallèle est instable sur Python 3.14 + Windows.
    """
    try:
        return asyncio.run(metric.single_turn_ascore(sample))
    except Exception as e:
        print(f"      [skip] {metric.__class__.__name__}: {e}")
        return None


print("\n⏳  Évaluation RAGAS en cours...")
t0 = time.perf_counter()
for i, row in enumerate(dataset, 1):
    print(f"\n  Q{i:>2}/{len(dataset)} : {row['user_input'][:60]}...")
    sample = SingleTurnSample(
        user_input         = row["user_input"],
        response           = row["response"],
        retrieved_contexts = row["retrieved_contexts"],
        reference          = row["reference"],
    )
    for name, metric in metrics:
        row[name] = score(metric, sample)
        print(f"    {name:<40} {row[name]:.3f}" if row[name] is not None
              else f"    {name:<40}   NaN")

print(f"\n✅  Évaluation finie en {time.perf_counter() - t0:.1f}s")


# ── 5. Export + moyennes ────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(dataset)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\n📄  Résultats détaillés -> {CSV_PATH} ({len(df)} lignes)")

print("\n" + "=" * 70)
print("  SCORES MOYENS")
print("=" * 70)
for name, _ in metrics:
    series = df[name].dropna()
    if len(series):
        print(f"    {name:<40} {series.mean():.3f}   (sur {len(series)}/{len(df)} questions)")
    else:
        print(f"    {name:<40} aucune note calculée")
