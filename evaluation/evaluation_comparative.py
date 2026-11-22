"""
Évaluation RAGAS COMPARATIVE : RAG avancé (baseline) vs RAG agentique multi-agents.

Les deux systèmes partagent EXACTEMENT les mêmes briques :
  • même corpus, même index ChromaDB, mêmes embeddings ;
  • même LLM applicatif local (Ollama, par défaut qwen2.5:7b) ;
  • même prompt de rédaction (RGPD_PROMPT) pour le premier jet ;
  • même juge RAGAS local (Ollama, par défaut qwen2.5:14b, température 0).
La seule différence est l'orchestration : pipeline linéaire d'un côté,
graphe LangGraph multi-agents (planner → researcher → grader CRAG → rewriter →
writer → critic/reviser Reflexion) de l'autre. L'écart de score mesure donc
l'apport de l'approche agentique.

Le script est REPRENABLE : générations et notes sont sauvegardées au fil de
l'eau dans evaluation/results/ ; relancer la même commande reprend là où le
calcul s'était arrêté (pratique avec des modèles locaux lents).

Exemples :
  python evaluation/evaluation_comparative.py                       # gold_set.json
  python evaluation/evaluation_comparative.py --gold gold_set2.json
  python evaluation/evaluation_comparative.py --gold gold_set.json gold_set2.json
  python evaluation/evaluation_comparative.py --limit 3             # test rapide
  python evaluation/evaluation_comparative.py --llm-model qwen2.5:7b --judge-model qwen2.5:14b
  python evaluation/evaluation_comparative.py --fresh               # ignore le cache
"""
import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")            # console Windows en UTF-8
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "llm_context_precision_with_reference": "Context precision",
    "context_recall": "Context recall",
}
SYSTEM_LABELS = {"baseline": "RAG avancé (baseline)", "agentic": "RAG agentique (LangGraph)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", nargs="+", default=["gold_set.json"], help="gold set(s) dans evaluation/")
    p.add_argument("--systems", default="baseline,agentic", help="systèmes à évaluer (virgules)")
    p.add_argument("--limit", type=int, default=0, help="n'évaluer que les N premières questions")
    p.add_argument("--llm-model", default=None, help="modèle Ollama du RAG (sinon .env / config)")
    p.add_argument("--judge-model", default=None, help="modèle Ollama du juge RAGAS")
    p.add_argument("--fresh", action="store_true", help="ignorer les résultats en cache")
    p.add_argument("--timeout", type=int, default=900, help="timeout (s) d'un appel du juge")
    return p.parse_args()


ARGS = parse_args()


class _Tee:
    """Duplique la sortie console dans evaluation/results/run_log.txt."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


sys.stdout = _Tee(sys.stdout, RESULTS / "run_log.txt")
print(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv[1:])}")
if ARGS.llm_model:
    os.environ["LLM_MODEL"] = ARGS.llm_model        # lu par pydantic-settings

from dotenv import load_dotenv                       # noqa: E402

load_dotenv(ROOT / ".env")
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)                                    # chemins relatifs de config.py

from app.core.config import get_settings             # noqa: E402
from app.core.container import get_agentic, get_pipeline, get_retriever, init_container  # noqa: E402
from app.services.llm import build_judge_llm, llm_text  # noqa: E402
from app.services.pipeline import RGPD_PROMPT, _format_parents  # noqa: E402
from app.services.vectorstore import build_embedding_model  # noqa: E402

SETTINGS = get_settings()
SYSTEMS = [s.strip() for s in ARGS.systems.split(",") if s.strip()]


# ── utilitaires de cache JSON ───────────────────────────────────────
def _load(path: Path) -> dict:
    if ARGS.fresh or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ── les deux systèmes ───────────────────────────────────────────────
def run_baseline(question: str) -> dict:
    """Pipeline d'origine : multi-query → parents → rerank top-3 → génération."""
    retriever, pipeline = get_retriever(), get_pipeline()
    parents = retriever.retrieve_parents(question)
    reranked = retriever.rerank(question, parents)
    prompt = RGPD_PROMPT.format(context=_format_parents(reranked), question=question)
    answer = llm_text(pipeline.llm.invoke(prompt))
    if "Réponse :" in answer:
        answer = answer[answer.rfind("Réponse :") + len("Réponse :"):].strip()
    return {"response": answer, "retrieved_contexts": [d.page_content for d in reranked],
            "trace": ["baseline: multi-query → parent recovery → rerank → génération"]}


def run_agentic(question: str) -> dict:
    out = get_agentic().run(question)
    return {"response": out["answer"], "retrieved_contexts": out["contexts"],
            "trace": out["trace"]}


RUNNERS = {"baseline": run_baseline, "agentic": run_agentic}


# ── génération ──────────────────────────────────────────────────────
def generate(gold: list[dict], stem: str) -> dict:
    path = RESULTS / f"generations_{stem}.json"
    cache = _load(path)
    if cache.get("_llm") and cache["_llm"] != SETTINGS.resolved_model:
        print(f"⚠️  cache généré avec {cache['_llm']} ≠ {SETTINGS.resolved_model} → régénération")
        cache = {}
    cache["_llm"] = SETTINGS.resolved_model
    for system in SYSTEMS:
        cache.setdefault(system, {})
        print(f"\n{'=' * 72}\n  Génération — {SYSTEM_LABELS[system]}  [{SETTINGS.resolved_model}]\n{'=' * 72}")
        for i, item in enumerate(gold, 1):
            q = item["question"]
            if q in cache[system]:
                print(f"   {i:>2}/{len(gold)}  (cache)  {q[:60]}")
                continue
            t0 = time.perf_counter()
            try:
                res = RUNNERS[system](q)
            except Exception as e:                   # on n'arrête pas toute l'éval
                print(f"   {i:>2}/{len(gold)}  ERREUR {e}")
                continue
            res["latency_s"] = round(time.perf_counter() - t0, 1)
            cache[system][q] = res
            _save(path, cache)
            print(f"   {i:>2}/{len(gold)}  ({res['latency_s']:>5.1f}s, "
                  f"{len(res['retrieved_contexts'])} ctx)  {q[:60]}")
    return cache


# ── juge RAGAS ──────────────────────────────────────────────────────
def build_metrics():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=ARGS.timeout, max_retries=3, max_wait=30)
    judge = build_judge_llm(SETTINGS, ARGS.judge_model)
    judge_name = getattr(judge, "model", "?")
    llm = LangchainLLMWrapper(judge, run_config=run_config)
    emb = LangchainEmbeddingsWrapper(build_embedding_model(SETTINGS.embedding_model),
                                     run_config=run_config)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }
    return metrics, judge_name


def _score(metric, sample) -> float | None:
    try:
        v = asyncio.run(metric.single_turn_ascore(sample, timeout=ARGS.timeout))
        v = float(v)
        return None if v != v else v                 # NaN → None
    except TypeError:                                # signature sans timeout
        try:
            v = float(asyncio.run(metric.single_turn_ascore(sample)))
            return None if v != v else v
        except Exception as e:
            print(f"      [skip] {metric.__class__.__name__}: {e}")
            return None
    except Exception as e:
        print(f"      [skip] {metric.__class__.__name__}: {str(e)[:200]}")
        return None


def evaluate(gold: list[dict], gens: dict, stem: str, metrics: dict, judge_name: str) -> dict:
    from ragas import SingleTurnSample

    path = RESULTS / f"scores_{stem}.json"
    cache = _load(path)
    if cache.get("_judge") and cache["_judge"] != judge_name:
        cache = {}
    if cache.get("_llm") and cache["_llm"] != gens.get("_llm"):
        cache = {}
    cache["_judge"], cache["_llm"] = judge_name, gens.get("_llm")
    for system in SYSTEMS:
        cache.setdefault(system, {})
        print(f"\n{'=' * 72}\n  Notation RAGAS — {SYSTEM_LABELS[system]}  [juge : {judge_name}]\n{'=' * 72}")
        for i, item in enumerate(gold, 1):
            q = item["question"]
            g = gens.get(system, {}).get(q)
            if g is None:
                continue
            row = cache[system].setdefault(q, {})
            print(f"  Q{i:>2}/{len(gold)} : {q[:62]}")
            sample = SingleTurnSample(
                user_input=q, response=g["response"],
                retrieved_contexts=g["retrieved_contexts"] or [""],
                reference=item["reference"],
            )
            for name in METRICS:
                if row.get(name) is not None:
                    print(f"    {name:<40} {row[name]:.3f} (cache)")
                    continue
                row[name] = _score(metrics[name], sample)
                _save(path, cache)
                print(f"    {name:<40} " + (f"{row[name]:.3f}" if row[name] is not None else "  NaN"))
    return cache


# ── rapport ─────────────────────────────────────────────────────────
def report(gold: list[dict], gens: dict, scores: dict, stem: str, judge_name: str) -> dict:
    import pandas as pd

    means: dict[str, dict[str, float | None]] = {}
    for system in SYSTEMS:
        rows = []
        for item in gold:
            q = item["question"]
            g = gens.get(system, {}).get(q)
            if not g:
                continue
            s = scores.get(system, {}).get(q, {})
            rows.append({
                "user_input": q, "response": g["response"],
                "retrieved_contexts": g["retrieved_contexts"], "reference": item["reference"],
                "latency_s": g.get("latency_s"), "trace": " | ".join(g.get("trace", [])),
                **{m: s.get(m) for m in METRICS},
            })
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"resultats_{stem}_{system}.csv", index=False, encoding="utf-8-sig")
        means[system] = {m: (float(df[m].dropna().mean()) if len(df) and df[m].notna().any() else None)
                         for m in METRICS}
        means[system]["latency_s"] = float(df["latency_s"].mean()) if len(df) else None
        means[system]["n"] = len(df)

    def avg4(s):
        vals = [means[s][m] for m in METRICS if means[s][m] is not None]
        return sum(vals) / len(vals) if vals else None

    lines = [
        f"# Comparaison RAGAS — `{stem}.json`",
        "",
        f"- LLM applicatif (identique pour les deux systèmes) : **{gens.get('_llm')}** (Ollama, local)",
        f"- Juge RAGAS (identique) : **{judge_name}** (Ollama, local, température 0)",
        f"- Questions : {len(gold)}",
        "",
        "| Métrique | " + " | ".join(SYSTEM_LABELS[s] for s in SYSTEMS)
        + (" | Δ (agentique − baseline) |" if len(SYSTEMS) == 2 else " |"),
        "|---|" + "---|" * len(SYSTEMS) + ("---|" if len(SYSTEMS) == 2 else ""),
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    for m in METRICS + ["_avg"]:
        label = "**Moyenne des 4 métriques**" if m == "_avg" else METRIC_LABELS[m]
        vals = [avg4(s) if m == "_avg" else means[s][m] for s in SYSTEMS]
        line = f"| {label} | " + " | ".join(fmt(v) for v in vals)
        if len(SYSTEMS) == 2:
            a, b = vals[0], vals[1]
            line += " | " + ("—" if a is None or b is None else f"{b - a:+.3f}")
        lines.append(line + " |")
    lat = [means[s]["latency_s"] for s in SYSTEMS]
    lines.append("| Latence moyenne (s) | " + " | ".join("—" if v is None else f"{v:.1f}" for v in lat)
                 + (" | — |" if len(SYSTEMS) == 2 else " |"))
    md = "\n".join(lines) + "\n"
    (RESULTS / f"comparaison_{stem}.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    try:                                            # graphique
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.6))
        labels = [METRIC_LABELS[m] for m in METRICS] + ["Moyenne"]
        width = 0.38
        colors = {"baseline": "#8a94a6", "agentic": "#2f6fdf"}
        for k, s in enumerate(SYSTEMS):
            vals = [means[s][m] or 0 for m in METRICS] + [avg4(s) or 0]
            xs = [i + (k - (len(SYSTEMS) - 1) / 2) * width for i in range(len(labels))]
            bars = ax.bar(xs, vals, width, label=SYSTEM_LABELS[s], color=colors.get(s))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score RAGAS")
        ax.set_title(f"RAG avancé vs RAG agentique — {stem} ({gens.get('_llm')}, juge {judge_name})",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(RESULTS / f"comparaison_{stem}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"(graphique non généré : {e})")
    return {"means": means, "avg": {s: avg4(s) for s in SYSTEMS}}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"⏳  Initialisation du RAG (LLM : {SETTINGS.resolved_provider} / {SETTINGS.resolved_model})")
    t0 = time.perf_counter()
    init_container(with_agentic="agentic" in SYSTEMS)
    print(f"✅  RAG prêt en {time.perf_counter() - t0:.1f}s")

    golds = {}
    for name in ARGS.gold:
        gold = json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))
        golds[Path(name).stem] = gold[: ARGS.limit] if ARGS.limit else gold

    # 1) toutes les générations d'abord (le LLM applicatif reste chargé en mémoire)
    gens = {stem: generate(gold, stem) for stem, gold in golds.items()}
    # 2) puis toutes les notations (le juge prend le relais)
    metrics, judge_name = build_metrics()
    summary = {}
    for stem, gold in golds.items():
        scores = evaluate(gold, gens[stem], stem, metrics, judge_name)
        summary[stem] = report(gold, gens[stem], scores, stem, judge_name)

    _save(RESULTS / "summary.json", {"llm": SETTINGS.resolved_model, "judge": judge_name,
                                     "results": summary})
    print(f"📄  Résultats détaillés dans {RESULTS}")


if __name__ == "__main__":
    main()
