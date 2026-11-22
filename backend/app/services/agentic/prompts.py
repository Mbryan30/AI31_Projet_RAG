"""
Prompts et schémas de sortie structurée des agents.

Chaque agent a un rôle unique (principe des systèmes multi-agents) :
  Planner  → comprend, route et décompose la question
  Grader   → juge la pertinence des articles récupérés (CRAG)
  Rewriter → reformule les requêtes si le contexte est insuffisant
  Writer   → rédige la réponse à partir du contexte validé
  Critic   → auto-critique la réponse (Reflection / Reflexion)
  Reviser  → corrige la réponse à partir de la critique
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ── 1. Planner / Router ─────────────────────────────────────────────

class Plan(BaseModel):
    route: Literal["rgpd", "hors_sujet"] = Field(
        description="'rgpd' si la question porte sur la protection des données / le RGPD, "
                    "sinon 'hors_sujet'."
    )
    sous_questions: list[str] = Field(
        description="1 à 3 requêtes de recherche courtes, en français juridique, qui couvrent "
                    "chacune un aspect distinct de la question."
    )
    articles_candidats: list[int] = Field(
        description="Numéros (1 à 99) des articles du RGPD les plus susceptibles de contenir "
                    "la réponse (5 maximum). Ce sont des hypothèses : elles seront vérifiées."
    )


PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es l'agent PLANIFICATEUR d'un système de question-réponse sur le RGPD
(Règlement (UE) 2016/679, 99 articles).

Ta mission :
1. Décider si la question relève du RGPD / de la protection des données.
2. La décomposer en 1 à 3 requêtes de recherche précises (vocabulaire du règlement :
   « responsable du traitement », « personne concernée », « autorité de contrôle »…).
   Une question simple n'a besoin que d'une seule requête.
3. Proposer les numéros d'articles du RGPD qui contiennent probablement la réponse
   (définitions → article 4 ; principes → 5 ; licéité → 6 ; etc.).

Question : {question}"""
)


# ── 2. Grader (CRAG) ────────────────────────────────────────────────

class Grade(BaseModel):
    pertinents: list[int] = Field(
        description="Numéros [i] des extraits qui contiennent des informations utiles pour "
                    "répondre à la question, du plus utile au moins utile."
    )
    suffisant: bool = Field(
        description="True si les extraits pertinents suffisent à répondre complètement."
    )
    manque: str = Field(
        default="",
        description="Si insuffisant : ce qui manque pour répondre (une phrase).",
    )


GRADER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es l'agent ÉVALUATEUR DE DOCUMENTS (Corrective RAG).
Pour chaque extrait du RGPD ci-dessous, décide s'il est utile pour répondre à la question.
Sois exigeant : un article seulement « voisin » du sujet n'est PAS pertinent.

Question : {question}

Extraits :
{documents}"""
)


# ── 3. Rewriter ─────────────────────────────────────────────────────

class Rewrite(BaseModel):
    requetes: list[str] = Field(description="1 à 3 nouvelles requêtes de recherche.")
    articles_candidats: list[int] = Field(
        default_factory=list,
        description="Numéros d'articles supplémentaires à consulter (optionnel).",
    )


REWRITER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es l'agent REFORMULATEUR. La recherche précédente n'a pas fourni assez
d'informations pour répondre.

Question : {question}
Requêtes déjà essayées : {tried}
Ce qui manque : {missing}

Propose de nouvelles requêtes de recherche, différentes des précédentes, avec le
vocabulaire exact du RGPD, et éventuellement d'autres numéros d'articles à consulter."""
)


# ── 4. Writer ───────────────────────────────────────────────────────
# Le premier jet utilise volontairement le MÊME prompt que le pipeline de base
# (app.services.pipeline.RGPD_PROMPT) : le gain mesuré vient donc de l'agentique
# (meilleur contexte + réflexion), pas d'un prompt différent.


# ── 5. Critic (Reflection / Reflexion) ──────────────────────────────

class Critique(BaseModel):
    fidele: bool = Field(
        description="True si CHAQUE affirmation de la réponse est appuyée par le contexte."
    )
    repond_a_la_question: bool = Field(
        description="True si la réponse traite directement et complètement la question posée."
    )
    note: int = Field(description="Note globale de 1 à 10.")
    affirmations_non_fondees: list[str] = Field(
        default_factory=list,
        description="Affirmations absentes du contexte ou contredites par lui.",
    )
    elements_manquants: list[str] = Field(
        default_factory=list,
        description="Éléments de réponse attendus par la question mais absents de la réponse.",
    )
    elements_superflus: list[str] = Field(
        default_factory=list,
        description="Passages hors sujet ou trop généraux qui n'aident pas à répondre.",
    )
    requetes_recherche: list[str] = Field(
        default_factory=list,
        description="Requêtes pour aller chercher dans le RGPD les éléments manquants "
                    "(vide si le contexte suffit).",
    )


CRITIC_PROMPT = ChatPromptTemplate.from_template(
    """Tu es l'agent CRITIQUE, relecteur juridique exigeant.
Évalue la réponse ci-dessous en la confrontant STRICTEMENT au contexte RGPD fourni.

Vérifie :
- fidélité : chaque affirmation (y compris dans « Points de vigilance » et
  « Recommandation pratique ») doit être appuyée par le contexte ;
- pertinence : la réponse répond-elle directement à LA question posée, sans digresser ?
- complétude : manque-t-il un élément que la question demande explicitement ?

Contexte RGPD :
{context}

Question : {question}

Réponse à évaluer :
{answer}"""
)


# ── 6. Reviser ──────────────────────────────────────────────────────

REVISER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es l'agent RÉVISEUR. Réécris la réponse en tenant compte de la critique.

Règles :
- Appuie-toi UNIQUEMENT sur le contexte RGPD ci-dessous ; supprime toute affirmation
  qui n'y figure pas (même si elle te paraît vraie).
- Commence par répondre directement à la question ; supprime les passages superflus.
- Ajoute les éléments manquants s'ils figurent dans le contexte, en citant l'article.
- Conserve la structure en 5 sections (## 1. Réponse directe … ## 5. Recommandation
  pratique). Les sections 4 et 5 restent courtes et strictement liées à la question.
- Termine par : ⚠️ Cette réponse est informative et ne constitue pas un avis juridique.

Contexte RGPD :
{context}

Question : {question}

Réponse précédente :
{answer}

Critique :
{critique}

Réponse révisée :"""
)


OFF_TOPIC_ANSWER = (
    "## 1. Réponse directe\n"
    "Cette question ne semble pas porter sur le RGPD ou la protection des données "
    "personnelles : je ne peux répondre qu'à partir du texte du règlement.\n\n"
    "⚠️ Cette réponse est informative et ne constitue pas un avis juridique."
)
