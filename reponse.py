import re, json, uuid, warnings
from langchain_core.documents import Document
import json

with open("parent_documents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# On reconstruit les Document à partir des dicos
parent_documents = [
    Document(page_content=d["page_content"], metadata=d["metadata"])
    for d in data
]

# Et on reconstruit l'index parent_id -> Document
parent_index = {doc.metadata["parent_id"]: doc for doc in parent_documents}

# ## 5. LLM – Qwen2.5-0.5B-Instruct
# 
# Modèle gratuit, léger (0.5B), instruct-tuned, multilingue. `temperature=0.3` pour des réponses factuelles stables.

# In[12]:


from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, List, Any

model_name = "Qwen/Qwen2.5-3B-Instruct"   # ← ou 1.5B / 3B selon ton choix
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model_hf   = AutoModelForCausalLM.from_pretrained(model_name)

# ID du token de fin de tour Qwen — c'est LUI qui arrête la génération proprement
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

pipe = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer,
    max_new_tokens=800,
    temperature=0.3,
    do_sample=True,
    repetition_penalty=1.15,
    eos_token_id=[tokenizer.eos_token_id, im_end_id],
    pad_token_id=tokenizer.eos_token_id,
)


class LocalLLMWrapper(LLM):
    hf_pipeline: Any
    hf_tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "local_hf"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # Formate selon le chat template natif de Qwen 2.5-Instruct
        messages = [{"role": "user", "content": prompt}]
        formatted = self.hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        output = self.hf_pipeline(formatted)[0]["generated_text"]

        # Retire le prompt formaté du début de la sortie
        if output.startswith(formatted):
            output = output[len(formatted):].strip()

        # Filet de sécurité : nettoie un éventuel <|im_end|> restant
        output = output.replace("<|im_end|>", "").strip()
        return output


llm = LocalLLMWrapper(hf_pipeline=pipe, hf_tokenizer=tokenizer)
print("✅  LLM prêt")

# ## 6. Multi-Query Retriever
# 
# Le LLM reformule la question en 4 variantes pour élargir la couverture sémantique, puis les résultats sont fusionnés et dédupliqués.

# In[13]:


from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Tu es un assistant spécialisé en droit RGPD.
Reformule la question suivante en 4 variantes différentes pour améliorer
la recherche dans une base documentaire juridique.
Génère uniquement les 4 questions, une par ligne, sans numérotation ni tiret.

Question originale : {question}
Variantes :"""
)

vector_store = Chroma(
                    persist_directory = "./chroma_rgpd",
                    embedding_function = embedding_model
)

base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

def generate_queries(question: str) -> list:
    output = llm.invoke(multi_query_prompt.format(question=question))
    return [q.strip() for q in output.split("\n") if q.strip()][:4]

def multi_query_search(question: str) -> list:
    queries  = generate_queries(question)
    all_docs = []
    for q in queries:
        all_docs.extend(base_retriever.invoke(q))
    # Déduplication par contenu
    return list({doc.page_content: doc for doc in all_docs}.values())

print("✅  Multi-Query prêt")


# ## 7. Parent Document Retriever
# 
# Les child chunks trouvés servent d'index → on remonte l'article complet via `parent_id`. Chaque article n'est retourné qu'une seule fois (déduplication par `set`).

# In[14]:


def retrieve_parent_documents(question: str, top_k: int = 8) -> list:
    child_hits      = multi_query_search(question)
    seen_parent_ids = set()
    parent_docs     = []
    best_child      = {}  

    """ - pid renvoie l'indentifiant du document parent associé à un chunk enfant, permettant de regrouper les chunks par article complet. 
        - Doc c'est le document complet de l'article, avec son contenu et ses métadonnées, récupéré à partir de l'index parent_index en utilisant le parent_id.
        - parent_docs est la liste finale des documents parents (articles complets) qui seront retournés par la fonction, après avoir vérifié que leur parent_id n'a pas déjà été vu et qu'ils existent dans l'index.

    """

    for child in child_hits:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            doc = parent_index.get(pid)
            if doc:
                parent_docs.append(doc)
                best_child[pid] = child.page_content  
        if len(parent_docs) >= top_k:
            break

    return parent_docs, best_child  

print("✅  Parent Document Retriever prêt")


# ## 8. Re-ranker – Cross-Encoder
# 
# `ms-marco-MiniLM-L-6-v2` score chaque paire (question, article) pour classer les 3 articles les plus pertinents. Plus précis que la similarité cosine seule.

# In[15]:


from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

def rerank_documents(question: str, docs: list, best_child: dict, top_n: int = 3) -> list:  # ← NOUVEAU : best_child
    if not docs:
        return []

    pairs = []
    for doc in docs:
        pid  = doc.metadata.get("parent_id")
        text = best_child.get(pid, doc.page_content[:512])
        pairs.append((question, text))

    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

print("✅  Re-ranker prêt")


# ## 9. Prompt RGPD & Pipelines
# 
# Deux points d'entrée :
# - `smart_rag_pipeline()` — **à utiliser toujours** : détecte les questions sur un article précis et bypasse le LLM
# - `full_rag_pipeline()` — pipeline sémantique complet (appelé automatiquement par `smart_rag_pipeline`)
# 

# In[16]:


from langchain_core.prompts import ChatPromptTemplate

RGPD_PROMPT = """Tu es un assistant expert en conformité RGPD.
Tu réponds UNIQUEMENT à partir des articles RGPD fournis dans le contexte.
Si l'information n'est pas présente, dis : "Je ne peux pas répondre avec certitude à partir des extraits RGPD fournis."
RÈGLE ABSOLUE : Ne génère jamais un article absent du contexte.

Contexte RGPD (articles complets) :
{context}

Question : {question}

Réponds en suivant EXACTEMENT cette structure :

## 1. Réponse directe
[Réponse claire et concise]

## 2. Articles RGPD concernés
[Liste des articles et chapitres du contexte]

## 3. Explication détaillée
[Ce que disent les articles pertinents]

## 4. Points de vigilance
[Base légale, consentement, minimisation, données sensibles]

## 5. Recommandation pratique
[Conseil concret et actionnable]

⚠️ Cette réponse est informative et ne constitue pas un avis juridique.

Réponse :
"""

prompt_template = ChatPromptTemplate.from_template(RGPD_PROMPT)

def format_parent_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(
            f"{'='*60}\n{m['chapitre']} – {m['titre_chapitre']}\n"
            f"{m['article']} – {m['titre_article']}\n{'='*60}\n{doc.page_content}"
        )
    return "\n\n".join(parts)


# ── Pipeline sémantique complet ──────────────────────────────────────
def full_rag_pipeline(question: str) -> dict:
    parent_docs, best_child = retrieve_parent_documents(question, top_k=8)  
    reranked_docs = rerank_documents(question, parent_docs, best_child, top_n=3) 

    articles_cites = [
        f"{d.metadata['article']} ({d.metadata['chapitre']} – {d.metadata['titre_chapitre']})"
        for d in reranked_docs
    ]

    final_prompt = prompt_template.format(
        context=format_parent_docs(reranked_docs),
        question=question
    )
    raw = llm.invoke(final_prompt)

    # Supprime le prompt répété si présent
    marker = "Réponse :"
    if marker in raw:
        raw = raw[raw.rfind(marker) + len(marker):].strip()

    return {"question": question, "reponse": raw, "articles_cites": articles_cites}


# ── Routeur principal ─────────────────────────────────────────────────
def smart_rag_pipeline(question: str) -> dict:
# Ordre des traitements :
# 1. Recherche explicite d'un ou plusieurs articles.
#    Exemples :
#    - "Que dit l'article 5 ?"
#    - "Compare l'article 5 et l'article 6"
#    - "Montre-moi les articles 5, 6 et 7"
# 2. Recherche explicite d'un chapitre à partir de son titre.
#    Exemple :
#    - "Quels articles concernent les droits de la personne concernée ?"
# 3. Si aucun article ou chapitre n'est détecté,
#    on utilise le pipeline RAG classique.
# ==========================================================
# ROUTEUR 1 : RECHERCHE D'UN OU PLUSIEURS ARTICLES
# ==========================================================
# On cherche d'abord les formulations du type :
# "article 5", "article 20", "article premier", etc.
    articles_detectes = re.findall(
        r"article\s+(\d+|premier)",
        question,
        re.IGNORECASE
    )
    # Liste finale des numéros d'articles détectés
    numeros = []
    if articles_detectes:
        # Exemple :
        # "article 5 et article 6"
        # -> ['5', '6']
        numeros = [a.capitalize() for a in articles_detectes]
    else:
        # Gestion des formulations du type :
        # "articles 5, 6 et 7"
        # "articles 12, 13, 14"
        match_plural = re.search(
            r"articles?\s+([\d,\set]+)",
            question,
            re.IGNORECASE
        )
        if match_plural:
            numeros = re.findall(r"\d+", match_plural.group(1))
    # Si au moins un numéro d'article a été trouvé
    if numeros:
        articles_trouves = []
        # Recherche des articles correspondants dans le corpus
        for numero in numeros:
            article_key = f"Article {numero}"
            found = [
                d for d in parent_documents
                if d.metadata["article"].lower()
                == article_key.lower()
            ]
            if found:
                articles_trouves.append(found[0])
        # ======================================================
        # CAS 1 : UN SEUL ARTICLE TROUVÉ
        # ======================================================
        if len(articles_trouves) == 1:
            d = articles_trouves[0]
            m = d.metadata
            reponse = (
                f"## 1. Réponse directe\n"
                f"L'{m['article']} se trouve dans le "
                f"{m['chapitre']} – {m['titre_chapitre']}.\n\n"
                f"## 2. Article concerné\n"
                f"{m['article']} – {m['titre_article']}\n\n"
                f"## 3. Contenu complet\n"
                f"{d.page_content}"
            )
            return {
                "question": question,
                "reponse": reponse,
                "articles_cites": [
                    f"{m['article']} ({m['chapitre']} – {m['titre_chapitre']})"
                ]
            }
        # ======================================================
        # CAS 2 : PLUSIEURS ARTICLES TROUVÉS
        # ======================================================
        if len(articles_trouves) > 1:
            reponse = (
                "## 1. Réponse directe\n"
                f"{len(articles_trouves)} articles ont été trouvés.\n\n"
            )
            reponse += "## 2. Articles concernés\n\n"
            for d in articles_trouves:
                m = d.metadata
                reponse += (
                    f"### {m['article']} – {m['titre_article']}\n"
                    f"Chapitre : {m['chapitre']} – "
                    f"{m['titre_chapitre']}\n\n"
                    f"{d.page_content}\n\n"
                    f"{'-' * 60}\n\n"
                )
            return {
                "question": question,
                "reponse": reponse,
                "articles_cites": [
                    f"{d.metadata['article']} "
                    f"({d.metadata['chapitre']} – "
                    f"{d.metadata['titre_chapitre']})"
                    for d in articles_trouves
                ]
            }
        # Aucun article du corpus ne correspond
        return {
            "question": question,
            "reponse": (
                "Aucun des articles demandés n'existe "
                "dans le RGPD."
            ),
            "articles_cites": []
        }
    # ==========================================================
    # ROUTEUR 2 : RECHERCHE D'UN CHAPITRE PAR SON TITRE
    # ==========================================================
    chapitre_match = next(
        (
            c for c in parent_documents
            if c.metadata["titre_chapitre"].lower()
            in question.lower()
        ),
        None
    )
    if chapitre_match:
        m = chapitre_match.metadata
        # Récupération de tous les articles du chapitre
        articles_du_chapitre = [
            d for d in parent_documents
            if d.metadata["chapitre"] == m["chapitre"]
        ]
        reponse = (
            f"## 1. Réponse directe\n"
            f"Le chapitre qui traite de "
            f"« {m['titre_chapitre']} » "
            f"est le **{m['chapitre']}**.\n\n"
            f"## 2. Articles concernés\n"
            + "\n".join(
                f"- {a.metadata['article']} – "
                f"{a.metadata['titre_article']}"
                for a in articles_du_chapitre
            )
        )
        return {
            "question": question,
            "reponse": reponse,
            "articles_cites": [
                f"{a.metadata['article']} ({m['chapitre']})"
                for a in articles_du_chapitre
            ]
        }
    # ==========================================================
    # ROUTEUR 3 : RECHERCHE SÉMANTIQUE (RAG COMPLET)
    # ==========================================================
    # Si aucun article ou chapitre n'est identifié,
    # on utilise le pipeline RAG classique :
    # recherche vectorielle + reranking + LLM.
    return full_rag_pipeline(question)



print("✅  Pipelines prêts  →  utilise smart_rag_pipeline() pour toutes tes questions")