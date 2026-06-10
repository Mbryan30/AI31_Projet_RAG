#!/usr/bin/env python
# coding: utf-8

# # RAG Avancé – RGPD
# ## Architecture complète
# 
# ```
# Question utilisateur
#         │
#         ├─ article précis ?  →  Métadonnées (fiable à 100%)
#         │
#         ├─ Chapitre précis ?  →  Métadonnées (fiable à 100%)
#         │
#         └─ question sémantique ?
#                 │
#         Multi-Query Retriever  (4 variantes)
#                 │
#         Vector Store ChromaDB  (child chunks + métadonnées)
#                 │
#         Parent Document Retriever  (article complet)
#                 │
#         Re-ranker Cross-Encoder  (top 3)
#                 │
#         LLM Qwen2.5-0.5B-Instruct
#                 │
#         Réponse structurée + articles cités
# ```
# 

# ## 1. Imports
# 
# ```bash
# pip install langchain langchain-community langchain-chroma chromadb sentence-transformers transformers accelerate beautifulsoup4 requests
# ```

# In[7]:


import re, json, uuid, warnings
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ## 2. Parsing du RGPD
# 
# Extraction de chaque article avec ses métadonnées : chapitre, titre, numéro, contenu.

# In[8]:


with open("L_2016119FR.01000101.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

def clean_text(element):
    return element.get_text(" ", strip=True) if element else ""

resultat = []
chapitres = soup.find_all("div", id=re.compile(r"^cpt_[IVXLCDM]+$"))

for chapitre in chapitres:
    chapitre_nom   = clean_text(chapitre.find("p", class_="oj-ti-section-1"))
    titre_chapitre = clean_text(chapitre.find("p", class_="oj-ti-section-2"))
    if not chapitre_nom.startswith("CHAPITRE"):
        continue

    chapitre_data = {"chapitre": chapitre_nom, "titre_chapitre": titre_chapitre, "contenu": []}
    articles_deja_vus = set()

    for article in chapitre.find_all("div", id=re.compile(r"^art_\d+$")):
        article_id = article.get("id")
        if article_id in articles_deja_vus:
            continue
        articles_deja_vus.add(article_id)

        numero_article = clean_text(article.find("p", class_="oj-ti-art"))
        titre_article  = clean_text(article.find("p", class_="oj-sti-art"))
        if not numero_article.startswith("Article"):
            continue

        contenu_article = "\n".join(
            clean_text(p) for p in article.find_all("p", class_="oj-normal") if clean_text(p)
        )
        chapitre_data["contenu"].append({
            "article": numero_article, "titre_article": titre_article, "contenu_article": contenu_article
        })

    resultat.append(chapitre_data)

print(f"✅  {len(resultat)} chapitres extraits")
print(f"✅  {sum(len(c['contenu']) for c in resultat)} articles extraits")

with open("rgpd_structure.json", "w", encoding="utf-8") as f:
    json.dump(resultat, f, ensure_ascii=False, indent=4)


# ## 3. Documents LangChain – stratégie Parent / Child
# 
# | | Child chunk | Parent document |
# |---|---|---|
# | Rôle | Recherche sémantique | Génération LLM |
# | Taille | ~300 tokens | Article entier |
# | Stocké dans | ChromaDB | `parent_index` (dict) |
# 
# Chaque child chunk hérite des métadonnées de son article parent et d'un `parent_id` pour remonter au texte complet.
# 

# In[9]:


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("rgpd_structure.json", "r", encoding="utf-8") as f:
    data = json.load(f)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

parent_documents = []
child_documents  = []

for chapitre in data:
    chapitre_nom   = chapitre.get("chapitre", "")
    titre_chapitre = chapitre.get("titre_chapitre", "")

    for article in chapitre.get("contenu", []):
        numero_article  = article.get("article", "")
        titre_article   = article.get("titre_article", "")
        contenu_article = article.get("contenu_article", "")
        if not contenu_article.strip():
            continue

        parent_id   = str(uuid.uuid4())
        parent_text = (
            f"{chapitre_nom} – {titre_chapitre}\n"
            f"{numero_article} – {titre_article}\n\n"
            f"{contenu_article}"
        )
        metadata_base = {
            "chapitre": chapitre_nom, "titre_chapitre": titre_chapitre,
            "article":  numero_article, "titre_article": titre_article,
            "source":   "RGPD", "parent_id": parent_id,
        }

        parent_documents.append(Document(
            page_content=parent_text,
            metadata={**metadata_base, "type": "parent"}
        ))
        for i, chunk in enumerate(child_splitter.split_text(contenu_article)):
            child_documents.append(Document(
                page_content=chunk,
                metadata={**metadata_base, "type": "child_chunk", "chunk_index": i}
            ))

# Index de récupération rapide parent_id → document complet
parent_index = {doc.metadata["parent_id"]: doc for doc in parent_documents}

print(f"✅  {len(parent_documents)} articles parents")
print(f"✅  {len(child_documents)} child chunks")


# In[10]:


parent_index


# ## 4. Embeddings & Vector Store
# 
# Modèle `paraphrase-multilingual-MiniLM-L12-v2` — supporte nativement le français juridique.

# In[11]:


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

vector_store = Chroma.from_documents(
    documents=child_documents,
    embedding=embedding_model,
    collection_name="rgpd_chunks"
)

base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

print(f"✅  Vector store : {vector_store._collection.count()} chunks indexés")


# ## 5. LLM – Qwen2.5-0.5B-Instruct
# 
# Modèle gratuit, léger (0.5B), instruct-tuned, multilingue. `temperature=0.3` pour des réponses factuelles stables.

# In[12]:


from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, List, Any

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model_hf   = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer,
    max_new_tokens=356,
    temperature=0.3,
    do_sample=True,
)

# Wrapper LangChain compatible (supprime le prompt répété en sortie)
class LocalLLMWrapper(LLM):
    hf_pipeline: Any

    @property
    def _llm_type(self) -> str:
        return "local_hf"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        output = self.hf_pipeline(prompt)[0]["generated_text"]
        if output.startswith(prompt):
            output = output[len(prompt):].strip()
        return output

llm = LocalLLMWrapper(hf_pipeline=pipe)
print("✅  LLM prêt")


# ## 6. Multi-Query Retriever
# 
# Le LLM reformule la question en 4 variantes pour élargir la couverture sémantique, puis les résultats sont fusionnés et dédupliqués.

# In[13]:


from langchain_core.prompts import PromptTemplate

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Tu es un assistant spécialisé en droit RGPD.
Reformule la question suivante en 4 variantes différentes pour améliorer
la recherche dans une base documentaire juridique.
Génère uniquement les 4 questions, une par ligne, sans numérotation ni tiret.

Question originale : {question}
Variantes :"""
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

    # ── Routeur 1 : article précis ──
    match = re.search(r"article\s+(\d+|premier)", question, re.IGNORECASE)
    if match:
        numero      = match.group(1).capitalize()
        article_key = f"Article {numero}"
        found = [d for d in parent_documents
                 if d.metadata["article"].lower() == article_key.lower()]
        if found:
            d, m = found[0], found[0].metadata
            reponse = (
                f"## 1. Réponse directe\n"
                f"L'{article_key} se trouve dans le {m['chapitre']} – {m['titre_chapitre']}.\n\n"
                f"## 2. Article concerné\n{m['article']} – {m['titre_article']}\n\n"
                f"## 3. Contenu complet\n{d.page_content}"
            )
            return {
                "question":       question,
                "reponse":        reponse,
                "articles_cites": [f"{m['article']} ({m['chapitre']} – {m['titre_chapitre']})"]
            }
        return {
            "question":       question,
            "reponse":        f"{article_key} n'existe pas dans le RGPD (99 articles au total).",
            "articles_cites": []
        }

    # ── Routeur 2 : titre de chapitre détecté ──  ← HORS du if match
    chapitre_match = next(
        (c for c in parent_documents
         if c.metadata["titre_chapitre"].lower() in question.lower()),
        None
    )
    if chapitre_match:
        m = chapitre_match.metadata
        articles_du_chapitre = [
            d for d in parent_documents
            if d.metadata["chapitre"] == m["chapitre"]
        ]
        reponse = (
            f"## 1. Réponse directe\n"
            f"Le chapitre qui traite de « {m['titre_chapitre']} » "
            f"est le **{m['chapitre']}**.\n\n"
            f"## 2. Articles concernés\n" +
            "\n".join(
                f"- {a.metadata['article']} – {a.metadata['titre_article']}"
                for a in articles_du_chapitre
            )
        )
        return {
            "question":       question,
            "reponse":        reponse,
            "articles_cites": [
                f"{a.metadata['article']} ({m['chapitre']})"
                for a in articles_du_chapitre
            ]
        }

    # ── Routeur 3 : question sémantique ──
    return full_rag_pipeline(question)


print("✅  Pipelines prêts  →  utilise smart_rag_pipeline() pour toutes tes questions")


# ## 10. Affichage de la réponse
# 
# La fonction `display_response()` produit la sortie structurée visible dans les images de référence.

# In[ ]:


from IPython.display import display, HTML

def display_response(result: dict):
    """Affiche la réponse avec le rendu visuel structuré (cartes + badges articles)."""

    question       = result["question"]
    raw            = result["reponse"]
    articles_cites = result["articles_cites"]

    # Nettoie le prompt répété (sécurité supplémentaire)
    for marker in ["Réponse :", "## 1."]:
        if marker in raw:
            idx = raw.find(marker)
            raw = raw[idx:].strip()
            break


    # Parse les sections ## X. Titre  (NOUVEAU)
    import re as _re
    sections = _re.split(r"(## \d+\..*)", raw)
    parsed   = []
    i = 0
    while i < len(sections):
        s = sections[i].strip()
        if _re.match(r"## \d+\.", s) and i + 1 < len(sections):
            title   = _re.sub(r"## \d+\.\s*", "", s).strip()
            content = sections[i + 1].strip()
            parsed.append((title, content))
            i += 2
        else:
            i += 1

    # ── NOUVEAU : si le LLM n'a pas respecté la structure ──────────────
    if not parsed:
        # Affiche la réponse brute dans une seule carte neutre
        parsed = [("Réponse", raw)]

    # Couleurs par section
    dot_colors = ["#639922", "#185FA5", "#7F77DD", "#BA7517", "#0F6E56"]

    # ── Construction HTML ──
    cards_html = ""

    # Métriques (uniquement pour les réponses RAG sémantique, pas les lookups directs)
    if len(articles_cites) > 0 and "Trouvé directement" not in raw:
        cards_html += f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px">
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Variantes générées</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">4</p>
          </div>
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Articles récupérés</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">8</p>
          </div>
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Après re-ranking</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">{len(articles_cites)}</p>
          </div>
        </div>
        """

    # Sections de réponse
    for idx, (title, content) in enumerate(parsed):
        dot   = dot_colors[idx] if idx < len(dot_colors) else "#888"
        # Mise en forme spéciale pour vigilance (amber) et recommandation (green)
        if idx == 3:  # Points de vigilance
            inner = f'<div style="background:#faeeda;border:0.5px solid #BA7517;border-radius:8px;padding:10px 14px;font-size:14px;color:#633806;line-height:1.65">{content}</div>'
        elif idx == 4:  # Recommandation
            inner = f'<div style="background:#eaf3de;border:0.5px solid #3B6D11;border-radius:8px;padding:10px 14px;font-size:14px;color:#27500A;line-height:1.65">{content}</div>'
        else:
            inner = f'<p style="font-size:14px;color:var(--color-text-primary);line-height:1.65;margin:0">{content}</p>'

        cards_html += f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1rem 1.25rem;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0"></div>
            <span style="font-size:13px;font-weight:500;color:var(--color-text-primary)">{idx+1} — {title}</span>
          </div>
          {inner}
        </div>
        """

    # Badges articles cités
    if articles_cites:
        badges = ""
        for art in articles_cites:
            # Extrait "Article X" et le titre du chapitre
            m = _re.match(r"(Article \S+)\s*\((.+?)\)", art)
            if m:
                num   = m.group(1)
                chap  = m.group(2)
                label = num
            else:
                label = art
                chap  = ""
            badges += f'<span style="display:inline-flex;align-items:center;gap:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:20px;padding:4px 12px;font-size:12px;color:var(--color-text-secondary);margin:3px"><span style="font-weight:500;color:var(--color-text-info)">{label}</span>{chap}</span>'

        cards_html += f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1rem 1.25rem;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:#185FA5;flex-shrink:0"></div>
            <span style="font-size:13px;font-weight:500;color:var(--color-text-primary)">Articles RGPD mobilisés</span>
          </div>
          {badges}
        </div>
        """

    # Avertissement
    cards_html += '<p style="font-size:12px;color:var(--color-text-tertiary);margin:8px 0 0">⚠️ Cette réponse est informative et ne constitue pas un avis juridique.</p>'

    html_out = f"""
    <div style="font-family:sans-serif;padding:4px 0">
      <p style="font-size:13px;color:var(--color-text-secondary);margin:0 0 6px">Question posée</p>
      <div style="display:inline-block;background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:20px;padding:6px 16px;font-size:13px;color:var(--color-text-secondary);margin-bottom:14px">{question}</div>
      {cards_html}
    </div>
    """
    display(HTML(html_out))


print("✅  display_response() prête")


# ## 11. Tests

# In[18]:


# Question sémantique → pipeline RAG complet
result = smart_rag_pipeline(
    "Quelles sont les bases légales pour traiter des données personnelles ?"
)
display_response(result)


# In[19]:


# Question sur un article précis
result = smart_rag_pipeline(
    "Dans quel chapitre se trouve l'article 45 et l'article 56 ?"
)
display_response(result)


# In[20]:


# Question sémantique sur les droits
result = smart_rag_pipeline(
    "Quels sont les droits d'une personne concernée par un traitement de données ?"
)
display_response(result)


# In[21]:


# Question sémantique sur les droits
result = smart_rag_pipeline(
    "c'est quel chapitre qui parle de Dispositions relatives à des situations particulières de traitement ?"
)
display_response(result)




# In[ ]:





# In[ ]:





# ## 12. Metadata Filtering
# 
# Recherche ciblée sur un chapitre ou un article spécifique directement dans ChromaDB.

# In[22]:


def retrieve_with_metadata_filter(question: str, chapitre: str = None, article: str = None, top_k: int = 4):
    where_filter = {}
    if chapitre and article:
        where_filter = {"$and": [{"chapitre": chapitre}, {"article": article}]}
    elif chapitre:
        where_filter = {"chapitre": chapitre}
    elif article:
        where_filter = {"article": article}

    filtered_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, "filter": where_filter} if where_filter else {"k": top_k}
    )
    child_hits = filtered_retriever.invoke(question)

    seen, parents = set(), []
    for child in child_hits:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            p = parent_index.get(pid)
            if p:
                parents.append(p)
    return parents


# Exemple : articles sur les données sensibles dans le Chapitre II
docs = retrieve_with_metadata_filter("données sensibles", chapitre="CHAPITRE II")
print(f"✅  {len(docs)} articles trouvés dans le Chapitre II")
for d in docs:
    print(f"   • {d.metadata['article']} – {d.metadata['titre_article']}")


# In[23]:


# Question sur un article précis
result = smart_rag_pipeline(
    "combien de chapitres contient le RGPD ?"
)
display_response(result)


# In[ ]:




