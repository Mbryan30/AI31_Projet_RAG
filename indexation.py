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

# On transforme chaque Document en dico {page_content, metadata}
parents_a_sauver = [
    {"page_content": doc.page_content, "metadata": doc.metadata}
    for doc in parent_documents
]

with open("parent_documents.json", "w", encoding="utf-8") as f:
    json.dump(parents_a_sauver, f, ensure_ascii=False, indent=4)

print(f"✅  créetion de parent_documents.json")
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
    collection_name="rgpd_chunks",
    persist_directory="./chroma_rgpd"
)

# base_retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 6}
# )

print(f"✅  Vector store : {vector_store._collection.count()} chunks indexés")
