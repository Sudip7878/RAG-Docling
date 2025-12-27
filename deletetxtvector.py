from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

collection = vectordb._collection

print("📦 Total vectors BEFORE:", collection.count())

# 1️⃣ Get ALL documents + metadata
data = collection.get(include=["metadatas"])

ids_to_delete = []

for _id, metadata in zip(data["ids"], data["metadatas"]):
    source = metadata.get("source", "")
    if source.lower().endswith(".txt"):
        ids_to_delete.append(_id)

print(f"🗑️ Found {len(ids_to_delete)} TXT vectors")

# 2️⃣ Delete by IDs
if ids_to_delete:
    collection.delete(ids=ids_to_delete)

print("📦 Total vectors AFTER:", collection.count())
print("✅ TXT vectors deleted successfully")
