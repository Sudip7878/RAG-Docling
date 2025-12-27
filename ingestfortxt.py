from pathlib import Path

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


############################################################
# 1. Paths
############################################################
TXT_DIR = "Text"
CHROMA_DIR = "chroma_db"

############################################################
# 2. Load TXT files (ROBUST)
############################################################
documents = []

txt_path = Path(TXT_DIR)

if not txt_path.exists():
    raise FileNotFoundError(f"Folder '{TXT_DIR}' does not exist")

for txt_file in txt_path.rglob("*.txt"):  # 👈 recursive
    print(f"Processing: {txt_file}")

    try:
        text = txt_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = txt_file.read_text(encoding="latin-1")

    if not text.strip():
        print(f"⚠️ Skipping empty file: {txt_file.name}")
        continue

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": txt_file.name,
                "type": "txt"
            }
        )
    )

print(f"Total TXT documents loaded: {len(documents)}")

############################################################
# 3. STOP if nothing loaded
############################################################
if not documents:
    raise ValueError("❌ No TXT documents found. Check Text folder.")

############################################################
# 4. Chunking
############################################################
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

############################################################
# 5. Embeddings
############################################################
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############################################################
# 6. Store in ChromaDB
############################################################
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

vectorstore.add_documents(chunks)

print("✅ TXT files successfully stored in ChromaDB")
