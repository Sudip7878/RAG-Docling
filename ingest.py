import os
from pathlib import Path

# Docling
from docling.document_converter import DocumentConverter

# LangChain (NEW PATHS)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


############################################################
# 1. Paths
############################################################
PDF_DIR = "Book"
CHROMA_DIR = "chroma_db"

############################################################
# 2. Load & convert PDFs using Docling
############################################################
converter = DocumentConverter()

documents = []

for pdf_file in Path(PDF_DIR).glob("*.pdf"):
    print(f"Processing: {pdf_file.name}")

    result = converter.convert(pdf_file)
    text = result.document.export_to_markdown()

    documents.append(
        Document(
            page_content=text,
            metadata={"source": pdf_file.name}
        )
    )

print(f"Total documents loaded: {len(documents)}")

############################################################
# 3. Chunking
############################################################
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

############################################################
# 4. Embeddings
############################################################
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############################################################
# 5. Store in ChromaDB (AUTO-PERSIST)
############################################################
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

vectorstore.add_documents(chunks)

print("✅ PDFs successfully stored in ChromaDB")
