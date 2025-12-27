import os
from dotenv import load_dotenv
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------
# Load environment variables
# -----------------------------------
load_dotenv()

CHROMA_DIR = "chroma_db"

# -----------------------------------
# Load Embedding Model
# -----------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# Load Vector Store
# -----------------------------------
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# -----------------------------------
# Retrieve + Rank Documents
# -----------------------------------
def retrieve_and_rank(query: str, k: int = 5) -> List[Document]:
    """
    Performs similarity search and ranks documents
    based on relevance score.
    """
    results = vectordb.similarity_search_with_score(query, k=k)

    # Sort by similarity score (lower = better)
    ranked_results = sorted(results, key=lambda x: x[1])

    return [doc for doc, score in ranked_results]

# -----------------------------------
# Merge Retrieved Documents
# -----------------------------------
def merge_documents(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------------
# Load ChatGroq LLM
# -----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# -----------------------------------
# Prompt Template (Context Engineering)
# -----------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant.
Answer the question strictly using the context below.
If the answer is not found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------------
# RAG Pipeline
# -----------------------------------
def rag_query(user_query: str) -> str:
    docs = retrieve_and_rank(user_query)
    context = merge_documents(docs)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": user_query
    })

    return response.content

# -----------------------------------
# CLI Interface
# -----------------------------------
if __name__ == "__main__":
    print("✅ RAG system ready. Type 'exit' to quit.")

    while True:
        user_input = input("\n🧠 Ask your question: ")

        if user_input.lower() == "exit":
            break

        answer = rag_query(user_input)
        print("\n📌 Answer:\n", answer)
