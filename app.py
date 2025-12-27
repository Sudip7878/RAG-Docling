import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG QA System", layout="wide")

st.title("📚 RAG Question Answering System")
st.caption("Powered by ChromaDB + HuggingFace + ChatGroq")

# ----------------------------
# Constants
# ----------------------------
CHROMA_DIR = "chroma_db"

# ----------------------------
# Load embeddings & vector DB
# ----------------------------
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectordb

vectordb = load_vectordb()

st.sidebar.success(
    f"📦 Vectors loaded: {vectordb._collection.count()}"
)

# ----------------------------
# Load LLM
# ----------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.2
    )

llm = load_llm()

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

# ----------------------------
# RAG functions
# ----------------------------
def retrieve_and_rank(query, k=5):
    results = vectordb.similarity_search_with_score(query, k=k)
    results = sorted(results, key=lambda x: x[1])
    return [doc for doc, _ in results]

def run_rag(query):
    docs = retrieve_and_rank(query)

    if not docs:
        return "I don't know"

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content

# ----------------------------
# User Input
# ----------------------------
query = st.text_input("🧠 Ask your question")

if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = run_rag(query)

        st.subheader("📌 Answer")
        st.write(answer)
