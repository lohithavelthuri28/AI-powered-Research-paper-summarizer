# backend.py
# ---------------------------
# RAG demo using:
# - LangChain 0.2.x
# - Chroma 0.4.x (auto-persist)
# - Embeddings: langchain-huggingface (all-MiniLM-L6-v2)
# - LLM: Ollama (llama3.2:1b)
#
# Quick tips:
#   pip install:
#     langchain==0.2.14 langchain-community==0.2.5 langchain-text-splitters==0.2.4 \
#     langchain-huggingface==0.0.3 chromadb pypdf python-dotenv langchain-ollama
#
#   Make sure Ollama is running and model is available:
#     1) Start server:   ollama serve
#     2) Pull model:     ollama pull llama3.2:1b
#
# Run:
#   python backend.py
#   # or pass a query:
#   python backend.py "Your question here"
# ---------------------------

import os
import sys
import logging
from typing import List

from dotenv import load_dotenv

# LangChain + ecosystem
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_FILE = os.environ.get("PDF_PATH", "data/Ndalgo.pdf")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "my_chroma_db")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
K = int(os.environ.get("TOP_K", "4"))

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_and_split_pdf(path: str) -> List:
    """Load a PDF and split into chunks."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found at '{path}'. Put your file there or set PDF_PATH env var."
        )
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def get_vectorstore(docs: List, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Create or load a Chroma vector store.
    Chroma 0.4+ auto-persists; no manual db.persist() needed.
    """
    # If a persistent dir exists, load it; else create from documents
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        logger.info("Loading existing Chroma DB from '%s' ...", CHROMA_DIR)
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        logger.info("Creating new Chroma DB at '%s' ...", CHROMA_DIR)
        return Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

def build_prompts():
    system_template = """You are a teaching chatbot. Use only the source data provided to answer.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

{context}
"""
    question_prompt = PromptTemplate(
        template=system_template,
        input_variables=["context"]
    )

    refine_template = """You are a teaching chatbot. We have an existing answer: 
{existing_answer}

We have the following new context to consider:
{context}

Please refine the original answer if there's new or better information. 
If the new context does not change or add anything to the original answer, keep it the same.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

Question: {question}

Refined Answer:
"""
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=["existing_answer", "context", "question"]
    )

    return question_prompt, refine_prompt

def build_chain(db: Chroma, llm: Ollama) -> RetrievalQA:
    """Build a RetrievalQA chain (refine type) on LangChain 0.2.x."""
    question_prompt, refine_prompt = build_prompts()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=db.as_retriever(search_kwargs={"k": K}),
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "refine_prompt": refine_prompt,
            "document_variable_name": "context",
        },
    )
    return chain

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # 1) LLM via Ollama (make sure `ollama serve` is running and model is pulled)
    llm = Ollama(model="tinyllama",
    temperature=0.2,
    num_gpu=0,          # <-- force CPU
    num_ctx=1024 )

    # 2) Load & embed docs
    logger.info("Loading and chunking PDF: %s", DATA_FILE)
    docs = load_and_split_pdf(DATA_FILE)

    logger.info("Preparing embeddings (all-MiniLM-L6-v2) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info("Setting up Chroma vector store ...")
    db = get_vectorstore(docs, embeddings)

    # 3) Build chain
    logger.info("Building RetrievalQA chain (refine) ...")
    chain = build_chain(db, llm)

    # 4) Query
    query = " ".join(sys.argv[1:]).strip() or "Cultivate the here-and-now neurotransmitters"
    logger.info("Query: %s", query)

    try:
        result = chain.invoke({"query": query})
        # RetrievalQA returns a dict with "result" (answer text)
        print("\n--- Answer ---\n")
        print(result.get("result", "").strip())
        print("\n--------------\n")
    except Exception as e:
        logger.exception("Failed to run chain: %s", e)
        print(
            "\n[ERROR] Failed to generate an answer.\n"
            "• Ensure Ollama is running:   ollama serve\n"
            f"• Ensure model is available:  ollama pull {MODEL_NAME}\n"
            "• Check that your dependencies match the versions in the header comment.\n"
        )

if __name__ == "__main__":
    main()
