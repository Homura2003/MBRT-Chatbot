import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

VECTORSTORE_PATH = "radiologie_db"

def load_txt_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    return chunks

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def build_vectorstore(chunks):
    st.info("🚀 Nieuwe vectorstore wordt opgebouwd...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    texts = [chunk.page_content for chunk in chunks]

    progress = st.progress(0)
    total = len(texts)

    embedded_texts = []
    embedded_vectors = []

    for i, text in enumerate(texts):
        emb = embedding_model.embed_documents([text])  # embed één tekst
        embedded_texts.append(text)
        embedded_vectors.append(emb[0])  # eerste (en enige) vector
        progress.progress((i + 1) / total)

    # Combineer tekst + embedding en maak de vectorstore
    vectordb = FAISS.from_embeddings(
        [(text, vector) for text, vector in zip(embedded_texts, embedded_vectors)],
        embedding=embedding_model
    )

    vectordb.save_local(VECTORSTORE_PATH)
    st.success("✅ Vectorstore succesvol opgeslagen!")
    return vectordb

def load_vectorstore():
    st.info("📁 Bestaande vectorstore wordt geladen...")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        allow_dangerous_deserialization=True  
    )

# Streamlit-app
st.title("📚 Radiologie Assistent")

if not os.path.exists(VECTORSTORE_PATH):
    if os.path.exists("uploads") and any(f.endswith(".txt") for f in os.listdir("uploads")):
        txt_files = [f for f in os.listdir("uploads") if f.endswith(".txt")]
        selected_file = st.selectbox("Selecteer .txt-bestand", txt_files)
        if st.button("📂 Bouw vectorstore"):
            chunks = load_txt_chunks(os.path.join("uploads", selected_file))
            vectordb = build_vectorstore(chunks)
    else:
        st.warning("⚠️ Geen .txt-bestanden gevonden in map 'uploads'.")
else:
    vectordb = load_vectorstore()

