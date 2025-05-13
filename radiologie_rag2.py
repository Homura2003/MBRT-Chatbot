import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

VECTORSTORE_PATH = "radiologie_db"

def load_txt_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    return chunks

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

    # Gebruik Chroma in plaats van FAISS
    vectordb = Chroma.from_documents(
        embedded_texts,
        embedded_vectors,
        embedding=embedding_model
    )

    vectordb.save_local(VECTORSTORE_PATH)
    st.success("✅ Vectorstore succesvol opgeslagen!")
    return vectordb

def load_vectorstore():
    st.info("📁 Bestaande vectorstore wordt geladen...")
    
    # Controleer of het pad naar de vectorstore bestaat
    if os.path.exists(VECTORSTORE_PATH):
        # Gebruik de juiste methode om de vectorstore te laden
        vectordb = Chroma.load(VECTORSTORE_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        st.success("✅ Vectorstore succesvol geladen!")
        return vectordb
    else:
        st.error("❌ Geen vectorstore gevonden!")
        return None

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

