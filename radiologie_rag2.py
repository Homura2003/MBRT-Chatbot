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

def build_vectorstore(chunks):
    st.info("🚀 Nieuwe vectorstore wordt opgebouwd...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create FAISS vector store from documents
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # Save the vector store
    vectordb.save_local(VECTORSTORE_PATH)
    st.success("✅ Vectorstore succesvol opgeslagen!")
    return vectordb

def load_vectorstore():
    st.info("📁 Bestaande vectorstore wordt geladen...")

    if os.path.exists(VECTORSTORE_PATH):
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True  
        )
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
