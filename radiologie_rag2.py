import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import OllamaEmbeddings
import json

VECTORSTORE_PATH = "radiologie_db"
DOCUMENTS_FILE = os.path.join(VECTORSTORE_PATH, "documents.json")

def ensure_directory_exists():
    if not os.path.exists(VECTORSTORE_PATH):
        os.makedirs(VECTORSTORE_PATH)

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
    
    # Create in-memory vector store from documents
    vectordb = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # Save the embeddings and documents
    ensure_directory_exists()
    
    # Save documents
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in chunks], f, ensure_ascii=False, indent=2)
    
    st.success("✅ Vectorstore succesvol opgeslagen!")
    return vectordb

def load_vectorstore():
    st.info("📁 Bestaande vectorstore wordt geladen...")

    if not os.path.exists(DOCUMENTS_FILE):
        st.error("❌ Geen vectorstore gevonden!")
        return None

    try:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        
        # Load documents
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            chunks = [Document(**doc) for doc in docs_data]
        
        # Create new vector store
        vectordb = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        
        st.success("✅ Vectorstore succesvol geladen!")
        return vectordb
    except Exception as e:
        st.error(f"❌ Fout bij het laden van de vectorstore: {str(e)}")
        return None

# Streamlit-app
st.title("📚 Radiologie Assistent")

# Ensure the uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists(DOCUMENTS_FILE):
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

