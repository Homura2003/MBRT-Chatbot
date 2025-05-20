import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
import json
from typing import List, Optional, Tuple
import time

# Get HuggingFace token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]

VECTORSTORE_PATH = "radiologie_db"
DOCUMENTS_FILE = os.path.join(VECTORSTORE_PATH, "documents.json")

# Initialize session state for chat history and rate limiting
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Will store tuples of (human_message, ai_message)
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

def ensure_directory_exists():
    """Zorg ervoor dat alle benodigde mappen bestaan."""
    for path in [VECTORSTORE_PATH, "uploads"]:
        if not os.path.exists(path):
            os.makedirs(path)

@st.cache_resource
def get_embedding_model():
    """Cache het embedding model om herladen te voorkomen."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def load_txt_chunks(file_path: str) -> List[Document]:
    """Laad en split tekst uit een bestand."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = Document(page_content=text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents([doc])
        return chunks
    except Exception as e:
        st.error(f"Fout bij het laden van het bestand: {str(e)}")
        return []

def build_vectorstore(chunks: List[Document]) -> Optional[InMemoryVectorStore]:
    """Bouw de vectorstore op."""
    try:
        st.info("🚀 Nieuwe vectorstore wordt opgebouwd...")
        
        embedding_model = get_embedding_model()
        
        # Maak in-memory vector store van documenten
        vectordb = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        
        # Sla documenten op
        ensure_directory_exists()
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump([doc.dict() for doc in chunks], f, ensure_ascii=False, indent=2)
        
        st.success("✅ Vectorstore succesvol opgeslagen!")
        return vectordb
    except Exception as e:
        st.error(f"Fout bij het opbouwen van de vectorstore: {str(e)}")
        return None

def load_vectorstore() -> Optional[InMemoryVectorStore]:
    """Laad de bestaande vectorstore."""
    try:
        st.info("📁 Bestaande vectorstore wordt geladen...")

        if not os.path.exists(DOCUMENTS_FILE):
            st.error("❌ Geen vectorstore gevonden!")
            return None

        embedding_model = get_embedding_model()
        
        # Laad documenten
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
            chunks = [Document(**doc) for doc in docs_data]
        
        # Maak nieuwe vector store
        vectordb = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        
        st.success("✅ Vectorstore succesvol geladen!")
        return vectordb
    except Exception as e:
        st.error(f"❌ Fout bij het laden van de vectorstore: {str(e)}")
        return None

def rate_limit():
    """Implementeer rate limiting voor API calls."""
    current_time = time.time()
    if current_time - st.session_state.last_request_time < 1:  # 1 seconde cooldown
        time.sleep(1)
    st.session_state.last_request_time = current_time

# Streamlit-app
st.title("📚 Radiologie Assistent")

# Zorg ervoor dat de mappen bestaan
ensure_directory_exists()

# Initialiseer vector store
vectordb = None

# Laad of maak vector store
if not os.path.exists(DOCUMENTS_FILE):
    if os.path.exists("uploads") and any(f.endswith(".txt") for f in os.listdir("uploads")):
        txt_files = [f for f in os.listdir("uploads") if f.endswith(".txt")]
        selected_file = st.selectbox("Selecteer .txt-bestand", txt_files)
        if st.button("📂 Bouw vectorstore"):
            chunks = load_txt_chunks(os.path.join("uploads", selected_file))
            if chunks:
                vectordb = build_vectorstore(chunks)
    else:
        st.warning("⚠️ Geen .txt-bestanden gevonden in map 'uploads'.")
else:
    vectordb = load_vectorstore()

# Chat interface
if vectordb is not None:
    try:
        # Initialiseer de LLM met HuggingFace token
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=HUGGINGFACE_TOKEN,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )

        # Maak de conversationele chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # Toon chat berichten
        for human_msg, ai_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(human_msg)
            with st.chat_message("assistant"):
                st.write(ai_msg)

        # Chat input
        if prompt := st.chat_input("Stel een vraag over de radiologie documenten"):
            # Rate limiting
            rate_limit()
            
            # Toon gebruikersbericht
            with st.chat_message("user"):
                st.write(prompt)

            # Krijg antwoord van de chain
            with st.chat_message("assistant"):
                with st.spinner("Denken..."):
                    try:
                        response = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                        st.write(response["answer"])
                        
                        # Voeg het gesprek toe aan de chat geschiedenis
                        st.session_state.chat_history.append((prompt, response["answer"]))

                        # Toon bron documenten
                        with st.expander("Bronnen"):
                            for doc in response["source_documents"]:
                                st.write(doc.page_content)
                                st.write("---")
                    except Exception as e:
                        st.error(f"Fout bij het genereren van antwoord: {str(e)}")
    except Exception as e:
        st.error(f"Fout bij het initialiseren van chat interface: {str(e)}")



