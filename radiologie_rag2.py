import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
import json

# Get HuggingFace token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

VECTORSTORE_PATH = "radiologie_db"
DOCUMENTS_FILE = os.path.join(VECTORSTORE_PATH, "documents.json")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # Initialize HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
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
        # Initialize HuggingFace embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
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

# Initialize vector store
vectordb = None

# Load or create vector store
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

# Chat interface
if vectordb is not None:
    # Initialize the LLM with HuggingFace token
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=HUGGINGFACE_TOKEN,
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Stel een vraag over de radiologie documenten"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from the chain
        with st.chat_message("assistant"):
            with st.spinner("Denken..."):
                response = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                st.write(response["answer"])
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

                # Display source documents
                with st.expander("Bronnen"):
                    for doc in response["source_documents"]:
                        st.write(doc.page_content)
                        st.write("---")

