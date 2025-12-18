__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# --- SIDEBAR & API KEY ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password")
    st.divider()
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Process & Train")

# --- MAIN APP LOGIC ---
if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Initialize Clients
client = Groq(api_key=api_key)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Setup Storage (ChromaDB)
# We use a temporary directory for the cloud so it doesn't break permissions
DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

if process_btn and uploaded_files:
    status = st.empty()
    status.text("Processing files...")
    progress = st.progress(0)
    
    # 1. Reset Collection
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass
    collection = get_collection()
    
    # 2. Read Files
    all_chunks = []
    for file in uploaded_files:
        text = ""
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            text = file.read().decode("utf-8")
        
        # Simple chunking
        chunks = [line.strip() for line in text.split('\n') if line.strip()]
        all_chunks.extend(chunks)
        
    # 3. Batch Ingest
    BATCH_SIZE = 500
    total_chunks = len(all_chunks)
    
    if total_chunks > 0:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            embeddings = list(embedder.embed(batch))
            embeddings = [e.tolist() for e in embeddings] # convert to list
            ids = [f"id_{i+j}" for j in range(len(batch))]
            
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
            progress.progress(min((i + BATCH_SIZE) / total_chunks, 1.0))
        
        status.success(f"Processed {total_chunks} chunks! You can now ask questions.")
    else:
        status.error("No text found in files.")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # RAG Search
    collection = get_collection()
    try:
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        results = collection.query(query_embeddings=[q_embed], n_results=5)
        
        if results['documents']:
            context = "\n".join(results['documents'][0])
            # Generative Answer
            sys_prompt = f"Use this context to answer: {context}\n\nQuestion: {prompt}"
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sys_prompt}]
            )
            answer = response.choices[0].message.content
        else:
            answer = "I couldn't find any relevant information in the documents."
            
    except Exception as e:
        answer = f"Error during search: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
