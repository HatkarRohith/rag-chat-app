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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    except:
        pass

if not api_key:
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Please configure the Groq API Key to continue.")
        st.stop()

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Persistent storage
    DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_persistent")
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )
    return embedder, chroma_client

try:
    client = Groq(api_key=api_key)
    embedder, chroma_client = load_resources()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

# --- SIDEBAR ---
with st.sidebar:
    st.divider()
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Process & Train")

# --- PROCESSING LOGIC ---
if process_btn and uploaded_files:
    status = st.empty()
    progress_bar = st.progress(0)
    status.text("Processing files...")
    
    # 1. Clear old memory so we don't mix up files
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass
    collection = get_collection()
    
    all_chunks = []
    for file in uploaded_files:
        text = ""
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.endswith(".txt"):
                text = file.read().decode("utf-8")
            
            # Smart Chunking: 1000 chars is roughly 1 paragraph
            chunk_size = 1000
            overlap = 200
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]
                if len(chunk) > 50:
                    all_chunks.append(chunk)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            continue
            
    # Batch Insert
    BATCH_SIZE = 200
    total_chunks = len(all_chunks)
    if total_chunks > 0:
        status.text(f"Embedding {total_chunks} blocks...")
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            embeddings = list(embedder.embed(batch))
            embeddings = [e.tolist() for e in embeddings]
            ids = [f"id_{i+j}" for j in range(len(batch))]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
            progress_bar.progress(min((i + BATCH_SIZE) / total_chunks, 1.0))
        status.success(f"Success! Analyzed {total_chunks} blocks.")
    else:
        status.error("No valid text found.")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    collection = get_collection()
    try:
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        
        # --- THE "PERFECT FIT" LOGIC ---
        # We retrieve 10 chunks (roughly 10,000 characters or 2,500 tokens).
        # This gives us plenty of text for comparison but stays safely under the 6,000 limit.
        results = collection.query(query_embeddings=[q_embed], n_results=10)
        
        if results['documents'] and results['documents'][0]:
            context = "\n\n".join(results['documents'][0])
            
            sys_prompt = f"""
            You are a helpful expert assistant. Your task is to answer the user's question based ONLY on the provided context.
            If the answer is not in the context, say "I don't know".
            
            Context:
            {context}
            
            Question:
            {prompt}
            """
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sys_prompt}]
            )
            answer = response.choices[0].message.content
        else:
            answer = "I couldn't find relevant info in the docs."
            
    except Exception as e:
        # If it STILL crashes, we catch it and tell you nicely
        if "413" in str(e) or "rate_limit" in str(e):
            answer = "⚠️ **Text too large for free tier.** Please try asking about specific files one by one (e.g., 'Summarize PDF A' then 'Summarize PDF B')."
        else:
            answer = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
