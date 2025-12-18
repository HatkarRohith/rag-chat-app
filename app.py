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
        api_key = st.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Enter Groq API Key to continue.")
        st.stop()

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_persistent")
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )
    return embedder, chroma_client

client = Groq(api_key=api_key)
embedder, chroma_client = load_resources()

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

# --- SIDEBAR ---
with st.sidebar:
    st.divider()
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Process & Train")

# --- PROCESS FILES ---
if process_btn and uploaded_files:
    status = st.empty()
    status.text("Processing...")
    
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
            
            # --- ECONOMY CHUNKING ---
            # Smaller chunks (800 chars) to save tokens
            chunk_size = 800
            overlap = 100
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]
                if len(chunk) > 50:
                    all_chunks.append(chunk)
        except:
            continue
            
    if all_chunks:
        # Embed in small batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            embeddings = [e.tolist() for e in list(embedder.embed(batch))]
            ids = [f"id_{i+j}" for j in range(len(batch))]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
        status.success(f"Ready! Processed {len(all_chunks)} chunks.")
    else:
        status.error("No text found.")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    collection = get_collection()
    try:
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        
        # --- ECONOMY SEARCH ---
        # Retrieve only 5 chunks (approx. 1000 tokens)
        # This leaves 5000 tokens free for the answer + buffer
        results = collection.query(query_embeddings=[q_embed], n_results=5)
        
        if results['documents'] and results['documents'][0]:
            context = "\n".join(results['documents'][0])
            
            # Strict character limit (roughly 2000 tokens)
            if len(context) > 8000:
                context = context[:8000]
            
            sys_prompt = f"Answer using this context:\n\n{context}\n\nQuestion: {prompt}"
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sys_prompt}],
                max_tokens=500  # Prevent long rambling answers that burn credits
            )
            answer = response.choices[0].message.content
        else:
            answer = "I couldn't find relevant info."
            
    except Exception as e:
        if "rate_limit" in str(e).lower():
            answer = "⏳ **Rate Limit Hit:** You are asking too fast for the Free Tier! Please wait 60 seconds and try again."
        else:
            answer = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
