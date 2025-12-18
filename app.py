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

# --- AUTHENTICATION LOGIC ---
# 1. Try to get key from Streamlit Cloud Secrets first
api_key = st.secrets.get("GROQ_API_KEY")

# 2. If not found in Secrets, try local .env file (for local testing)
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    except:
        pass

# 3. If STILL not found, ask the user to enter it manually
if not api_key:
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Groq API Key", type="password")

# Stop the app if we still don't have a key
if not api_key:
    st.warning("Please configure the Groq API Key to continue.")
    st.stop()

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.divider()
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    process_btn = st.button("Process & Train")

# --- INITIALIZE CLIENTS ---
try:
    client = Groq(api_key=api_key)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# --- SETUP DATABASE ---
# We use a temporary directory for the cloud to avoid permission errors
DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
chroma_client = chromadb.PersistentClient(
    path=DB_DIR, 
    settings=Settings(anonymized_telemetry=False)
)

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

# --- DOCUMENT PROCESSING LOGIC ---
if process_btn and uploaded_files:
    status = st.empty()
    progress_bar = st.progress(0)
    
    status.text("Processing files...")
    
    # 1. Reset Collection (Clear old data)
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass  # Collection might not exist yet
    collection = get_collection()
    
    # 2. Read and Text Extraction
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
            
            # Simple chunking by newlines
            chunks = [line.strip() for line in text.split('\n') if line.strip()]
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")
            continue
        
    # 3. Batch Embed & Insert (Safety Batching)
    BATCH_SIZE = 500  # Safe limit to prevent crashing
    total_chunks = len(all_chunks)
    
    if total_chunks > 0:
        status.text(f"Embedding {total_chunks} chunks...")
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            
            # Generate Embeddings
            embeddings = list(embedder.embed(batch))
            embeddings = [e.tolist() for e in embeddings]
            
            # Create unique IDs
            ids = [f"id_{i+j}" for j in range(len(batch))]
            
            # Add to Database
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
            
            # Update Progress Bar
            progress_bar.progress(min((i + BATCH_SIZE) / total_chunks, 1.0))
        
        status.success(f"Success! Processed {total_chunks} chunks. You can now chat.")
    else:
        status.error("No valid text found in the uploaded documents.")

# ... inside the chat interface ...
    
    # Search ChromaDB
    # CHANGED: Increased from 5 to 15 so it can see multiple documents at once
    results = collection.query(query_embeddings=[q_embed], n_results=15)
    
    if results['documents'] and results['documents'][0]:
        context = "\n".join(results['documents'][0])
        
        # 3. Generate Answer with Groq
        # ... rest of the code ...

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Retrieve Context
    collection = get_collection()
    try:
        # Generate embedding for the question
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        
        # Search ChromaDB
        results = collection.query(query_embeddings=[q_embed], n_results=5)
        
        if results['documents'] and results['documents'][0]:
            context = "\n".join(results['documents'][0])
            
            # 3. Generate Answer with Groq
            sys_prompt = f"""
            You are a helpful assistant. Use the following context to answer the user's question.
            If the answer is not in the context, say you don't know.
            
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
            answer = "I couldn't find any relevant information in the uploaded documents."
            
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # 4. Display Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

