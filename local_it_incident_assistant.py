"""
Local IT Incident Resolution Assistant
Single-file prototype using Streamlit for local UI, OpenAI Chat API for core intelligence,
and local embeddings + FAISS for retrieval.

How it works:
 - Ingest PDFs from a folder and extract text (pdfplumber)
 - Ingest an Excel file and parse five required columns: Number, Short description, Description, Root cause description, Resolution note
 - Build sentence-transformers embeddings locally and index with FAISS
 - Streamlit UI: chat-like; user submits new incident text -> system searches the vector index, finds similar incidents and pdf passages -> sends retrieved context to OpenAI Chat API (GPT-4) with an instruction to: list top 4-5 probable root causes, provide a detailed resolution plan, and cite any similar historical incident Numbers

Requirements:
 pip install -r requirements.txt
 (See run instructions at bottom)

Set environment variable OPENAI_API_KEY before running.

"""

import os
import io
import time
import json
import pickle
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import google.generativeai as genai

os.environ["GEMINI_API_KEY"] = "AIzaSyCVzO-I8dk4B1JGj9J5MUzvG-jqNwTM-uo"
# Try to import faiss; fallback to sklearn if unavailable
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    _FAISS_AVAILABLE = False



# ---------------------- Configuration ----------------------
# ---------------------- Configuration ----------------------
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # small local model from sentence-transformers
EMBED_DIM = 384  # embedding dimension for the model above
INDEX_PATH = 'vector_index.faiss' if _FAISS_AVAILABLE else 'vector_index.pkl'
DOCS_STORE = 'documents_store.pkl'
GEMINI_MODEL = 'gemini-2.5-pro'  # or 'gemini-pro' if flash not available
TOP_K = 6


# ---------------------- Utilities ----------------------


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------------------- Ingestion ----------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                text.append(ptext)
    return '\n'.join(text)


def ingest_pdfs_from_upload(files) -> List[Dict[str, Any]]:
    docs = []
    for f in files:
        content = f.read()
        text = extract_text_from_pdf_bytes(content)
        if not text.strip():
            continue
        docs.append({
            'type': 'pdf',
            'source': f.name,
            'text': text[:20000],  # limit size per doc
        })
    return docs


def ingest_pdfs_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    docs = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.pdf'):
            with open(os.path.join(folder_path, fname), 'rb') as f:
                text = extract_text_from_pdf_bytes(f.read())
                docs.append({'type': 'pdf', 'source': fname, 'text': text[:20000]})
    return docs


def ingest_excel(file) -> List[Dict[str, Any]]:
    df = pd.read_excel(file)
    required_cols = ['Number', 'Short Description', 'Description', 'Root Cause Description', 'Resolution Notes']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")
    incidents = []
    for _, row in df.iterrows():
        incidents.append({
            'type': 'incident',
            'Number': str(row['Number']),
            'Short Description': str(row['Short Description']),
            'Description': str(row['Description']),
            'Root Cause Description': str(row['Root Cause Description']),
            'Resolution Notes': str(row['Resolution Notes'])
        })
    return incidents

# ---------------------- Indexing / Embeddings ----------------------

def build_corpus_and_embeddings(docs: List[Dict[str, Any]], model: SentenceTransformer) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    corpus = []
    texts = []
    metadata = []
    for d in docs:
        if d['type'] == 'pdf':
            # split into chunks
            text = d['text']
            chunk_size = 800
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                corpus.append({'text': chunk, 'source': d['source'], 'type': 'pdf'})
                texts.append(chunk)
                metadata.append({'source': d['source'], 'type': 'pdf'})
        else:
            # incident entry: use Short description + Description
            text = (d.get('Short description','') + '\n' + d.get('Description','')).strip()
            corpus.append({'text': text, 'incident': d['Number'], 'type': 'incident', 'root_cause': d.get('Root cause description',''), 'resolution': d.get('Resolution note','')})
            texts.append(text)
            metadata.append({'incident': d['Number'], 'type': 'incident'})

    # compute embeddings
    batch_size = 64
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    embeddings = np.array(embeddings).astype('float32')
    return corpus, embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_index_and_docs(index, corpus, embeddings):
    if _FAISS_AVAILABLE:
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_STORE, 'wb') as f:
            pickle.dump({'corpus': corpus}, f)
    else:
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'corpus': corpus}, f)


def load_index_and_docs():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_STORE):
        return None, None, None
    if _FAISS_AVAILABLE:
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_STORE, 'rb') as f:
            store = pickle.load(f)
        corpus = store['corpus']
        return index, corpus, None
    else:
        with open(INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
        embeddings = data['embeddings']
        corpus = data['corpus']
        nbrs = NearestNeighbors(n_neighbors=TOP_K, metric='cosine').fit(embeddings)
        return nbrs, corpus, embeddings

# ---------------------- Retrieval ----------------------

def query_index(query: str, model: SentenceTransformer, index_obj, corpus, embeddings_obj=None, top_k=TOP_K) -> List[Tuple[float, Dict[str,Any]]]:
    q_emb = model.encode([query])[0].astype('float32')
    if _FAISS_AVAILABLE:
        faiss.normalize_L2(q_emb.reshape(1, -1))
        D, I = index_obj.search(q_emb.reshape(1, -1), top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
    else:
        # sklearn NearestNeighbors returns distances (cosine) so convert
        distances, idxs = index_obj.kneighbors([q_emb], n_neighbors=top_k)
        scores = (1 - distances[0]).tolist()
        idxs = idxs[0].tolist()
    results = []
    for s, i in zip(scores, idxs):
        if i < 0 or i >= len(corpus):
            continue
        results.append((float(s), corpus[i]))
    return results

# ---------------------- Prompting ----------------------

def craft_prompt(user_incident: str, retrieved: List[Tuple[float, Dict[str,Any]]]) -> List[Dict[str,str]]:
    system = {
        'role': 'system',
        'content': (
            'You are an IT Incident Resolution Assistant. Given a new incident description and retrieved historical context (technical docs and past incidents), ' 
            'produce: 1) the most probable root cause(s) (up to 5) ranked with brief justification; 2) a clear, step-by-step remediation plan for immediate action and verification; ' 
            '3) list the Incident Number(s) of any historical incidents that are highly similar, and explain why they match. Use only the provided retrieved context when citing incident Numbers. ' 
            'If insufficient info, say what additional logs or checks are needed.'
        )
    }
    user_intro = {'role': 'user', 'content': f'New incident description:\n"""{user_incident}"""\n\nRetrieved context follows below.'}
    context_chunks = []
    for score, doc in retrieved:
        header = f'[score={score:.3f}]'
        if doc.get('type') == 'incident':
            short_desc = doc.get('text')[:800].replace('\n', ' ')
            text = f"{header} INCIDENT {doc.get('incident')} | Short+Desc: {short_desc} | RootCause: {doc.get('root_cause')[:300]} | Resolution: {doc.get('resolution')[:300]}"

        else:
            pdf_snippet = doc.get('text')[:800].replace('\n', ' ')
            text = f"{header} PDF {doc.get('source')} snippet: {pdf_snippet}"

        context_chunks.append(text)
    assistant_request = {'role': 'user', 'content': '\n\n'.join([user_intro['content']] + context_chunks)}
    return [system, assistant_request]

def call_gemini_chat(messages: list[dict[str, str]]) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"{role.upper()}: {content}\n"

    # Works in all current regions
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else str(response)

# ---------------------- Streamlit UI ----------------------

def main():
    st.set_page_config(page_title='Local IT Incident Resolution Assistant', layout='wide')
    st.title('Local IT Incident Resolution Assistant (Prototype)')

    model = load_embedding_model()

    # Sidebar: data ingestion
    with st.sidebar.expander('1) Ingest data (PDFs + Excel)'):
        st.write('Upload PDF files or point to a local folder, and upload the historical incidents Excel file.')
        pdf_upload = st.file_uploader('Upload PDFs (multiple)', accept_multiple_files=True, type=['pdf'])
        pdf_folder = st.text_input('Or enter local folder path containing PDFs (optional)')
        excel_file = st.file_uploader('Upload Excel with incident logs', type=['xls','xlsx'])
        if st.button('Ingest and build index'):
            all_docs = []
            try:
                if pdf_upload:
                    docs = ingest_pdfs_from_upload(pdf_upload)
                    all_docs.extend(docs)
                if pdf_folder:
                    docs = ingest_pdfs_from_folder(pdf_folder)
                    all_docs.extend(docs)
                if excel_file:
                    incidents = ingest_excel(excel_file)
                    all_docs.extend(incidents)
                if not all_docs:
                    st.warning('No documents found to ingest.')
                else:
                    with st.spinner('Computing embeddings and building index...'):
                        corpus, embeddings = build_corpus_and_embeddings(all_docs, model)
                        if _FAISS_AVAILABLE:
                            index = build_faiss_index(embeddings)
                            save_index_and_docs(index, corpus, embeddings)
                        else:
                            # save embeddings + corpus for sklearn
                            with open(INDEX_PATH, 'wb') as f:
                                pickle.dump({'embeddings': embeddings, 'corpus': corpus}, f)
                            with open(DOCS_STORE, 'wb') as f:
                                pickle.dump({'corpus': corpus}, f)
                        st.success('Index built and saved locally.')
            except Exception as e:
                st.exception(e)

    # Load index if exists
    index_obj, corpus, embeddings_obj = load_index_and_docs()
    if index_obj is None or corpus is None:
        st.info('No local index found. Ingest data first through the sidebar.')

    chat_col, meta_col = st.columns([3,1])
    with chat_col:
        st.subheader('Chat')
        chat_log = st.container()
        user_input = st.text_area('Describe the new incident (concise, include symptoms, timestamps, error messages):', height=150)
        if st.button('Analyze incident'):
            if not user_input.strip():
                st.warning('Enter an incident description.')
            elif index_obj is None:
                st.warning('Index not available. Ingest data first.')
            else:
                with st.spinner('Retrieving similar context and calling reasoning model...'):
                    retrieved = query_index(user_input, model, index_obj, corpus, embeddings_obj, top_k=TOP_K)
                    messages = craft_prompt(user_input, retrieved)
                    try:
                        answer = answer = call_gemini_chat(messages)
                    except Exception as e:
                        st.exception(e)
                        return
                st.markdown('**AI Response**')
                st.write(answer)
                # show retrieved incident Numbers
                sim_incidents = [doc for score, doc in retrieved if doc.get('type') == 'incident']
                if sim_incidents:
                    st.markdown('**Similar historical incidents (from index):**')
                    for s, doc in retrieved:
                        if doc.get('type') == 'incident':
                            short_text = doc.get('text')[:200].replace('\n', ' ')
                            st.write(f"Incident Number: {doc.get('incident')} — Short: {short_text} — score {s:.3f}")

    with meta_col:
        st.subheader('Index info')
        if corpus:
            n_incidents = sum(1 for d in corpus if d.get('type')=='incident')
            n_pdf_chunks = sum(1 for d in corpus if d.get('type')=='pdf')
            st.write(f'Total indexed items: {len(corpus)}')
            st.write(f'Incident entries: {n_incidents}')
            st.write(f'PDF chunks: {n_pdf_chunks}')
        st.markdown('---')
        st.write('Controls')
        if st.button('Clear saved index'):
            for p in [INDEX_PATH, DOCS_STORE]:
                if os.path.exists(p):
                    os.remove(p)
            st.success('Cleared')

if __name__ == '__main__':
    main()

# ---------------------- Run instructions ----------------------
# Save this file and run: streamlit run local_it_incident_assistant.py
# Requirements (example):
# pip install streamlit pandas pdfplumber sentence-transformers faiss-cpu openai scikit-learn
# Set OPENAI_API_KEY environment variable before running: export OPENAI_API_KEY="sk-..."
# Then open http://localhost:8501 in your browser.
