import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — Deep-space / editorial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #07080d;
    --surface:   #0f1117;
    --panel:     #13151e;
    --border:    #1e2130;
    --accent:    #6c63ff;
    --accent2:   #ff6584;
    --accent3:   #43e8ad;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --user-bg:   #1a1d2e;
    --bot-bg:    #111420;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.4rem 1rem 1.4rem;
}

/* ── Sidebar title ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
}
.sidebar-brand .logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 20px rgba(108,99,255,0.4);
}
.sidebar-brand h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px;
    color: var(--text) !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
.sidebar-brand span { color: var(--accent); }

/* ── Upload zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: block;
}

[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileDropzone"] p { color: var(--muted) !important; font-size: 0.8rem; }
[data-testid="stFileDropzone"] small { color: var(--muted) !important; }
[data-testid="stFileDropzone"] svg { fill: var(--accent) !important; }

/* ── Uploaded file badge ── */
.file-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.8rem;
    color: var(--text);
    word-break: break-all;
}
.file-badge .icon { color: var(--accent3); font-size: 1rem; }

/* ── Stats pill ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: var(--muted);
}
.stat-pill b { color: var(--accent3); }

/* ── Clear button ── */
.stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    padding: 0.5rem !important;
    transition: all 0.2s !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    border-color: var(--accent2) !important;
    color: var(--accent2) !important;
    background: rgba(255,101,132,0.05) !important;
}

/* ── Main chat area ── */
.main-wrap {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--bg);
}

/* ── Hero header ── */
.hero {
    padding: 2.5rem 3rem 1.5rem 3rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0d0e18 0%, var(--bg) 100%);
}
.hero-eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 0 0.3rem 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
}

/* ── Empty state ── */
.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    gap: 1.5rem;
}
.empty-orb {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #2d2860, #0d0e18);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    box-shadow: 0 0 40px rgba(108,99,255,0.2);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 30px rgba(108,99,255,0.2); }
    50%       { box-shadow: 0 0 60px rgba(108,99,255,0.4); }
}
.empty-msg {
    text-align: center;
    color: var(--muted);
    font-size: 0.9rem;
    max-width: 320px;
    line-height: 1.6;
}
.empty-msg strong {
    color: var(--text);
    display: block;
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    margin-bottom: 0.3rem;
}

/* ── Suggestion chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 0.5rem;
}
.chip {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(108,99,255,0.08);
}

/* ── Chat messages ── */
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 3rem;
    scroll-behavior: smooth;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.2rem !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessageContent"] {
    border-radius: var(--radius) !important;
    padding: 0.9rem 1.2rem !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    max-width: 75% !important;
}

/* Avatar */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border-radius: 10px !important;
    color: white !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--accent3), var(--accent)) !important;
    border-radius: 10px !important;
    color: #07080d !important;
}

/* ── Chat input bar ── */
.input-area {
    padding: 1rem 3rem 1.5rem 3rem;
    border-top: 1px solid var(--border);
    background: linear-gradient(0deg, #0a0b12 0%, var(--bg) 100%);
}

[data-testid="stChatInput"] textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: border-color 0.2s !important;
    caret-color: var(--accent) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.12) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alert / warning ── */
.stAlert {
    background: rgba(255,101,132,0.08) !important;
    border: 1px solid rgba(255,101,132,0.25) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

/* ── Thinking animation for assistant ── */
.thinking {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.7rem 0;
}
.thinking span {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: bounce 1.2s infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; background: var(--accent2); }
.thinking span:nth-child(3) { animation-delay: 0.4s; background: var(--accent3); }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
    40%            { transform: scale(1.1); opacity: 1; }
}

/* ── Source badge ── */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 0.7rem;
    background: rgba(67,232,173,0.08);
    border: 1px solid rgba(67,232,173,0.2);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: var(--accent3);
}

/* ── Dividers & misc ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "doc_name"    not in st.session_state: st.session_state.doc_name    = None
if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0


# ─────────────────────────────────────────────
#  PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore, len(splits)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🧠</div>
        <h1>Doc<span>Mind</span></h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="upload-label">📎 Source Document</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type="pdf", label_visibility="collapsed")

    # Show file badge once uploaded
    if uploaded_file:
        st.markdown(f"""
        <div class="file-badge">
            <span class="icon">📄</span>
            <span>{uploaded_file.name}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.doc_name:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">Chunks: <b>{st.session_state.chunk_count}</b></div>
            <div class="stat-pill">Model: <b>llama3.2</b></div>
            <div class="stat-pill">RAG: <b>ON</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages    = []
        st.session_state.vectorstore = None
        st.session_state.doc_name    = None
        st.session_state.chunk_count = 0
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem; font-size:0.72rem; color:#3d4250; line-height:1.7;">
        <b style="color:#4a5060;">How it works</b><br>
        1. Upload any PDF<br>
        2. Wait for indexing<br>
        3. Ask questions naturally<br><br>
        Powered by <b style="color:#6c63ff;">Ollama</b> + <b style="color:#43e8ad;">ChromaDB</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PROCESS PDF (once per upload)
# ─────────────────────────────────────────────
if uploaded_file and (st.session_state.doc_name != uploaded_file.name):
    with st.spinner("⚡ Indexing your document…"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                vs, chunks = process_pdf(tmp.name)

            st.session_state.vectorstore = vs
            st.session_state.doc_name    = uploaded_file.name
            st.session_state.chunk_count = chunks
            st.session_state.messages    = []   # fresh chat for new doc
        except Exception as e:
            st.error(f"❌ Failed to index document: {e}")


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
# Hero header
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Document Intelligence</div>
    <h1 class="hero-title">Ask anything about your <span>PDF</span></h1>
    <p class="hero-sub">Upload a document → get instant, grounded answers with no hallucinations.</p>
</div>
""", unsafe_allow_html=True)


# ── Empty state ───────────────────────────────
if not st.session_state.messages and not st.session_state.vectorstore:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-orb">🧠</div>
        <div class="empty-msg">
            <strong>No document loaded yet</strong>
            Upload a PDF from the sidebar and start asking questions instantly.
        </div>
        <div class="chip-row">
            <div class="chip">Summarise the document</div>
            <div class="chip">What are the key findings?</div>
            <div class="chip">List the main topics</div>
            <div class="chip">Explain section 1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.messages and st.session_state.vectorstore:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-orb">✅</div>
        <div class="empty-msg">
            <strong>Document ready!</strong>
            Your PDF has been indexed. Ask me anything about it below.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Chat history ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ── Chat input ────────────────────────────────
if st.session_state.vectorstore:
    prompt = st.chat_input("Ask anything about your document…")
else:
    prompt = st.chat_input("Upload a PDF first to start chatting…", disabled=True)


# ── Generate response ─────────────────────────
if prompt:
    # Guard
    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Store & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ── Retrieval ──
        try:
            relevant_docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
            context = "\n\n".join([d.page_content for d in relevant_docs])
        except Exception as e:
            err = f"❌ Retrieval error: {e}"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

        if not context.strip():
            msg = "⚠️ No relevant content found in the document for that question."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        # ── Build prompt ──
        full_prompt = f"""You are DocMind, an expert document analyst AI. \
Answer the user's question thoroughly and clearly using ONLY the context provided. \
If the answer is not in the context, say exactly: "I couldn't find that in the document."

Format your response with clear structure when appropriate — use bullet points or short paragraphs.

Context:
\"\"\"
{context}
\"\"\"

Question: {prompt}

Answer:"""

        # ── Stream response ──
        response_placeholder = st.empty()
        full_response = ""

        try:
            llm = ChatOllama(model="llama3.2", temperature=0.1)

            for chunk in llm.stream([HumanMessage(content=full_prompt)]):
                if hasattr(chunk, "content") and chunk.content is not None:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

        except Exception as e:
            full_response = f"❌ LLM error: {str(e)}\n\n*Make sure Ollama is running: `ollama serve` and the model is pulled: `ollama pull llama3.2`*"
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.stop()

        if not full_response.strip():
            full_response = "⚠️ No response generated. Try rephrasing your question."

        # Final render (remove cursor)
        source_note = f'\n\n<div class="source-badge">📎 Based on {len(relevant_docs)} chunks from <b>{st.session_state.doc_name}</b></div>'
        final = full_response + source_note
        response_placeholder.markdown(final, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": final})