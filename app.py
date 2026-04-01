import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
:root {
    --bg:#07080d; --surface:#0f1117; --panel:#13151e; --border:#1e2130;
    --accent:#6c63ff; --accent2:#ff6584; --accent3:#43e8ad;
    --text:#eceef5; --muted:#7c8499; --r:14px;
}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;background:var(--bg)!important;color:var(--text)!important;font-size:16px!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}

section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);min-width:280px!important;}
section[data-testid="stSidebar"]>div:first-child{padding:2rem 1.5rem;}

.brand{display:flex;align-items:center;gap:12px;margin-bottom:2rem;}
.brand .logo{width:42px;height:42px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 20px rgba(108,99,255,.4);}
.brand h1{font-size:1.45rem!important;font-weight:800!important;margin:0!important;padding:0!important;line-height:1!important;color:var(--text)!important;}
.brand h1 span{color:var(--accent);}
.lbl{font-size:.75rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:.6rem;}

[data-testid="stFileUploader"]{background:var(--panel)!important;border:1.5px dashed var(--border)!important;border-radius:var(--r)!important;padding:1.2rem!important;transition:border-color .2s;}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}
[data-testid="stFileUploader"] label{display:none!important;}
[data-testid="stFileDropzone"]{background:transparent!important;border:none!important;}
[data-testid="stFileDropzone"] p,[data-testid="stFileDropzone"] small{color:var(--muted)!important;font-size:.9rem!important;}
[data-testid="stFileDropzone"] svg{fill:var(--accent)!important;}

.badge{display:flex;align-items:center;gap:10px;background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:10px;padding:.85rem 1rem;margin-top:.8rem;font-size:.9rem;word-break:break-all;}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:1rem;}
.pill{background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:4px 13px;font-size:.78rem;color:var(--muted);}
.pill b{color:var(--accent3);}

.stButton>button{width:100%;background:transparent!important;border:1px solid var(--border)!important;color:var(--muted)!important;border-radius:10px!important;font-family:'Outfit',sans-serif!important;font-size:.9rem!important;padding:.6rem!important;transition:all .2s!important;margin-top:1rem!important;}
.stButton>button:hover{border-color:var(--accent2)!important;color:var(--accent2)!important;background:rgba(255,101,132,.06)!important;}
.hint{margin-top:2rem;font-size:.78rem;color:#3d4250;line-height:1.9;}
.hint b{color:#4a5060;} .ha{color:var(--accent);font-weight:600;} .hg{color:var(--accent3);font-weight:600;}

.hero{padding:2.8rem 3.5rem 1.8rem;border-bottom:1px solid var(--border);background:linear-gradient(180deg,#0d0e18,var(--bg));}
.eyebrow{font-size:.72rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:.5rem;}
.hero h1{font-size:2.1rem!important;font-weight:800!important;margin:0 0 .4rem!important;padding:0!important;line-height:1.15!important;letter-spacing:-.5px;}
.hero h1 span{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{color:var(--muted);font-size:1.05rem;margin:0;}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem;gap:1.6rem;}
.orb{width:90px;height:90px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#2d2860,#0d0e18);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:2.2rem;animation:pulse 3s ease-in-out infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 30px rgba(108,99,255,.2);}50%{box-shadow:0 0 60px rgba(108,99,255,.42);}}
.emsg{text-align:center;color:var(--muted);font-size:1rem;max-width:340px;line-height:1.7;}
.emsg strong{display:block;color:var(--text);font-size:1.15rem;font-weight:700;margin-bottom:.3rem;}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;}
.chip{background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:7px 18px;font-size:.88rem;font-weight:500;color:var(--muted);}

[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:0!important;margin-bottom:1.4rem!important;}
[data-testid="stChatMessageContent"]{border-radius:var(--r)!important;padding:1rem 1.4rem!important;font-size:1rem!important;line-height:1.8!important;max-width:78%!important;}
[data-testid="stChatMessageContent"] p,[data-testid="stChatMessageContent"] li{font-size:1rem!important;line-height:1.8!important;}
[data-testid="chatAvatarIcon-user"]{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;border-radius:10px!important;}
[data-testid="chatAvatarIcon-assistant"]{background:linear-gradient(135deg,var(--accent3),var(--accent))!important;border-radius:10px!important;color:#07080d!important;}

[data-testid="stChatInput"] textarea{background:var(--panel)!important;border:1.5px solid var(--border)!important;border-radius:12px!important;color:var(--text)!important;font-family:'Outfit',sans-serif!important;font-size:1rem!important;padding:.95rem 1.3rem!important;caret-color:var(--accent)!important;}
[data-testid="stChatInput"] textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(108,99,255,.12)!important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted)!important;}
[data-testid="stChatInput"] button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;border:none!important;border-radius:8px!important;}

.src{display:inline-flex;align-items:center;gap:6px;margin-top:.9rem;background:rgba(67,232,173,.08);border:1px solid rgba(67,232,173,.22);border-radius:8px;padding:5px 12px;font-size:.78rem;font-weight:600;color:var(--accent3);}
.stAlert{background:rgba(255,101,132,.08)!important;border:1px solid rgba(255,101,132,.25)!important;border-radius:var(--r)!important;font-size:.92rem!important;}
hr{border-color:var(--border)!important;margin:1.2rem 0!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for k, v in [("messages",[]),("vectorstore",None),("doc_name",None),("chunk_count",0)]:
    if k not in st.session_state: st.session_state[k] = v

# ── PDF processing ──
def process_pdf(path):
    docs = PyPDFLoader(path).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100,
             separators=["\n\n","\n","."," ",""]).split_documents(docs)
    vs = Chroma.from_documents(splits, OllamaEmbeddings(model="nomic-embed-text"),
         collection_metadata={"hnsw:space":"cosine"})
    return vs, len(splits)

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🧠</div><h1>Doc<span>Mind</span></h1></div>', unsafe_allow_html=True)
    st.markdown('<span class="lbl">📎 Source Document</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f'<div class="badge"><span>📄</span><span>{uploaded_file.name}</span></div>', unsafe_allow_html=True)
    if st.session_state.doc_name:
        st.markdown(f'<div class="stats"><div class="pill">Chunks: <b>{st.session_state.chunk_count}</b></div><div class="pill">Model: <b>llama3.2</b></div><div class="pill">RAG: <b>ON</b></div></div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation"):
        for k, v in [("messages",[]),("vectorstore",None),("doc_name",None),("chunk_count",0)]:
            st.session_state[k] = v
        st.rerun()

    st.markdown('<div class="hint"><b>How it works</b><br>1. Upload any PDF<br>2. Wait for indexing<br>3. Ask questions naturally<br><br>Powered by <span class="ha">Ollama</span> + <span class="hg">ChromaDB</span></div>', unsafe_allow_html=True)

# ── Index PDF ──
if uploaded_file and st.session_state.doc_name != uploaded_file.name:
    with st.spinner("⚡ Indexing your document…"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                vs, chunks = process_pdf(tmp.name)
            st.session_state.update(vectorstore=vs, doc_name=uploaded_file.name, chunk_count=chunks, messages=[])
        except Exception as e:
            st.error(f"❌ Failed to index document: {e}")

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-Powered Document Intelligence</div>
  <h1>Ask anything about your <span>PDF</span></h1>
  <p>Upload a document → get instant, grounded answers with no hallucinations.</p>
</div>""", unsafe_allow_html=True)

# ── Empty states ──
ss = st.session_state
if not ss.messages and not ss.vectorstore:
    st.markdown("""<div class="empty"><div class="orb">🧠</div>
    <div class="emsg"><strong>No document loaded yet</strong>Upload a PDF from the sidebar and start asking questions.</div>
    <div class="chips"><div class="chip">Summarise the document</div><div class="chip">What are the key findings?</div><div class="chip">List the main topics</div><div class="chip">Explain section 1</div></div>
    </div>""", unsafe_allow_html=True)
elif not ss.messages:
    st.markdown('<div class="empty"><div class="orb">✅</div><div class="emsg"><strong>Document ready!</strong>Ask me anything about it below.</div></div>', unsafe_allow_html=True)

# ── Chat history ──
for msg in ss.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Input ──
prompt = st.chat_input("Ask anything about your document…" if ss.vectorstore else "Upload a PDF first…", disabled=not ss.vectorstore)

# ── Response ──
if prompt:
    ss.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            docs = ss.vectorstore.similarity_search(prompt, k=4)
            context = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            msg = f"❌ Retrieval error: {e}"
            st.markdown(msg); ss.messages.append({"role":"assistant","content":msg}); st.stop()

        if not context.strip():
            msg = "⚠️ No relevant content found for that question."
            st.markdown(msg); ss.messages.append({"role":"assistant","content":msg}); st.stop()

        sys_prompt = f'''You are DocMind, an expert document analyst. Answer using ONLY the context below.
If not found, say: "I couldn't find that in the document."
Use bullet points or short paragraphs when helpful.

Context:\n"""\n{context}\n"""\n\nQuestion: {prompt}\n\nAnswer:'''

        ph, full = st.empty(), ""
        try:
            for chunk in ChatOllama(model="llama3.2", temperature=0.1).stream([HumanMessage(content=sys_prompt)]):
                if chunk.content:
                    full += chunk.content
                    ph.markdown(full + "▌", unsafe_allow_html=True)
        except Exception as e:
            full = f"❌ LLM error: {e}\n\n*Run `ollama serve` and `ollama pull llama3.2`*"
            ph.markdown(full); ss.messages.append({"role":"assistant","content":full}); st.stop()

        if not full.strip(): full = "⚠️ No response generated. Try rephrasing."
        final = full + f'<div class="src">📎 Based on {len(docs)} chunks from <b>{ss.doc_name}</b></div>'
        ph.markdown(final, unsafe_allow_html=True)
        ss.messages.append({"role":"assistant","content":final})