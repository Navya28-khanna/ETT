import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# --- Page UI Configuration ---
st.set_page_config(page_title="Pro Doc QA", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 Notebook Pro")
    uploaded_file = st.file_uploader("Upload Source PDF", type="pdf")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RAG Logic ---
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)
    
    # Speed Tip: Persist the DB so it's not recreated constantly
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    return vectorstore

if uploaded_file:
    if "vectorstore" not in st.session_state:
        with st.spinner("Indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.vectorstore = process_pdf(tmp.name)
            st.success("Document ready!")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retrieval
            relevant_docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([d.page_content for d in relevant_docs])
            
            # Fast Response Tip: Use ChatOllama with streaming
            llm = ChatOllama(model="llama3.2", temperature=0) 
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer strictly using context:"
            
            # We use a placeholder to update text live
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm.stream(full_prompt):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
