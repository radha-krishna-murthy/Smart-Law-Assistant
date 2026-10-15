import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Smart Law Assistant", page_icon="⚖️", layout="wide")

load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except Exception:  # no secrets.toml present
    api_key = os.getenv("GOOGLE_API_KEY")

PROMPT = ChatPromptTemplate.from_template(
    """
    You are a smart Legal Assistant for a practicing lawyer. Given the legal document context below,
    extract and summarize important insights, clauses, references to laws, or overlooked critical information.
    Be accurate, detailed, and avoid guessing. If the answer is not in the context, say so.
    Point out any red flags, legal loopholes, or implications.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

MODELS = {
    "Gemini 2.5 Flash (fast)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (more accurate)": "gemini-2.5-pro",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1000px;}
    .hero {padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1.2rem;
           background: linear-gradient(135deg, #1e3a5f 0%, #3b2a6b 100%); color: #fff;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
    .hero p {margin: .3rem 0 0; opacity: .85;}
    .chip {display:inline-block; padding:.15rem .6rem; border-radius:999px;
           background:rgba(128,128,128,.18); font-size:.8rem; margin-right:.4rem;}
    </style>
    <div class="hero">
      <h1>⚖️ Smart Law Assistant</h1>
      <p>Upload a contract, NDA or agreement and ask questions. Answers stream in with the source passages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("GOOGLE_API_KEY is not set. Add it to a `.env` file or your deployment secrets.")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_path = temp_pdf.name
    try:
        docs = PyPDFLoader(temp_path).load()
    finally:
        os.remove(temp_path)

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    if not chunks:
        raise ValueError("No extractable text found (scanned PDFs need OCR).")
    return FAISS.from_documents(chunks, get_embeddings()), len(docs), len(chunks)


def stream_answer(question, retriever, llm, sources):
    """Yield answer tokens; collect retrieved chunks into `sources` as they arrive."""
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, PROMPT))
    for part in chain.stream({"input": question}):
        if "context" in part:
            sources.extend(part["context"])
        if "answer" in part:
            yield part["answer"]


def render_sources(sources):
    with st.expander(f"📑 Source passages ({len(sources)})"):
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get("page")
            label = f"**Passage {i}**" + (f" · page {int(page) + 1}" if page is not None else "")
            st.markdown(label)
            st.caption(doc.page_content)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")

    st.header("⚙️ Settings")
    model_label = st.selectbox("Model", list(MODELS))
    top_k = st.slider("Passages to retrieve", 2, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Not legal advice. Verify all output against the source document.")

# ---------- State ----------
st.session_state.setdefault("messages", [])

if uploaded_file:
    file_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("file_key") != file_key:
        try:
            with st.spinner("Reading and embedding document…"):
                vs, n_pages, n_chunks = process_pdf(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Could not process PDF: {e}")
            st.stop()
        st.session_state.update(
            vectors=vs, file_key=file_key, doc_info=(uploaded_file.name, n_pages, n_chunks), messages=[]
        )

if "vectors" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

name, n_pages, n_chunks = st.session_state.doc_info
st.markdown(
    f'<span class="chip">📄 {name}</span><span class="chip">{n_pages} pages</span>'
    f'<span class="chip">{n_chunks} chunks</span>',
    unsafe_allow_html=True,
)

# ---------- Chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("elapsed"):
            st.caption(f"⏱️ {msg['elapsed']:.1f}s")

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    st.markdown("- What are the termination conditions?\n- Summarize the key obligations of each party.\n- Are there any red-flag clauses?")

if question := st.chat_input("Ask a question about this document…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=MODELS[model_label],
        temperature=temperature,
    )
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})

    with st.chat_message("assistant"):
        sources = []
        start = time.perf_counter()
        try:
            answer = st.write_stream(stream_answer(question, retriever, llm, sources))
        except Exception as e:
            answer = f"⚠️ Request failed: {e}"
            st.error(answer)
        elapsed = time.perf_counter() - start
        if sources:
            render_sources(sources)
        st.caption(f"⏱️ {elapsed:.1f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "elapsed": elapsed}
    )
