import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Load GROQ API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Groq LLM (LLaMA3-8B)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192"
)

# Deep Legal Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a smart Legal Assistant for a practicing lawyer. Given the legal document context below,
    extract and summarize important insights, clauses, references to laws, or overlooked critical information.
    Be accurate, detailed, and avoid guessing. Point out any red flags, legal loopholes, or implications.
    
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Streamlit UI
st.title("⚖️ Law Assistant - Groq RAG Chatbot")
uploaded_file = st.file_uploader("📄 Upload a Legal PDF Document", type="pdf")
user_question = st.text_input("🔍 Ask a question related to this document")

# Create vector store using uploaded PDF
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_path = temp_pdf.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(chunks, embeddings)

# Embedding trigger
if uploaded_file:
    with st.spinner("🔄 Processing and embedding document..."):
        st.session_state.vectors = process_pdf(uploaded_file)
    st.success("✅ Document embedded and ready!")

# Ask question
if user_question and "vectors" in st.session_state:
    with st.spinner("🧠 Thinking..."):
        retriever = st.session_state.vectors.as_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_question})
        end = time.process_time()

        st.markdown("### ✅ Answer")
        st.write(response["answer"])
        st.markdown(f"⏱️ Response Time: `{end - start:.2f}` seconds")

        with st.expander("📑 Relevant Context Chunks"):
            for doc in response.get("context", []):
                st.write(doc.page_content)
                st.markdown("---")
