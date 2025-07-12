# ⚖️ Law Assistant RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that helps lawyers and legal professionals analyze legal documents (e.g., service agreements, contracts, NDAs) by answering questions based on the uploaded PDF using powerful open-source and hosted LLM tools.

---

## 🚀 Features

- 📄 Upload and process legal PDF documents
- 🔍 Ask detailed legal questions from the document
- 🧠 Uses **HuggingFace** for local text embeddings (no API required)
- 🤖 Powered by **Groq LLaMA-3** model for fast, high-quality answers
- 🔎 Shows relevant context used to generate each response
- 🧾 Built with **LangChain**, **Streamlit**, and **FAISS**

---
## 📦 Setup Instructions

1. **Clone this repo**
   ```bash
   git clone https://github.com/radha-krishna-murthy/Smart-Law-Assistant.git
2. Create a .env file
GROQ_API_KEY=sk-your-groq-api-key

3. Install Dependencies
pip install -r requirements.txt

5. Run the App
streamlit run app.py
