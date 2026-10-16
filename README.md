# ⚖️ Smart Law Assistant

A Retrieval-Augmented Generation (RAG) chatbot that helps lawyers and legal professionals analyze legal documents (service agreements, contracts, NDAs) by answering questions based on an uploaded PDF.

## Features

- Upload and process legal PDF documents
- Ask detailed questions about the document
- Text embeddings via the Gemini API (`gemini-embedding-001`), so no local torch model and a small deploy footprint
- Streaming answers from Google Gemini (2.5 Flash or 2.5 Pro)
- Chat interface with history, model/temperature/retrieval settings and page-numbered source passages
- Shows the context chunks used for each answer
- Built with LangChain, Streamlit and FAISS

## Setup

1. Clone the repo
   ```bash
   git clone https://github.com/radha-krishna-murthy/smart-law-assistant.git
   cd smart-law-assistant
   ```
2. Create a `.env` file with your Gemini key (get one at https://aistudio.google.com/apikey)
   ```
   GOOGLE_API_KEY=your-gemini-api-key
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app
   ```bash
   streamlit run app.py
   ```

## Deploying on Streamlit Community Cloud

Push the repo to GitHub, create a new app pointing at `app.py`, and add `GOOGLE_API_KEY` under Settings → Secrets.

## Disclaimer

This tool is an aid for legal professionals and does not provide legal advice. Verify all output against the source document.
