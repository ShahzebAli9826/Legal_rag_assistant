# Legal RAG Assistant

A Web-based Legal Document Question-Answering (QA) System built with **Python 3.11**, **Streamlit**, **LangChain**, **FAISS**, **BM25**, and **Google Gemini 2.5 Flash** / OpenAI.

## Features

- **Strict Python 3.11**: Fully tested and optimized for Python 3.11.
- **Google Gemini 2.5 Flash & OpenAI**: Powered by Gemini 2.5 Flash for fast, grounded legal reasoning.
- **Dual Document QA Mode**:
  - **Upload Mode**: Directly upload case PDFs, document images (`PNG`/`JPG`), or text files to ask questions specific to your document.
  - **Vector DB Mode**: Automatically searches prebuilt vector database (`FAISS` + `BM25`) if no file is uploaded.
- **Hybrid Retrieval (RRF)**: Combines **BM25 keyword search** and **FAISS vector similarity search** using Reciprocal Rank Fusion (RRF).
- **Zero Hallucination Guarantee**: Strict prompt rules force answers to be grounded only in retrieved/uploaded legal text.
- **Source Citations**: Displays exact document name, page number, and snippet previews for every answer.
- **Clean Streamlit UI**: Sleek chat interface with sidebar status metrics and index management.

## Project Architecture

```text
Legal_Rag/
├── app.py                      # Main Streamlit web application entry point
├── config/
│   └── settings.py             # System configuration, paths, and environment variables
├── data/
│   ├── sample_cases.json       # Indian legal case dataset
│   ├── vectorstore/            # Persisted FAISS and BM25 index files
│   └── metadata/               # Corpus metadata tracking
├── ingestion/
│   ├── pdf_loader.py           # PyMuPDF & dataset text loaders
│   ├── text_cleaner.py         # Legal text normalization
│   ├── chunker.py              # Recursive character splitter with deterministic IDs
│   ├── file_extractor.py       # PDF / Image / Text upload extractor
│   └── build_index.py          # Offline index creation script
├── retrieval/
│   ├── embeddings.py           # HuggingFace BAAI/bge-small-en-v1.5 embeddings
│   ├── vector_store.py         # FAISS vector store manager
│   ├── bm25.py                 # BM25 keyword index
│   ├── hybrid.py               # Reciprocal Rank Fusion (RRF) retriever
│   └── retriever.py            # Unified retriever engine
├── rag/
│   ├── chain.py                # Main RAG chain connecting Retrieval, Context & LLM
│   ├── context_builder.py      # Formats retrieved context for LLMs
│   ├── prompt.py               # Prompt templates loader
│   └── answer_validator.py     # Grounding validator
├── ui/
│   ├── chat.py                 # Main chat interface component
│   ├── sidebar.py              # Sidebar status and controls
│   └── sources.py              # Grounded source citation renderer
├── prompts/
│   └── legal_qa.txt            # Grounded legal QA prompt template
└── tests/                      # Pytest unit tests
```

---

## 🚀 Setup & Running Instructions (For New Machines / Fresh Clones)

### Step 1: Clone Repository
```powershell
git clone https://github.com/ShahzebAli9826/Legal_rag_assistant.git
cd Legal_rag_assistant
```

### Step 2: Create the Virtual Environment FIRST
*(Note: `.venv` is ignored in Git and must be created on new laptops before activating!)*

```powershell
py -3.11 -m venv .venv
```
*(If `py -3.11` is not found, use: `python -m venv .venv`)*

### Step 3: Activate Virtual Environment

**PowerShell**:
```powershell
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell shows a script execution error, run `Set-ExecutionPolicy Unrestricted -Scope Process` first, or activate via CMD below)*:

**Command Prompt (CMD)**:
```cmd
.\.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure `.env` File

Copy `.env.example` to create `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set your **Gemini API Key**:

```env
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### Step 6: Build Knowledge Base Index

```powershell
python -m ingestion.build_index
```

### Step 7: Launch the Application

```powershell
streamlit run app.py
```

Open your browser at: **[http://localhost:8501](http://localhost:8501)**
