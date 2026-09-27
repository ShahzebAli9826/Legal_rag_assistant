# Legal RAG Assistant — Complete Implementation Specification

> **Project:** Legal RAG Assistant  
> **Primary language:** Python **3.11 only**  
> **Application type:** Web-based legal document Question Answering (QA) system  
> **Core architecture:** Retrieval-Augmented Generation (RAG)  
> **Frontend:** Streamlit  
> **LLM:** OpenAI  
> **Retrieval:** Hybrid search (BM25 keyword retrieval + FAISS vector retrieval)  
> **PDF processing:** PyMuPDF  
> **RAG orchestration:** LangChain  
> **Optional agent orchestration:** AutoGen  
> **Environment management:** python-dotenv  
> **Tokenization:** tiktoken

---

## 1. Project Overview

The Legal RAG Assistant is a Python-based web application that allows a user to upload or work with legal PDF documents and ask natural-language questions about their contents.

The system must not depend only on the language model's general knowledge. Instead, it should:

1. Read legal PDF documents.
2. Extract their text.
3. Clean and normalize the extracted text.
4. Split the text into meaningful chunks.
5. Generate embeddings for the chunks.
6. Store the embeddings in a FAISS vector index and maintain a BM25 keyword index.
7. Convert the user's question into an embedding.
8. Retrieve candidates using both BM25 keyword search and FAISS semantic search, then merge and rank the results.
9. Pass the retrieved chunks and the question to an OpenAI LLM.
10. Generate an answer grounded in the retrieved document content.
11. Display the answer and source information in Streamlit.

The primary purpose is **document-grounded legal question answering**, not independent legal advice.

---

# 2. Strict Python Requirement

## Python Version: 3.11 ONLY

The entire project must be developed and executed using:

```text
Python 3.11.x
```

Do **not** use:

- Python 3.10
- Python 3.12
- Python 3.13
- Python 3.14

The virtual environment must be created with Python 3.11.

Check the version:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

Also verify:

```powershell
python -c "import sys; print(sys.version)"
```

If `python` points to another version, use the Python 3.11 executable explicitly.

---

# 3. Main Objectives

The implementation should satisfy these objectives:

- Process Indian legal PDF documents.
- Build a searchable legal knowledge base.
- Provide semantic retrieval rather than only keyword matching.
- Use RAG to ground generated answers in retrieved legal text.
- Provide a conversational Streamlit interface.
- Preserve document metadata such as:
  - document name
  - page number
  - section/chapter when available
  - chunk ID
- Show source references with answers.
- Reduce hallucination by instructing the LLM to answer only from retrieved context.
- Handle multiple legal documents.
- Allow future expansion of the knowledge base.
- Keep the architecture modular so retrieval, embedding, LLM, and UI components can be changed independently.

---

# 4. High-Level Architecture

```text
                    LEGAL PDF DOCUMENTS
                             |
                             v
                    +------------------+
                    |  PDF Ingestion   |
                    |    PyMuPDF       |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Text Cleaning &  |
                    | Normalization    |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Text Chunking    |
                    |    LangChain     |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Embedding Model  |
                    +------------------+
                             |
                             v
                    +------------------+
                    |   FAISS Index    |
                    | Vector Database  |
                    +------------------+
                             |
                             |
                    USER QUESTION
                             |
                             v
                    +------------------+
                    | Query Processing |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Query Embedding  |
                    +------------------+
                             |
                             v
                    +------------------+
                    | FAISS Retrieval  |
                    +------------------+
                             |
                             v
                    Relevant Legal Chunks
                             |
                             v
                    +------------------+
                    | Context Builder  |
                    +------------------+
                             |
                             v
                    +------------------+
                    |   OpenAI LLM     |
                    |    via LangChain |
                    +------------------+
                             |
                             v
                    Grounded Answer
                             |
                             v
                    +------------------+
                    |    Streamlit     |
                    |       UI         |
                    +------------------+
```

---

# 5. Technology Stack

## 5.1 Programming Language

### Python 3.11

Python is the only programming language required for the backend, RAG pipeline, document processing, and application logic.

---

## 5.2 Frontend

### Streamlit

Streamlit will provide:

- PDF upload
- Document selection
- Chat interface
- Question input
- Answer display
- Source display
- Retrieval/debug information where required
- Application status messages

Streamlit provides `st.file_uploader`, `st.chat_message`, and `st.chat_input`, which are suitable for this application.

Official documentation:

https://docs.streamlit.io/

---

## 5.3 PDF Processing

### PyMuPDF

PyMuPDF will be responsible for:

- Opening PDFs
- Reading pages
- Extracting text
- Preserving page-level metadata
- Detecting pages with little/no extractable text
- Supporting future OCR handling

Official documentation:

https://pymupdf.readthedocs.io/

Important implementation rule:

Each extracted page should retain its page number.

Example metadata:

```python
{
    "source": "Bharatiya_Nyaya_Sanhita_2023.pdf",
    "page": 42
}
```

---

# 6. Text Processing and Chunking

Extracted PDF text must not be embedded as one huge document.

The pipeline should divide the document into smaller chunks.

Recommended initial configuration:

```text
chunk_size: 800
chunk_overlap: 150
```

These values are starting values, not permanent constants. Retrieval quality should be evaluated before changing them.

Recommended splitting order:

```text
Paragraph
    ↓
Section / line boundary
    ↓
Sentence boundary
    ↓
Word boundary
```

The chunking component should preserve metadata.

Example:

```python
{
    "text": "...legal text...",
    "metadata": {
        "source": "Indian_Contract_Act_1872.pdf",
        "page": 18,
        "chunk_id": "contract_18_03"
    }
}
```

---

# 7. Embeddings

Each document chunk must be converted into a numerical vector.

The implementation should use a dedicated embedding model.

Recommended initial model:

```text
BAAI/bge-small-en-v1.5
```

Reasons:

- Lightweight
- Good semantic retrieval performance for a local embedding model
- Easy to run from Python
- Avoids sending document text to a second external embedding service
- Suitable for an MCA-level RAG implementation

Embedding configuration:

```text
normalize_embeddings = True
```

The same embedding model must be used for:

- Document chunks
- User queries

Do not build the database with one embedding model and query it with another.

---

# 8. Vector Database

## FAISS

FAISS will be the initial vector database/retrieval engine.

It is responsible for:

- Storing document vectors
- Performing similarity search
- Returning nearest document chunks
- Supporting efficient retrieval

Official documentation:

https://faiss.ai/

The vector index should be persisted to disk rather than rebuilt every time the Streamlit application starts.

Recommended structure:

```text
data/
└── vectorstore/
    ├── index.faiss
    └── index.pkl
```

The exact persistence format may vary depending on the LangChain FAISS integration.

---

# 9. Metadata Strategy

Metadata is extremely important for a legal RAG system.

Every chunk should retain at least:

```text
document_id
document_name
source_path
page_number
chunk_id
```

Optional metadata:

```text
act_name
year
section_number
chapter
article
document_type
```

Example:

```python
{
    "document_id": "bns_2023",
    "document_name": "Bharatiya Nyaya Sanhita, 2023",
    "source_path": "data/bns_2023.pdf",
    "page_number": 54,
    "chunk_id": "bns_2023_p54_c02",
    "document_type": "Act",
    "year": 2023
}
```

This metadata will later be used to display sources with answers.

---

# 10. Knowledge Base

The initial corpus should consist of official Indian legal documents published by authoritative government/court sources.

The initial dataset can include post-1950 legislation covering different domains, for example:

- Constitution of India, 1950
- Special Marriage Act, 1954
- Hindu Marriage Act, 1955
- Hindu Succession Act, 1956
- Arbitration and Conciliation Act, 1996
- Information Technology Act, 2000
- Right to Information Act, 2005
- Protection of Women from Domestic Violence Act, 2005
- Companies Act, 2013
- Juvenile Justice Act, 2015
- Real Estate (Regulation and Development) Act, 2016
- Insolvency and Bankruptcy Code, 2016
- Consumer Protection Act, 2019
- Digital Personal Data Protection Act, 2023
- Bharatiya Nyaya Sanhita, 2023

Documents should preferably be downloaded from official sources such as:

- India Code
- Supreme Court of India
- Ministry websites
- Department of Telecommunications
- Other official Government of India portals

The actual corpus should be recorded in a metadata file so that every PDF can be traced to its official source.

---

# 11. Document Ingestion Pipeline

The ingestion pipeline should be separate from the Streamlit application.

Recommended command:

```powershell
python -m ingestion.build_index
```

The ingestion process:

```text
PDF
 ↓
Validate PDF
 ↓
Extract pages using PyMuPDF
 ↓
Clean text
 ↓
Attach page metadata
 ↓
Chunk text
 ↓
Generate embeddings
 ↓
Build FAISS index
 ↓
Persist FAISS index
 ↓
Save document/chunk metadata
```

---

# 12. Document Validation

Before processing a PDF:

1. Verify that the file exists.
2. Verify that it is a PDF.
3. Open it with PyMuPDF.
4. Check page count.
5. Attempt text extraction.
6. Detect pages with empty/very small extracted text.
7. Log extraction problems.

Example validation output:

```text
Document: BNS_2023.pdf
Pages: 358
Extracted pages: 358
Empty pages: 0
Status: SUCCESS
```

If a scanned PDF contains images instead of text, OCR may be required.

OCR should be treated as a separate module rather than silently mixing OCR logic into normal extraction.

---

# 13. Text Cleaning

Legal PDFs can contain:

- repeated headers
- repeated footers
- page numbers
- excessive whitespace
- broken line breaks
- hyphenated words
- encoding artifacts

The cleaner should perform conservative normalization.

It should NOT:

- remove legal section numbers
- remove article numbers
- remove subsection numbers
- rewrite legal language
- summarize the document
- alter the original meaning

The source text should remain as close as possible to the official document.

---

# 14. Chunking Strategy

Initial implementation:

```text
Chunk size = 800 characters/tokens according to the selected splitter configuration
Overlap = 150
```

The implementation should use LangChain's text splitting utilities.

Every chunk should contain metadata inherited from its source page.

Example:

```text
Document
  Page 25
     Chunk 1
     Chunk 2
     Chunk 3

  Page 26
     Chunk 4
     Chunk 5
```

The chunk ID must be deterministic.

---

# 15. Retrieval Pipeline

When the user asks:

> What happens if a borrower fails to repay the loan?

The system should execute:

```text
User Question
      ↓
Query validation
      ↓
Query embedding
      ↓
FAISS similarity search
      ↓
Top-K relevant chunks
      ↓
Optional filtering
      ↓
Context construction
      ↓
LLM
```

Initial retrieval configuration:

```text
top_k = 5
```

During experimentation, test:

```text
top_k = 3
top_k = 5
top_k = 8
top_k = 10
```

Do not assume that a larger `top_k` is always better.

---

# 16. Optional Retrieval Improvement

Hybrid retrieval is part of the main implementation, not a later optional feature. Run BM25 keyword retrieval and FAISS dense-vector retrieval in parallel for every user query.

Architecture:

```text
                 User Query
                     |
          +----------+----------+
          |                     |
          v                     v
      BM25 Search          Vector Search
          |                     |
          +----------+----------+
                     |
                     v
              Merge Results
                     |
                     v
          Reciprocal Rank Fusion (RRF)
                     |
                     v
               Final Context
```

Use Reciprocal Rank Fusion (RRF) to combine the ranked lists without comparing raw BM25 scores with FAISS distances. A cross-encoder reranker may be added later, but is optional.

---

### Hybrid retrieval behavior

For each question:

1. Tokenize the query for BM25 and search the corpus.
2. Embed the same query and search the FAISS index.
3. Retrieve a candidate list from each retriever (initially 10–20 candidates each).
4. Merge duplicate chunks using a stable `chunk_id`.
5. Apply Reciprocal Rank Fusion (RRF), for example `score = sum(1 / (k + rank))` across retrievers, with `k` initially 60.
6. Return the highest-ranked unique chunks (initially 5) with their page and document metadata.
7. Pass only those merged results to the RAG context builder.

Build the BM25 corpus from the same chunk records used to build FAISS. Persist the chunk metadata/text needed to reconstruct BM25, or rebuild BM25 from processed chunks at startup. Keep the FAISS index persistent so embeddings are not regenerated for every app launch.

# 17. Context Builder

The context builder converts retrieved chunks into an LLM-readable context.

Example:

```text
SOURCE 1
Document: Consumer Protection Act, 2019
Page: 24

[retrieved legal text]


SOURCE 2
Document: Consumer Protection Act, 2019
Page: 25

[retrieved legal text]
```

The context must include source metadata.

This makes it possible to show the user where the answer came from.

---

# 18. LLM Integration

The LLM will be accessed through LangChain's OpenAI integration.

The LLM prompt must strongly prioritize retrieved evidence.

Recommended system behavior:

```text
You are a legal document question-answering assistant.

Answer the user's question using the supplied document context.

Rules:
1. Use the retrieved context as the primary source.
2. Do not invent legal provisions.
3. Do not fabricate sections, cases, dates, or citations.
4. If the retrieved context does not contain enough information, say so.
5. Clearly distinguish information found in the document from general explanation.
6. Provide document/page references when available.
7. Do not present the response as a substitute for professional legal advice.
```

The exact prompt should be stored in:

```text
prompts/legal_qa.txt
```

rather than hard-coded throughout the application.

---

# 19. Hallucination Control

The application should explicitly handle insufficient evidence.

Bad behavior:

```text
Question
   ↓
No relevant evidence
   ↓
LLM guesses answer
```

Desired behavior:

```text
Question
   ↓
Retrieval
   ↓
Insufficient relevant context
   ↓
Assistant says:
"I could not find sufficient information in the
uploaded documents to answer this question."
```

The system should not fabricate a legal answer.

---

# 20. Source Citations

Each generated answer should provide source information where possible.

Example UI:

```text
Answer:
The agreement provides that...

Sources:
1. Loan_Agreement.pdf — Page 12
2. Loan_Agreement.pdf — Page 13
```

For statutory documents:

```text
Source:
Bharatiya Nyaya Sanhita, 2023
Page: 84
Section: [if metadata extraction identifies it]
```

The source information should come from retrieval metadata, not from the LLM inventing references.

---

# 21. Streamlit User Interface

The application should contain the following areas.

## Sidebar

```text
LEGAL RAG ASSISTANT

Documents
[Upload PDF]

Knowledge Base
[Build/Refresh Index]

Retrieval
Top K: 5

Settings
[Model]
[Temperature]
```

## Main area

```text
LEGAL RAG ASSISTANT

Ask questions about your legal documents.

User:
What are the conditions for termination?

Assistant:
[Generated answer]

Sources:
[Document + page]
```

Streamlit's chat interface should use:

```python
st.chat_message()
st.chat_input()
```

and uploaded PDFs can be handled with:

```python
st.file_uploader()
```

---

# 22. Document Upload Behavior

There should be two supported workflows.

## Workflow A — Prebuilt Knowledge Base

```text
Official PDFs
     ↓
Offline ingestion
     ↓
FAISS index
     ↓
Streamlit application
     ↓
Question answering
```

This is the recommended production/demo workflow.

## Workflow B — User Upload

```text
User uploads PDF
       ↓
Temporary storage
       ↓
Extract
       ↓
Chunk
       ↓
Embed
       ↓
Add to FAISS
       ↓
Ask questions
```

For the first version, Workflow A should be implemented first.

---

# 23. Conversation Memory

The UI should maintain the current chat history using Streamlit session state.

Example:

```python
st.session_state.messages
```

However, conversation memory should not replace document retrieval.

For every new question:

```text
Current Question
      +
Relevant Document Context
      +
Optional Conversation Context
      ↓
LLM
```

The legal document remains the authoritative knowledge source for document-specific questions.

---

# 24. AutoGen

The synopsis lists AutoGen for AI-agent workflows.

For the initial implementation, AutoGen should NOT be forced into every request.

If an agent workflow is required, it can be added as:

```text
User Query
    ↓
Query/Planning Agent
    ↓
Retrieval Agent
    ↓
Answer Generation Agent
    ↓
Validation/Reflection Agent
    ↓
Final Answer
```

Possible responsibilities:

### Query Agent

Determines:

- what the user is asking
- whether document retrieval is required
- which document type may be relevant

### Retrieval Agent

Calls the retriever and collects relevant chunks.

### Answer Agent

Generates an answer from retrieved context.

### Validation Agent

Checks:

- Is the answer supported?
- Are sources present?
- Did the answer introduce unsupported facts?

For an MCA project, this agent layer can be presented as an advanced extension if the basic RAG pipeline already satisfies the core requirements.

---

# 25. Recommended Project Architecture

```text
Legal_RAG_Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── IMPLEMENTATION.md
├── .env
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   │   ├── processed/
│   ├── vectorstore/
│   └── metadata/
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── metadata.py
│   └── build_index.py
│
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── bm25.py
│   └── hybrid.py
│
├── rag/
│   ├── __init__.py
│   ├── prompt.py
│   ├── context_builder.py
│   ├── chain.py
│   └── answer_validator.py
│
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── retrieval_agent.py
│   ├── answer_agent.py
│   └── validation_agent.py
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py
│   ├── chat.py
│   └── sources.py
│
├── prompts/
│   ├── legal_qa.txt
│   └── validation.txt
│
├── tests/
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   └── test_metadata.py
│
└── logs/
```

---

# 26. Responsibilities of Important Files

## `app.py`

Main Streamlit entry point.

Responsibilities:

- Initialize application
- Load configuration
- Initialize retriever
- Display UI
- Receive user query
- Call RAG pipeline
- Display answer
- Display sources

---

## `ingestion/pdf_loader.py`

Responsibilities:

- Open PDFs
- Extract page text
- Return page-level documents
- Attach source/page metadata

---

## `ingestion/text_cleaner.py`

Responsibilities:

- Normalize whitespace
- Remove obvious extraction artifacts
- Preserve legal meaning

---

## `ingestion/chunker.py`

Responsibilities:

- Split text
- Preserve metadata
- Generate deterministic chunk IDs

---

## `ingestion/build_index.py`

Responsibilities:

```text
Load PDFs
→ Extract
→ Clean
→ Chunk
→ Embed
→ Build FAISS
→ Save index
```

Run:

```powershell
python -m ingestion.build_index
```

---

## `retrieval/embeddings.py`

Responsibilities:

- Load embedding model
- Generate document embeddings
- Generate query embeddings

---

## `retrieval/vector_store.py`

Responsibilities:

- Create FAISS index
- Save index
- Load index
- Add documents
- Search documents

---

## `retrieval/retriever.py`

Responsibilities:

- Accept query
- Perform retrieval
- Return top-K chunks

---

## `rag/context_builder.py`

Responsibilities:

- Receive retrieved chunks
- Format context
- Include metadata
- Prevent excessively large prompts

---

## `rag/chain.py`

Responsibilities:

```text
Question
   ↓
Retriever
   ↓
Context
   ↓
Prompt
   ↓
OpenAI
   ↓
Answer
```

---

## `rag/answer_validator.py`

Responsibilities:

- Check whether sources exist
- Detect empty answers
- Detect obvious unsupported claims
- Enforce "insufficient evidence" behavior

---

# 27. Configuration Management

Use `.env` for secrets.

Example:

```env
OPENAI_API_KEY=your_api_key_here
```

Do NOT write:

```python
OPENAI_API_KEY = "sk-..."
```

inside source code.

Use `python-dotenv` to load environment variables.

---

# 28. `.gitignore`

The following should not be committed:

```text
.env
.venv/
__pycache__/
*.pyc

data/vectorstore/
data/processed/

logs/

.vscode/
.idea/
```

If legal PDFs have redistribution restrictions, do not commit them to GitHub either.

---

# 29. requirements.txt

The final `requirements.txt` should contain the actual packages used by the implementation.

Core dependencies:

```text
streamlit
langchain
langchain-community
langchain-openai
faiss-cpu
pymupdf
openai
tiktoken
python-dotenv
sentence-transformers
transformers
torch
```

If AutoGen is actually implemented:

```text
autogen-agentchat
```

If BM25 is implemented:

```text
rank-bm25
```

The project must be tested with **Python 3.11 only**.

Do not install packages into the global Python environment.

---

# 30. Virtual Environment Setup

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 31. Environment Verification

Before running the application:

```powershell
python --version
pip --version
```

Then:

```powershell
python -c "import streamlit, pymupdf, faiss, langchain, openai; print('Environment OK')"
```

The project should not continue if Python is not 3.11.

---

# 32. Running the Project

## Step 1 — Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 2 — Verify Python

```powershell
python --version
```

## Step 3 — Add legal PDFs

Put any legal PDFs you have directly inside the existing data folder (no separate legal_documents folder):

```text
data/
```

## Step 4 — Build the vector database

```powershell
python -m ingestion.build_index
```

## Step 5 — Run Streamlit

```powershell
streamlit run app.py
```

---

# 33. First-Time Index Build

The first build should produce a summary similar to:

```text
==================================================
LEGAL RAG INDEX BUILDER
==================================================

Documents found: 15

Processing:
[1/15] Constitution_1950.pdf
[2/15] Special_Marriage_Act_1954.pdf
...
[15/15] BNS_2023.pdf

Pages processed: XXXX
Chunks created: XXXX

Generating embeddings...
Building FAISS index...

Index saved successfully.

==================================================
INDEX BUILD COMPLETE
==================================================
```

The exact number of chunks will depend on the documents and chunking configuration.

---

# 34. Query Execution

Example:

```text
User:
What is the purpose of the Digital Personal Data Protection Act?

System:
1. Convert question to embedding.
2. Search FAISS.
3. Retrieve top 5 chunks.
4. Build context.
5. Send context + question to OpenAI.
6. Generate grounded answer.
7. Display sources.
```

---

# 35. Legal Prompt Design

The prompt should explicitly state:

```text
You are a Legal Document Assistant.

You answer questions using the supplied legal document context.

Rules:
- Do not fabricate legal provisions.
- Do not invent section numbers.
- Do not invent court decisions.
- Do not assume that information absent from the retrieved context is present.
- If the context is insufficient, say that the available documents do not provide enough information.
- Identify the source document and page when available.
- Explain legal text in understandable language.
- Do not claim to be a lawyer.
- Do not present generated output as formal legal advice.
```

---

# 36. Retrieval Evaluation

The project should not be considered complete merely because the chatbot produces answers.

Retrieval must be evaluated.

Create a test set:

```text
questions/
└── evaluation_questions.json
```

Example:

```json
[
  {
    "question": "What is the purpose of the Consumer Protection Act, 2019?",
    "expected_document": "Consumer_Protection_Act_2019.pdf"
  },
  {
    "question": "What is the definition of personal data?",
    "expected_document": "DPDP_Act_2023.pdf"
  }
]
```

Measure:

### Retrieval Hit Rate

Did the relevant document/chunk appear in the retrieved results?

### Source Accuracy

Does the answer cite a source actually used by retrieval?

### Answer Grounding

Is the generated answer supported by retrieved text?

### No-Evidence Behavior

Does the system correctly refuse to answer when the documents do not contain enough information?

---

# 37. Testing Strategy

Use `pytest`.

Tests should cover:

## PDF Loader

- Valid PDF
- Empty PDF
- Corrupted PDF
- Multiple pages

## Chunker

- Chunk creation
- Chunk overlap
- Metadata preservation
- Deterministic chunk IDs

## Retriever

- Relevant question
- Irrelevant question
- Top-K behavior
- Metadata return

## RAG

- Context is passed to LLM
- Empty context behavior
- Source metadata returned

## UI

Manual testing through Streamlit.

---

# 38. Error Handling

The application should gracefully handle:

### Missing API key

```text
OPENAI_API_KEY is not configured.
Please configure the .env file.
```

### Missing vector index

```text
Knowledge base not found.
Run:

python -m ingestion.build_index
```

### Invalid PDF

```text
The uploaded file could not be processed as a valid PDF.
```

### No relevant results

```text
No sufficiently relevant information was found
in the available legal documents.
```

### API failure

Display a user-friendly message and log the technical error.

Do not expose API keys or sensitive configuration in the UI.

---

# 39. Logging

Use Python's standard `logging` module.

Log:

- ingestion start/end
- document name
- page count
- chunk count
- index creation
- query execution
- retrieval count
- errors
- API failures

Do not log:

- API keys
- complete confidential documents
- unnecessary user information

---

# 40. Security Requirements

The application should:

- Store API keys in `.env`.
- Never commit `.env`.
- Validate uploaded files.
- Restrict uploads to PDF.
- Avoid trusting the uploaded filename as a filesystem path.
- Store uploads in application-controlled directories.
- Avoid exposing raw environment variables.
- Avoid executing uploaded document contents as code.

Uploaded filenames should not directly determine filesystem paths.

---

# 41. Legal Safety

The application should clearly state:

```text
This system is an AI-based document research assistant.
It provides information based on the documents available
to it and is not a substitute for professional legal advice.
Users should independently verify important legal information
with authoritative sources or qualified legal professionals.
```

The system should distinguish:

```text
Document content
        ≠
AI-generated explanation
        ≠
Professional legal advice
```

---

# 42. Performance Considerations

Do not rebuild the vector database every time Streamlit starts.

Correct:

```text
Build index once
       ↓
Save index
       ↓
Load index in app
       ↓
Answer queries
```

Incorrect:

```text
Streamlit starts
       ↓
Read PDFs found directly in data/
       ↓
Generate all embeddings
       ↓
Build FAISS
       ↓
Finally show UI
```

The second approach unnecessarily increases startup time and API/model usage.

---

# 43. Caching

Streamlit caching can be used for expensive resources such as:

- Embedding model
- FAISS index
- LLM client configuration

For example:

```python
@st.cache_resource
def load_vectorstore():
    ...
```

This prevents repeated initialization during Streamlit reruns.

---

# 44. Recommended Development Order

Do not implement everything simultaneously.

## Phase 1 — Environment

```text
Python 3.11
Virtual environment
requirements.txt
.env
```

## Phase 2 — PDF Processing

```text
PDF
 ↓
PyMuPDF
 ↓
Page text
```

## Phase 3 — Chunking

```text
Page text
 ↓
LangChain splitter
 ↓
Chunks + metadata
```

## Phase 4 — Embeddings

```text
Chunks
 ↓
Embedding model
 ↓
Vectors
```

## Phase 5 — FAISS

```text
Vectors
 ↓
FAISS
 ↓
Persisted index
```

## Phase 6 — Retrieval

```text
Question
 ↓
Embedding
 ↓
FAISS
 ↓
Top-K chunks
```

## Phase 7 — LLM

```text
Question + Retrieved Context
 ↓
OpenAI
 ↓
Answer
```

## Phase 8 — Streamlit

```text
Upload/select document
 ↓
Chat UI
 ↓
Question
 ↓
Answer + Sources
```

## Phase 9 — Evaluation

```text
Test questions
 ↓
Retrieval evaluation
 ↓
Answer evaluation
 ↓
Improve chunking/retrieval/prompt
```

## Phase 10 — Advanced Features

After the core hybrid RAG system works, consider:

```text
Cross-encoder reranking
AutoGen Agents
Advanced answer validation
Improved citations
```

---

# 45. Final System Flow

The final system should work as follows:

```text
                 OFFLINE INDEXING
                       |
                       v
             Official Legal PDFs
                       |
                       v
                  PyMuPDF
                       |
                       v
                Text Cleaning
                       |
                       v
                 Text Chunking
                       |
                       v
                 Embeddings
                       |
                       v
                    FAISS
                       |
                       v
               Persisted Index
                       |
                       |
              ONLINE QUERY FLOW
                       |
                       v
                  User Query
                       |
                       v
               Query Embedding
                       |
                       v
                FAISS Search
                       |
                       v
              Relevant Chunks
                       |
                       v
               Context Builder
                       |
                       v
               Prompt Template
                       |
                       v
                 OpenAI LLM
                       |
                       v
              Answer Validation
                       |
                       v
               Final Response
                       |
                       v
              Streamlit Interface
                       |
                       v
             Answer + Sources
```

---

# 46. Minimum Viable Product

The first working version is complete when all of these work:

- [ ] Python 3.11 environment
- [ ] Legal PDF files placed directly in data/
- [ ] PDF extraction
- [ ] Text cleaning
- [ ] Chunking
- [ ] Embedding generation
- [ ] FAISS index
- [ ] Persistent vector store
- [ ] Query embedding
- [ ] Top-K retrieval
- [ ] OpenAI LLM
- [ ] RAG prompt
- [ ] Streamlit chat UI
- [ ] Source/page display
- [ ] Error handling
- [ ] `.env` API key management
- [ ] Basic tests
- [ ] Evaluation questions

---

# 47. Advanced Version

After the MVP:

```text
MVP
 |
 +--> Hybrid BM25 + Vector Retrieval
 |
 +--> Cross-Encoder Reranking
 |
 +--> Query Rewriting
 |
 +--> AutoGen Agent Workflow
 |
 +--> Answer Validation
 |
 +--> Better Section Detection
 |
 +--> OCR for scanned PDFs
 |
 +--> Document Comparison
 |
 +--> Legal Document Summarization
 |
 +--> Multi-document Question Answering
```

These should be implemented only after the basic RAG system is stable.

---

# 48. Recommended Final Architecture

For the MCA project, the recommended final architecture is:

```text
                         STREAMLIT
                            |
                    +-------+-------+
                    |               |
               Documents         Questions
                    |               |
                    v               v
                INGESTION        RETRIEVAL
                    |               |
                PyMuPDF          FAISS
                    |               |
                 Chunking         Top-K
                    |               |
               Embeddings            |
                    |               |
                    +-------+-------+
                            |
                         CONTEXT
                            |
                            v
                      LANGCHAIN RAG
                            |
                            v
                       OPENAI LLM
                            |
                            v
                    ANSWER VALIDATOR
                            |
                            v
                    ANSWER + SOURCES
```

This is the baseline architecture that should be implemented before adding agent-based orchestration.

---

# 49. Definition of Done

The project can be considered implementation-complete when:

1. The application runs using **Python 3.11 only**.
2. Official Indian legal PDFs are successfully indexed.
3. FAISS persists the vector database.
4. The application can load the existing index without rebuilding it.
5. A user can ask natural-language questions.
6. Relevant legal chunks are retrieved.
7. OpenAI generates an answer using retrieved context.
8. The answer displays document/page source information.
9. The system refuses to fabricate an answer when evidence is insufficient.
10. API keys are stored securely.
11. Basic unit tests pass.
12. Retrieval and answer quality are evaluated on a fixed question set.
13. The Streamlit interface is usable for demonstration.
14. The complete project can be reproduced on another machine using Python 3.11 and the documented setup instructions.

---

# 50. Official Documentation References

- Python 3.11: https://docs.python.org/3.11/
- Streamlit: https://docs.streamlit.io/
- PyMuPDF: https://pymupdf.readthedocs.io/
- FAISS: https://faiss.ai/
- LangChain: https://python.langchain.com/
- OpenAI Platform: https://platform.openai.com/docs/
- Tiktoken: https://github.com/openai/tiktoken
- AutoGen: https://microsoft.github.io/autogen/

---

# 51. Important Implementation Rule

**Python 3.11 is the project's only supported Python version.**

Every setup instruction, virtual environment, dependency installation, test, and execution command in this project must use Python 3.11.

Before running anything, always verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

If it is not Python 3.11, do not run the project until the correct interpreter/environment is selected.
