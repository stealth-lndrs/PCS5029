# 🧩 **PHASE 0 — MASTER OVERVIEW PROMPT (Run first in Codex)**

```markdown
# PROJECT OVERVIEW — DO NOT GENERATE CODE YET

You are Codex. Your task is to generate a complete Python project, but we start with THIS overview prompt.  
**In this prompt do not generate any Python code. Only output the PROJECT STRUCTURE and MODULE DESCRIPTIONS.**

---

# 🧠 PROJECT DESCRIPTION

We are building a **local, multilingual Question Answering System** using:

- Python
- NiceGUI (UI)
- Local LLM: **Gemma 2 / 3 Instruct 9B**, loaded via vLLM or transformers
- RAG pipeline with embeddings + vector DB
- Document upload functionality (PDFs)
- Full OCR support for scanned PDFs (using PaddleOCR)
- Chunking + metadata extraction
- Embedding model: **BGE-M3** (multilingual)
- Vector DB: **Chroma or FAISS**
- Optional fine-tuning via **QLoRA**
- Single-user for now, architecture prepared for multi-user later
- PDFs may be large (10–1000 pages), multilingual, with or without text layer

The system must:

1. Allow user to upload PDFs.
2. Extract text (OCR when needed) page-by-page.
3. Chunk text with overlap.
4. Compute embeddings and store chunks in vector DB.
5. Build RAG context at query time.
6. Answer questions in **Portuguese**, but documents may be in any language.
7. Show citations (doc, page).
7. Provide a NiceGUI chat interface.
8. Provide document management UI.
9. Support fine-tuning dataset generation + QLoRA.

---

# 🚀 GPU CONSTRAINTS

Inference and training will happen on a GPU with **20 GB VRAM (AWS g6.xlarge)**.

Gemma 9B must run in **4-bit/8-bit mode** for inference.  
QLoRA fine-tuning must be supported.

---

# 📂 REQUIRED PROJECT STRUCTURE

Codex must create the following folders and files (no code yet):

```
project/
  app.py
  backend/
    __init__.py
    config.py
    ocr.py
    pdf_ingestion.py
    chunker.py
    embeddings.py
    vectordb.py
    rag.py
    llm.py
  ui/
    main_ui.py
    chat_ui.py
    documents_ui.py
  finetune/
    generate_dataset.py
    train_lora.py
  data/
    pdfs/
    chunks/
    vectordb/
    adapters/
```

Each module has the following purpose:

- **config.py** — Global settings, model paths, embedding config, chunk sizes.
- **ocr.py** — PaddleOCR wrapper for detecting scanned PDFs and extracting text per page.
- **pdf_ingestion.py** — Streams PDFs page-by-page, applies OCR, stores raw text, sends chunks to DB.
- **chunker.py** — Token-based chunking with metadata.
- **embeddings.py** — Embedding model loader (BGE-M3) + encoding functions.
- **vectordb.py** — Wrapper around Chroma or FAISS: insert, search, delete, filter.
- **rag.py** — Build retrieval pipeline + prompt construction for context.
- **llm.py** — Local Gemma runtime (transformers/vLLM), unified generate() function.
- **main_ui.py** — NiceGUI layout + state + navigation.
- **chat_ui.py** — Chat interface + websocket streaming of LLM responses.
- **documents_ui.py** — Upload UI, delete UI, ingestion progress.
- **generate_dataset.py** — Create synthetic Q&A using OpenAI APIs.
- **train_lora.py** — QLoRA fine-tuning script for Gemma 9B.
- **app.py** — Entry point, builds NiceGUI app with routes.

---

# 🎯 WHAT TO OUTPUT NOW

Output ONLY:

1. The directory structure.
2. Short description of each file.
3. No Python code.

Wait for further prompts.

```

---

# 🧩 **PHASE 1 — MULTI-PROMPT SEQUENCE (A → N)**  
Each block below is a **separate prompt** you will send to Codex after the previous one completes.

---

# 🔷 **PROMPT A — Create Project Structure + Empty Files**

```markdown
# PROMPT A — CREATE PROJECT STRUCTURE

Generate the full project directory structure with empty files exactly as defined in the Overview Prompt (PHASE 0).  
For each file, insert only:

```
# <filename>
# (empty placeholder)
```

Do NOT generate any implementation code yet.

Follow EXACTLY this tree:

project/
  app.py
  backend/
    __init__.py
    config.py
    ocr.py
    pdf_ingestion.py
    chunker.py
    embeddings.py
    vectordb.py
    rag.py
    llm.py
  ui/
    main_ui.py
    chat_ui.py
    documents_ui.py
  finetune/
    generate_dataset.py
    train_lora.py
  data/
    pdfs/
    chunks/
    vectordb/
    adapters/

Return the full file contents for all placeholders.
```

---

# 🔷 **PROMPT B — Implement OCR Module (PaddleOCR)**

```markdown
# PROMPT B — IMPLEMENT backend/ocr.py

Write the complete implementation for backend/ocr.py.

Requirements:
- Use PaddleOCR (multilingual).
- Detect whether a page has text; if not → run OCR.
- Function: `extract_page_text(pdf_path, page_number)`  
- Function: `is_page_scanned(text)` → returns True if text too short.
- Handle large PDFs efficiently.
- Include robust error handling and logging.
- Return UTF-8 cleaned text.

Only output backend/ocr.py.
```

---

# 🔷 **PROMPT C — Implement PDF Ingestion Pipeline**

```markdown
# PROMPT C — IMPLEMENT backend/pdf_ingestion.py

Implement:
- Streaming read of PDF pages using PyMuPDF (fitz).
- For each page:
  - Try direct text extraction.
  - If text likely from scanned page → call OCR module.
  - Normalize whitespace.
  - Save clean extracted text to data/chunks or in memory.
  - Pass each page’s text to chunker module.
- Store metadata (doc_id, filename, page number).
- Push chunks to embedding + vectorDB pipeline.

Main function:
`ingest_pdf(pdf_path: str, doc_id: str) -> List[ChunkMetadata]`

No blocking load of whole PDF.
Handle documents up to 1000 pages.

Only output backend/pdf_ingestion.py.
```

---

# 🔷 **PROMPT D — Implement Chunker**

```markdown
# PROMPT D — IMPLEMENT backend/chunker.py

Implement:
- Token-based chunking (600 tokens with 100-token overlap).
- Use a tokenizer (HuggingFace tokenizers library).
- Chunk metadata fields:
  - chunk_id
  - doc_id
  - page
  - start_token
  - end_token
  - text
- Function: `chunk_text(doc_id, page_num, text) -> List[Chunk]`

Only output backend/chunker.py.
```

---

# 🔷 **PROMPT E — Implement Embedding Pipeline**

```markdown
# PROMPT E — IMPLEMENT backend/embeddings.py

Implement:
- Load multilingual BGE-M3 embedding model (sentence-transformers or HF).
- GPU acceleration if available.
- Function: `embed_text_list(texts: List[str]) -> List[List[float]]`
- Function: `embed_query(text: str)`.

All embeddings must be float32 or float16.

Only output backend/embeddings.py.
```

---

# 🔷 **PROMPT F — Implement Vector DB Wrapper**

```markdown
# PROMPT F — IMPLEMENT backend/vectordb.py

Implement wrapper using Chroma or FAISS.

Functions:
- `init_vector_db(path)`
- `add_chunks(chunks, embeddings)` — store with metadata
- `search(query_embedding, k=8, filters=None)` — return list of matched chunks
- `delete_doc(doc_id)`
- `list_docs()`

All metadata must be preserved.

Only output backend/vectordb.py.
```

---

# 🔷 **PROMPT G — Implement RAG Orchestration**

```markdown
# PROMPT G — IMPLEMENT backend/rag.py

Implement:
- Retrieval pipeline:
  - embed query
  - search vector DB
  - rank results
- Prompt builder:
  ```
  You are an assistant…
  Question:
  {query}
  Context:
  [Doc {doc_id}, page {page}]
  {chunk_text}
  ---
  ```
- Function: `build_prompt(query, retrieved_chunks)`
- Function: `retrieve_context(query, k=8)`
- Function: `answer_question(query) -> str` (calls LLM.generate)

Only output backend/rag.py.
```

---

# 🔷 **PROMPT H — Implement LLM Wrapper (Gemma + vLLM)**

```markdown
# PROMPT H — IMPLEMENT backend/llm.py

Implement:
- Load Gemma 2/3 Instruct 9B in 4-bit mode
- Option 1: transformers
- Option 2: vLLM server (HTTP client)
- Unified function:
  `generate(prompt: str, max_tokens=512, temperature=0.2, stream=False)`
- If stream=True → yield tokens incrementally

Model must answer in Portuguese.

Only output backend/llm.py.
```

---

# 🔷 **PROMPT I — Implement Chat UI**

```markdown
# PROMPT I — IMPLEMENT ui/chat_ui.py

Requirements:
- Use NiceGUI chat components.
- Display user messages and assistant responses.
- Stream LLM responses token-by-token via websocket.
- Show document citations below each answer.

Functions:
- `chat_page()`
- Handler for sending a query to RAG pipeline.

Only output ui/chat_ui.py.
```

---

# 🔷 **PROMPT J — Implement Document Management UI**

```markdown
# PROMPT J — IMPLEMENT ui/documents_ui.py

Implement:
- Upload control.
- Show list of documents with metadata.
- Delete document.
- Trigger ingestion pipeline.
- Show progress (page X/Y).

Functions:
- `documents_page()`

Only output ui/documents_ui.py.
```

---

# 🔷 **PROMPT K — Implement Main App + Routing**

```markdown
# PROMPT K — IMPLEMENT ui/main_ui.py and app.py

Implement:
- Main layout with sidebar:
  - Chat
  - Documents
- Routing
- State (current theme, user session)
- Launch NiceGUI app

Only output ui/main_ui.py and app.py.
```

---

# 🔷 **PROMPT L — Synthetic Dataset Generator (OpenAI API)**

```markdown
# PROMPT L — IMPLEMENT finetune/generate_dataset.py

Implement:
- Read chunked text
- Ask OpenAI (gpt-4.1-mini) to generate Q&A pairs in Portuguese
- Save JSONL file compatible with SFT/LoRA
- Controls:
  - questions_per_chunk
  - multi-hop generation

Only output finetune/generate_dataset.py.
```

---

# 🔷 **PROMPT M — QLoRA Fine-Tuning Script**

```markdown
# PROMPT M — IMPLEMENT finetune/train_lora.py

Implement:
- Load Gemma 9B in 4-bit
- Apply QLoRA adapters
- Train using JSONL dataset
- Save adapter weights to data/adapters/<theme>

Only output finetune/train_lora.py.
```

---

# 🔷 **PROMPT N — Final Integration and Fix Pass**

```markdown
# PROMPT N — FINAL REVIEW

Task:
- Inspect all generated modules.
- Fix incorrect imports.
- Add missing __init__.py adjustments.
- Ensure path references are correct.
- Generate a final README.md with:
  - installation steps
  - how to run the app
  - how to fine-tune Gemma
  - how to add new documents

Output:
- README.md
- List of integration fixes.

Do NOT rewrite full files unless needed.
```

