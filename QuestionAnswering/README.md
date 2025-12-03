# Local Multilingual Question Answering Suite

This repository hosts a NiceGUI-based application that ingests multilingual PDFs, builds a retrieval-augmented generation (RAG) pipeline powered by Gemma Instruct 9B (4-bit), and supports QLoRA fine-tuning workflows.

## 1. Installation

1. **Prerequisites**
   - Python 3.10+
   - CUDA-capable GPU with ≥20 GB VRAM (for Gemma + QLoRA)
   - System packages for PDF rendering: `poppler-utils` (for `pdf2image`) and `libgl1`.
2. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
3. **Install Python dependencies**
   ```bash
   pip install nicegui pdfplumber pdf2image paddleocr pymupdf sentence-transformers
   pip install chromadb transformers accelerate bitsandbytes peft
   pip install paddlepaddle==2.6.0 --extra-index-url https://www.paddlepaddle.org.cn/whl/mkl/
   pip install openai
   ```
   > Adjust `paddlepaddle` wheel index for your platform and install additional GPU drivers if required.

## 2. Running the App

1. Initialize the environment variables (optional but recommended):
   ```bash
   export LLM_BACKEND=transformers   # or vllm
   export LLM_MODEL_NAME=google/gemma-2-9b-it
   export BGE_MODEL_NAME=BAAI/bge-m3
   ```
2. Launch the NiceGUI server from the repository root:
   ```bash
   python -m project.app
   ```
3. Open the browser at the address printed by NiceGUI (default `http://localhost:8080`).
4. Use the sidebar to switch between **Chat** and **Documentos**.

## 3. Adding New Documents

1. Navigate to the **Documentos** page.
2. Upload PDFs via the `Enviar PDF` control; large files (10–1000 pages) are handled in a streaming fashion.
3. Start ingestion for each uploaded document. The UI displays real-time progress (`Página X/Y`).
4. During ingestion the pipeline extracts text (OCR when required), chunks it, embeds with BGE-M3, and writes vectors into Chroma stored under `project/data/vectordb`.
5. Once ingested, the content becomes available to the chat interface with citations per response.

> Advanced: You can also ingest programmatically via `project/backend/pdf_ingestion.py.ingest_pdf` if you want to automate uploads, ensuring `vectordb.init_vector_db()` has been called.

## 4. Fine-tuning Gemma with QLoRA

### 4.1 Generate a Dataset
1. Populate `project/data/chunks` by ingesting documents.
2. Provide your OpenAI API key: `export OPENAI_API_KEY=...`.
3. Run the generator:
   ```bash
   python project/finetune/generate_dataset.py \
       --output project/data/finetune/qa_dataset.jsonl \
       --questions-per-chunk 4 \
       --multi-hop
   ```

### 4.2 Train QLoRA Adapters
1. Ensure you have enough GPU memory (20 GB VRAM recommended) and install `bitsandbytes`, `accelerate`, and `peft`.
2. Launch training:
   ```bash
   python project/finetune/train_lora.py \
       --data project/data/finetune/qa_dataset.jsonl \
       --theme my_theme \
       --epochs 3 \
       --batch-size 1 \
       --lr 2e-4
   ```
3. The adapters will be stored in `project/data/adapters/my_theme/`. Point the runtime to them via PEFT utilities when loading Gemma for inference.

## 5. vLLM Mode (Optional)

- Run a separate vLLM server hosting Gemma 9B (4-bit) and set `LLM_BACKEND=vllm` and `VLLM_BASE_URL` accordingly. The NiceGUI chat will stream tokens via HTTP.

## 6. Directory Layout

```
project/
  app.py                  # NiceGUI entrypoint
  backend/                # OCR, ingestion, chunking, embeddings, vector DB, RAG, LLM
  ui/                     # Chat and document management interfaces
  finetune/               # Dataset generation + QLoRA scripts
  data/                   # PDFs, chunks, vector DB, adapters, generated datasets
```

You're ready to ingest multilingual PDFs, chat in Portuguese, and optionally fine-tune Gemma using RAG-curated data.
