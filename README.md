# RAG Engine

RAG Engine is a Python application for indexing PDF documents and retrieving
relevant text with Google Gemini embeddings. Its primary workflow stores
documents locally in Chroma. A separate hybrid-search demo combines dense
embeddings with BM25 sparse vectors in Milvus.

This is a command-line application and Python package; it does not currently
include a web server or graphical interface.

## How it works

### Local PDF workflow

1. The ingestion command reads PDF files from `data/` (or `RAG_DATA_DIR`).
2. It extracts their text, splits it into overlapping chunks, and embeds the
   chunks with `gemini-embedding-001`.
3. It stores the chunks and vectors persistently in `chroma_db/` (or
   `RAG_DB_DIR`).
4. The query command finds the three most similar chunks for a question and
   asks Gemini to assess each chunk against the question. It prints the chunks
   and a `YES`/`NO` verdict for each one.

The local CLI currently reports matching chunks and per-chunk verification;
it does not print a synthesized answer. The package's `LocalQueryEngine` also
provides a `generate_answer()` method for grounded answer generation.

### Hybrid Milvus demo

The hybrid demo creates dense and sparse vectors, stores sample documents in
Milvus, and performs a fused search. It uses hard-coded example text rather
than PDFs from the local ingestion workflow.

> **Warning:** the demo recreates the `enterprise_resume_rag` collection each
> time it runs. The existing collection with that name is dropped first.
> Do not run it against a Milvus instance if that collection contains data you
> need to keep.

## Requirements

- Python 3.10 or newer
- A Google AI API key for embeddings and Gemini requests
- For the optional hybrid demo: Docker and Docker Compose, or another reachable
  Milvus server

## Step-by-step: run the local PDF workflow

Run these commands from the project root.

### 1. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the application

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation makes the `rag_engine` package importable and installs
the `rag-ingest`, `rag-query`, and `rag-hybrid-demo` commands. Alternatively,
`python -m pip install -r requirements.txt` installs this project from the
requirements file.

### 3. Configure the Google API key

Copy `.env.example` to `.env` in the project root and replace the placeholder:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
```

For example, on macOS/Linux:

```bash
cp .env.example .env
```

Do not commit a real API key. The repository's `.gitignore` excludes `.env`.
You can set `GOOGLE_API_KEY` in your shell instead if you prefer.

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_EMBEDDING_MODEL` | No override | First embedding model to try in the hybrid workflow |
| `RAG_DATA_DIR` | `<project-root>/data` | Directory containing input PDF files |
| `RAG_DB_DIR` | `<project-root>/chroma_db` | Directory where Chroma persists its database |

Relative directory values are resolved from the current working directory.

### 4. Add PDF files

Put one or more `.pdf` files in the `data/` directory. The ingestion command
creates this directory if it is missing. Only PDF files are currently ingested;
the `.txt` setting in the configuration is not used by the local ingestion
command.

### 5. Ingest the PDFs

```bash
rag-ingest
```

The command prints how many text chunks were indexed. If no PDFs are present,
it prints a warning and indexes nothing. Ingestion calls the Google embedding
service and can take time depending on document size and network/API limits.

### 6. Search the indexed documents

```bash
rag-query "What technical skills are listed?"
```

Quote the full question so it is passed as one command-line argument. The
command prints up to three matching text chunks and asks Gemini for a `YES` or
`NO` verification verdict for each chunk. Run ingestion before querying; the
query command does not build the database itself.

The Chroma database persists between runs. The ingestion command processes all
PDFs each time, so running it repeatedly on the same files can add duplicate
chunks. There is currently no automatic deduplication or update/removal
workflow.

## Optional: run the hybrid Milvus demo

The supplied Compose configuration uses an ARM64 Milvus image. Check that this
image is compatible with your machine before starting the services.

From the project root:

```bash
docker compose up -d
docker compose ps
rag-hybrid-demo
```

The demo needs the same `GOOGLE_API_KEY` configuration and a reachable Milvus
service on `localhost:19530`. It inserts two built-in sample documents,
performs a search for `"Rust"`, and prints the ranked matches.

To stop the services while preserving their named data volumes:

```bash
docker compose down
```

Avoid `docker compose down --volumes` unless you intend to delete the Milvus,
MinIO, and etcd data volumes.

## Run with the compatibility scripts

The original root-level scripts are still supported:

```bash
python run_ingest.py
python run_query.py "What technical skills are listed?"
python run_pipeline.py
```

The installed commands are the recommended interface.

## Project structure

```text
rag_engine/
├── cli/        # Command-line entry points
├── ingestion/  # PDF loading and chunking
├── retrieval/  # Local Chroma and hybrid Milvus search
├── storage/    # Chroma and Milvus providers
├── config.py   # Data/database paths and chunk settings
└── embeddings.py
data/           # Input PDFs (created if missing)
chroma_db/      # Persistent local vector database (created on use)
```

## Troubleshooting

- **`GOOGLE_API_KEY` is missing:** check that `.env` is in the project root or
  the current working directory, or export the variable in your shell.
- **No PDFs were processed:** confirm that the files end in `.pdf` and are in
  `data/` or the directory configured by `RAG_DATA_DIR`.
- **No matching chunks:** make sure ingestion completed successfully and that
  the query relates to content in the indexed PDFs.
- **Milvus connection error:** confirm the Compose services are running and
  that port `19530` is available.
- **Embedding model error:** check API-key access, network connectivity, and
  the Google API's available embedding models.
