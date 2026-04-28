# Codebase Knowledge AI

Codebase Knowledge AI is a production-oriented Python project for understanding a repository through code-aware ingestion, chunking, embeddings, and relationship graphs.

This implementation is currently at **Step 3**:
- recursive repository scanning
- language detection for Python and JS/TS
- filtering for large repositories
- AST-aware chunking for Python
- structure-aware chunking for JS/TS with line-based fallback
- chunk metadata including symbol name and line ranges
- pluggable embeddings
- local FAISS vector storage
- metadata-aware retrieval
- lightweight reranking for more accurate matches
- relationship graph extraction with NetworkX
- graph-enhanced retrieval context

## Project Structure

```text
codebase_ai/
  cli/
  embedding/
  graph/
  ingestion/
  llm/
  parsing/
  retrieval/
main.py
requirements.txt
README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Scan a repository:

```bash
python3 main.py index /path/to/repository
```

Example:

```bash
python3 main.py index .
```

Filter the scan for large repositories:

```bash
python3 main.py index /path/to/repository \
  --language python \
  --include 'src/**/*.py' \
  --exclude 'tests/*' \
  --max-files 500
```

Preview code chunks:

```bash
python3 main.py index . --show-chunks
```

Use a production embedding model:

```bash
python3 main.py index /path/to/repo \
  --build-vector-index \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Or with OpenAI embeddings:

```bash
export OPENAI_API_KEY=your_key_here
python3 main.py index /path/to/repo \
  --build-vector-index \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

Ask a semantic code question:

```bash
python3 main.py ask "How does retrieval work?"
```

Ask with metadata filters:

```bash
python3 main.py ask "scanner filters" \
  --language python \
  --file-glob 'codebase_ai/ingestion/*' \
  --chunk-type method
```

## Step 3 Output

The scanner now prints:
- relative file path
- detected language
- line count
- file size

The chunk preview prints:
- file path
- chunk type
- function/class name when available
- start and end lines

The vector indexing step persists:
- `.codebase_ai/index/chunks.faiss`
- `.codebase_ai/index/chunks.json`
- `.codebase_ai/index/embeddings.npy`
- `.codebase_ai/index/graph.json`
- `.codebase_ai/index/manifest.json`

The retrieval step supports metadata filters for:
- language
- file glob
- chunk type
- exact symbol name

The relationship graph currently captures:
- file imports
- file to chunk containment
- function and method call relationships
- class/base usage
- API call patterns such as `requests.get(...)`, `httpx.post(...)`, `fetch(...)`, and `axios.get(...)`

The `ask` command now expands top semantic matches with graph neighbors so answers include connected functions, files, and external API usage.

LLM-backed answers are available through:
- OpenAI by default via `OPENAI_API_KEY`
- Ollama as a local fallback via `--llm-provider ollama`

Example:

```bash
python3 main.py ask "How does retrieval work?" --llm-provider openai
python3 main.py ask "How does retrieval work?" --llm-provider ollama --llm-model llama3.1
```

Default local semantic embeddings use `sentence-transformers/all-MiniLM-L6-v2`. You can also use OpenAI embeddings.
