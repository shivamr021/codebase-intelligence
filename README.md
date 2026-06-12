# Codebase Intelligence

🔗 Live Demo: https://codebase-intelligence.shivamrathod145.workers.dev/

An AI-powered repository analysis platform that ingests a GitHub repository, builds a dependency graph, generates architecture diagrams, performs semantic code search, and reviews code for potential issues using Large Language Models.

## Overview

Understanding an unfamiliar codebase is difficult.

Developers often spend hours navigating files, tracing imports, identifying entry points, and understanding system architecture before they can contribute effectively.

Codebase Intelligence aims to reduce that onboarding time by combining:

* Static dependency analysis
* Semantic search with vector embeddings
* Architecture summarization
* LLM-powered code understanding
* Interactive graph visualization

The system converts a repository into a searchable knowledge base and provides multiple ways to explore and understand its structure.

---

## Features

### Repository Ingestion

* Clone public GitHub repositories
* Parse supported source files
* Chunk code into semantic units
* Generate embeddings
* Store repository knowledge in Qdrant

### Semantic Code Search

Ask natural language questions such as:

* "How does authentication work?"
* "Where is the dependency graph generated?"
* "How is metadata stored?"

The system retrieves relevant code chunks and uses an LLM to generate contextual answers.

### Dependency Graph Generation

Automatically builds a repository dependency graph by analyzing imports and relationships between files.

Provides:

* Node count
* Edge count
* Circular dependency detection
* Interactive graph visualization

### Architecture Diagram Generation

Generates:

* High-level architecture summary
* Mermaid architecture diagram

Architecture generation combines:

* Dependency graph analysis
* Repository structure
* Selected code samples

### Bug Review

Performs AI-assisted code review and highlights:

* Logic issues
* Crash risks
* Security concerns
* Incorrect API usage
* Import issues

Results are returned as structured findings.

### Metadata Persistence

Repository metadata is cached and persisted:

* Architecture summaries
* Mermaid diagrams
* Dependency graph statistics
* Graph structure

This reduces regeneration costs and improves response times.

---

## Example Workflow

### 1. Ingest Repository

```http
POST /api/v1/ingest
```

Input:

```json
{
  "github_url": "https://github.com/user/repository"
}
```

---

### 2. Generate Architecture

```http
GET /api/v1/diagram?repo_name=my-repo
```

Returns:

* Architecture summary
* Mermaid diagram

---

### 3. Explore Dependency Graph

```http
GET /api/v1/graph?repo_name=my-repo
```

Returns:

* Interactive graph
* Graph statistics

---

### 4. Ask Questions

```http
POST /api/v1/query
```

Example:

```json
{
  "repo_name": "my-repo",
  "question": "How does authentication work?"
}
```

---

### 5. Run Code Review

```http
GET /api/v1/bugs?repo_name=my-repo
```

Returns AI-generated findings about potential issues.

---

## Architecture

### Core Components

#### Ingestion Pipeline

Responsible for:

* Repository cloning
* File discovery
* Code chunking
* Embedding generation
* Vector storage

#### Retrieval Engine

Responsible for:

* Semantic similarity search
* Context retrieval
* Repository knowledge access

#### Dependency Graph Engine

Responsible for:

* Import analysis
* Graph construction
* Graph serialization
* Graph rendering

#### LLM Layer

Responsible for:

* Question answering
* Architecture generation
* Bug review

#### Metadata Layer

Responsible for:

* Architecture persistence
* Graph persistence
* Repository metadata storage

---

## Tech Stack

### Backend

* Python
* FastAPI

### AI / ML

* Groq
* Sentence Transformers
* Vector Embeddings

### Vector Database

* Qdrant

### Graph Analysis

* NetworkX
* PyVis

### Data Validation

* Pydantic

---

## Project Structure

```text
api/
├── routes/
├── models.py

core/
├── embeddings/
├── graph/
├── ingestion/
├── llm/
├── retrieval/
├── storage/

config.py
main.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/shivamr021/codebase-intelligence.git
cd codebase-intelligence
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

---

## Run Locally

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Limitations

This project combines static analysis with LLM reasoning.

As a result:

* Architecture diagrams are generated from available repository context and may not perfectly represent runtime behavior.
* Bug review results should be treated as suggestions, not guaranteed defects.
* Dependency graphs are based on import relationships and not full runtime execution flow.
* Large repositories may require additional context selection strategies.

---

## AI-Assisted Development

This project was developed with significant AI assistance during:

* Architecture design discussions
* Prompt engineering
* Debugging workflows
* Documentation generation
* Refactoring and review

All final implementation decisions, integration work, testing, and project direction were performed by the author.

AI was used as a development assistant rather than an autonomous code generator.

---

## Future Improvements

* Incremental repository updates
* Commit-aware indexing
* Multi-language support
* Advanced static analysis
* Architecture quality improvements
* Repository comparison mode
* Pull request analysis

---

## 🔗 Deployment & Architecture Links

This project utilizes a decoupled architecture, separating the web interface from the AI inference engine.

* **Live Web App:** [Codebase Intelligence (Cloudflare)](https://codebase-intelligence.shivamrathod145.workers.dev/)
* **Frontend Source Code:** [shivamr021/codebase-intelligence-web](https://github.com/shivamr021/codebase-intelligence-web)
* **Backend API Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces/shivamr021/codebase-intelligence)
* **Backend Source Code:** You are here.

---

## Author

**Shivam Rathod**

GitHub:
https://github.com/shivamr021

LinkedIn:
https://www.linkedin.com/in/shivamrathod021/

---

## License

This project is licensed under the MIT License.
