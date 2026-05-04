# AI Regulatory Knowledge Assistant

A portfolio project demonstrating a document-grounded AI assistant for regulatory and technical knowledge retrieval.

This project is designed for AI Engineer, RAG Engineer, LLM Engineer, and AI Automation role applications. It combines a Python/FastAPI backend, Streamlit frontend, PostgreSQL with pgvector, OpenAI embeddings, retrieval-augmented generation, LangGraph orchestration, query history, citation-style source references, Docker setup, and evaluation workflows.

Important: this project is for education and portfolio demonstration only. It is not production software and does not provide legal, regulatory, quality, or compliance advice. Sample documents should be synthetic, public, or otherwise safe to use.

## 1. Project Overview

The AI Regulatory Knowledge Assistant lets a user ask natural-language questions about a local collection of regulatory or technical documents. The app ingests documents, chunks them, creates embeddings, stores vectors in PostgreSQL using pgvector, retrieves relevant chunks for a question, and generates a grounded answer with source references.

It also includes an Excel compliance-checking MVP. Users can upload a `.xlsx` workbook with product values, run selected deterministic FSANZ infant-formula numeric checks, or run a broader AI-assisted review that retrieves relevant regulatory context and asks the LLM to screen each row with citations.

Supported document types:

- `.txt`
- `.pdf`
- `.docx`

Main user flow:

```text
upload or place documents
  -> ingest metadata
  -> chunk text
  -> generate embeddings
  -> ask a question
  -> retrieve relevant chunks
  -> generate grounded answer
  -> display sources and query history
```

## 2. Why I Built This

I built this project to demonstrate practical AI engineering skills beyond a notebook-only workflow. Regulatory and technical documents are often long, dense, and difficult to search manually, which makes them a suitable domain for a RAG assistant.

The project shows how I think about:

- grounding LLM answers in retrieved evidence
- building a simple backend/frontend application around an AI workflow
- storing and searching embeddings with PostgreSQL and pgvector
- structuring a maintainable Python project
- using LangGraph to make the RAG workflow easier to explain and extend
- evaluating the system with both simple checks and optional RAGAS metrics

## 3. Key Features

- FastAPI backend with health check, `/ask`, `/history`, `/check-excel`, and `/review-excel-ai` endpoints
- Streamlit frontend for asking questions and reviewing sources
- PostgreSQL database with pgvector vector search
- Document ingestion for `.txt`, `.pdf`, and `.docx`
- Text chunking with overlap for better retrieval context
- OpenAI `text-embedding-3-small` embeddings
- pgvector similarity search over document chunks
- Grounded answer generation using retrieved context
- Citation-style source labels such as `[Source 1]`
- Query history stored in PostgreSQL
- LangGraph workflow with retrieval, sufficiency check, generation, verification, and final response nodes
- Simple CSV evaluation for regression checks
- Optional RAGAS evaluation for advanced RAG diagnostics
- Docker Compose setup for PostgreSQL, backend, and frontend
- Excel upload checker with deterministic and AI-assisted review modes

## 4. Tech Stack

Backend:

- Python
- FastAPI
- Pydantic
- psycopg2
- python-dotenv
- openpyxl

Frontend:

- Streamlit
- requests

AI and RAG:

- OpenAI embeddings
- OpenAI chat model
- LangGraph
- pgvector
- Optional RAGAS

Database:

- PostgreSQL
- pgvector extension

Dev and deployment:

- Docker
- Docker Compose
- pytest

## 5. Architecture

```text
                +----------------------+
                |   Streamlit Frontend |
                |   localhost:8501     |
                +----------+-----------+
                           |
                           | HTTP /ask, /history, /check-excel, /review-excel-ai
                           v
                +----------------------+
                |   FastAPI Backend    |
                |   localhost:8000     |
                +----------+-----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+--------------------+          +----------------------+
| OpenAI API         |          | PostgreSQL + pgvector |
| embeddings + chat  |          | documents, chunks,    |
+--------------------+          | query history         |
                                +----------------------+
```

Excel compliance checks are handled separately from the LLM workflow:

```text
Excel workbook
  -> parse rows with openpyxl
  -> normalize units and parameters
  -> apply deterministic Python rules
  -> return PASS, FAIL, or NEEDS_REVIEW with source notes
```

This separation is intentional. The LLM can help answer document questions, but numerical compliance calculations should be explicit, testable Python logic.

The app also provides an AI-assisted Excel review path for broader screening:

```text
Excel workbook
  -> parse rows with openpyxl
  -> retrieve regulatory chunks for each row
  -> ask the LLM to assess only from retrieved context
  -> return PASS, FAIL, NEEDS_REVIEW, or INSUFFICIENT_CONTEXT
  -> display citations and retrieved source excerpts
```

This mode is broader than deterministic checking, but it is not final compliance confirmation.

Project structure:

```text
ai-regulatory-knowledge-assistant/
  backend/
    app/
      api/
      database/
      ingestion/
      rag/
      graph/
      evaluation/
      utils/
    main.py
    Dockerfile
  frontend/
    api_client.py
    streamlit_app.py
    Dockerfile
  data/
    sample_docs/
  evaluation/
  scripts/
  sql/
  tests/
  docker-compose.yml
  requirements.txt
  requirements-ragas.txt
  .env.example
```

## 6. RAG Workflow

The core RAG pipeline is:

```text
question
  -> create question embedding
  -> retrieve similar document chunks from pgvector
  -> format retrieved chunks as context
  -> ask the chat model to answer only from that context
  -> return answer and source metadata
```

The answer prompt instructs the model to:

- answer only from the retrieved context
- avoid unsupported claims
- keep the answer concise and professional
- cite context using labels such as `[Source 1]`
- return a fallback message when the context is insufficient

Fallback message:

```text
I do not have enough information in the provided documents to answer this confidently.
```

## 7. LangGraph Workflow

The `/ask` endpoint uses a small LangGraph workflow instead of a direct single function call.

```text
question
  -> retrieve_context
  -> check_context_sufficiency
      -> if insufficient: return_final_answer
      -> if sufficient: generate_answer
  -> verify_answer
  -> return_final_answer
```

State fields:

- `question`
- `retrieved_chunks`
- `context_is_sufficient`
- `draft_answer`
- `verification_result`
- `final_answer`
- `sources`

Node responsibilities:

- `retrieve_context`: retrieves the top matching chunks from pgvector.
- `check_context_sufficiency`: checks whether the retrieved context is strong enough to answer.
- `generate_answer`: generates a grounded draft answer using retrieved chunks.
- `verify_answer`: checks that the answer includes a source citation.
- `return_final_answer`: returns the verified answer or a cautious fallback.

This is intentionally not a multi-agent system. The goal is to make the workflow clear, testable, and easy to explain in interviews.

## 8. Database Design

The database is intentionally small and focused on the RAG use case.

Tables:

```text
documents
  id
  filename
  source_type
  uploaded_at

document_chunks
  id
  document_id
  chunk_text
  chunk_index
  page_number
  embedding vector(1536)
  created_at

query_history
  id
  question
  answer
  created_at
```

The `document_chunks.embedding` column uses pgvector with 1536 dimensions for `text-embedding-3-small`.

## 9. How To Run Locally

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/regulatory_ai
```

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
```

Check the database:

```bash
python scripts/test_db_connection.py
```

Add safe documents to:

```text
data/sample_docs/
```

Prepare the RAG database:

```bash
python scripts/ingest_documents.py
python scripts/chunk_documents.py
python scripts/embed_chunks.py
```

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```text
FastAPI docs: http://localhost:8000/docs
Streamlit:    http://localhost:8501
```

### Excel Compliance Checker

The Streamlit app includes an `Excel compliance checker` tab. Upload a `.xlsx` workbook with these columns:

```text
parameter,value,unit,category
Energy,2720,kJ/L,
Protein,15,g/L,milk-based
Docosahexaenoic acid,10,mg/100 kJ,
Total trans fatty acids,3,% of total fatty acids,
```

Required columns:

- `parameter`
- `value`
- `unit`

Optional columns:

- `category`
- `notes`

Current deterministic checks:

- Energy content: `2510-2930 kJ/L`
- Milk-based protein: `0.43-0.72 g/100 kJ`
- Non-milk-based protein: `0.54-0.72 g/100 kJ`
- Docosahexaenoic acid: `<= 12 mg/100 kJ`
- Total trans fatty acids: `<= 4% of total fatty acids`

Protein values entered as `g/L` can be converted to `g/100 kJ` when the workbook also includes energy in `kJ/L`.

The same tab also includes `Run AI-assisted review`. This mode:

- retrieves relevant regulatory chunks for each Excel row
- asks the chat model to assess the row using only retrieved context
- returns `PASS`, `FAIL`, `NEEDS_REVIEW`, or `INSUFFICIENT_CONTEXT`
- shows source excerpts used for each row
- should be treated as screening support, not final compliance advice

Use deterministic checks when exact rules are already coded. Use AI-assisted review when you want broader coverage across workbook rows and are comfortable reviewing the cited evidence.

Run tests:

```bash
pytest
```

## 10. How To Run With Docker

From a fresh clone:

```bash
git clone <your-repo-url>
cd ai-regulatory-knowledge-assistant
cp .env.example .env
```

Edit `.env` and add:

```text
OPENAI_API_KEY=your_key_here
```

Build and start the database:

```bash
docker compose up -d postgres
```

Verify the backend can connect to PostgreSQL and pgvector:

```bash
docker compose run --rm backend python scripts/test_db_connection.py
```

Add `.txt`, `.pdf`, or `.docx` files to:

```text
data/sample_docs/
```

Prepare documents, chunks, and embeddings:

```bash
docker compose run --rm backend python scripts/ingest_documents.py
docker compose run --rm backend python scripts/chunk_documents.py
docker compose run --rm backend python scripts/embed_chunks.py
```

Start the app:

```bash
docker compose up backend frontend
```

Open:

```text
FastAPI health check: http://localhost:8000/
FastAPI Swagger docs: http://localhost:8000/docs
Streamlit app:        http://localhost:8501
```

Stop services:

```bash
docker compose down
```

Remove the database volume for a clean reset:

```bash
docker compose down -v
```

Docker note: inside Compose, the backend uses `postgres` as the database hostname. Local development uses `localhost`.

## 11. Example Questions

Example questions depend on the documents you place in `data/sample_docs/`.

For public FSANZ infant formula documents:

- What does Standard 2.9.1 regulate?
- Does Standard 2.9.1 apply in New Zealand?
- What does Division 3 of Standard 2.9.1 set out?
- How must energy content be calculated for infant formula products under Schedule 29?
- What nitrogen-to-protein conversion factor is used for infant formula products?
- How is vitamin A content calculated for infant formula products under Schedule 29?
- What is the limit for total trans fatty acids in infant formula products under Schedule 29?

Example Excel compliance checker rows:

- `Energy,2720,kJ/L,`
- `Protein,15,g/L,milk-based`
- `Docosahexaenoic acid,13,mg/100 kJ,`
- `Total trans fatty acids,3,% of total fatty acids,`

For synthetic technical documents:

- What should be checked before a product label is approved?
- What records should be kept for a product change?
- What information should be included in technical documentation?

## 12. Evaluation

The project includes two evaluation paths.

### Simple CSV Evaluation

The simple evaluator runs test questions through the LangGraph RAG workflow and writes a CSV of generated answers, sources, and keyword checks.

```bash
python scripts/run_evaluation.py
```

Run a PDF-specific evaluation set:

```bash
python scripts/run_evaluation.py \
  --questions evaluation/test_questions_pdf.csv \
  --results evaluation/results_pdf.csv
```

The output includes:

```text
question
expected_answer
generated_answer
retrieved_sources
answer_contains_expected_info
notes
```

This is a first-pass regression check. It is not a full semantic or regulatory correctness evaluation.

### Optional RAGAS Evaluation

RAGAS is optional and not required to run the app.

Install optional dependencies:

```bash
pip install -r requirements-ragas.txt
```

Run RAGAS:

```bash
python scripts/run_ragas_evaluation.py \
  --questions evaluation/test_questions_pdf.csv \
  --results evaluation/ragas_results_pdf.csv \
  --include-context-recall
```

Metrics:

- `faithfulness`: whether generated claims are supported by retrieved context.
- `answer_relevancy`: whether the answer addresses the question.
- `context_precision`: whether retrieved chunks are relevant and ranked well.
- `context_recall`: whether retrieved context covers the reference answer or expected context.

RAGAS scores should be treated as diagnostic signals. They can be affected by LLM judge behavior, PDF extraction quality, reference wording, chunking, and retrieval settings.

## 13. Screenshots

Add screenshots here after running the app locally or with Docker.

Suggested screenshots:

- Streamlit home screen with a regulatory question entered
- Generated answer with source expanders open
- Excel compliance checker with uploaded workbook results
- Recent query history in the sidebar
- FastAPI Swagger docs at `/docs`
- Optional evaluation CSV or terminal summary

```text
docs/screenshots/
  streamlit-answer.png
  streamlit-sources.png
  fastapi-docs.png
  evaluation-results.png
```

## 14. Limitations

- This is not production software.
- This is not legal, regulatory, compliance, or quality advice.
- Answers are only as good as the documents ingested and retrieved.
- PDF extraction can introduce formatting issues, especially with tables.
- The current retriever uses vector search only, not hybrid keyword plus vector search.
- Citations point to chunks, not exact clauses, paragraphs, or page coordinates.
- The context sufficiency and verification checks are intentionally simple.
- Query history stores questions and answers but does not include user accounts or authentication.
- There is no document upload UI yet; documents are added through `data/sample_docs/`.
- The Excel checker supports only a small set of deterministic demonstration rules.
- The AI-assisted Excel review depends on retrieval quality and model interpretation.
- AI-assisted Excel statuses are screening outputs, not authoritative compliance determinations.
- Excel checks do not replace expert regulatory, quality, or legal review.
- No production security hardening, access control, audit logging, or deployment pipeline is included.
- RAGAS metrics are useful for diagnostics but do not prove regulatory correctness.

## 15. Future Improvements

- Add a document upload workflow in the Streamlit UI.
- Expand the Excel checker with more structured templates and additional rules.
- Add human-review override fields and reviewer notes for AI-assisted Excel results.
- Add source-clause extraction so AI-assisted Excel reviews cite narrower clauses.
- Add table extraction from PDFs to help map clause references into deterministic checks.
- Add hybrid retrieval using keyword search plus vector search.
- Improve PDF table extraction and clause-level citation handling.
- Add reranking for retrieved chunks.
- Add user-selectable source filters.
- Add richer query history with retrieved sources and model settings.
- Add authentication for multi-user usage.
- Add structured observability for latency, retrieval scores, and model usage.
- Add more robust evaluation datasets with expected source passages.
- Add automated screenshots and demo assets for GitHub.
- Add CI checks for tests and Docker builds.

## 16. Interview Talking Points

- Why RAG is useful for dense regulatory and technical documents.
- How embeddings and pgvector support semantic search.
- Why chunk size and overlap affect retrieval quality.
- How the system reduces unsupported answers by grounding generation in retrieved context.
- Why the LangGraph workflow is intentionally simple and explainable.
- How context sufficiency and answer verification work in the current version.
- Why query history belongs in the database rather than only in frontend state.
- How simple CSV evaluation differs from RAGAS evaluation.
- Why RAGAS metrics should be interpreted carefully.
- What would be required before using a similar system in a real regulated environment.

## Disclaimer

This project is an educational portfolio application. It uses synthetic or public sample documents only and does not contain confidential company information. It is not intended for production use and must not be relied on as legal, regulatory, compliance, medical, or quality advice.
