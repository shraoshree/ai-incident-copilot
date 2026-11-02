# Incident Copilot

Incident investigation API built with **Python, FastAPI, Gemini, PostgreSQL, and pgvector**. It uses historical incidents and runbooks to suggest investigation steps, while keeping incident tracking and remediation decisions with the engineer.

## What it does

- **Incident investigation:** Accepts service, alert, severity, and description; returns a likely cause, investigation steps, and an immediate action.
- **Tool-calling agent:** Uses Gemini function calling to search historical incident knowledge. Tool access is read-only, arguments are validated, and execution is limited to three rounds.
- **Retrieval:** Supports keyword matching by default and optional semantic search with Gemini embeddings and PostgreSQL/pgvector (768-dimensional vectors, cosine similarity, top-3 matches).
- **Resilience:** Validates structured responses, retries temporary model failures in the standard analysis path, and provides fallback investigation guidance when the model or vector retrieval is unavailable.
- **Incident management:** Stores incidents, tracks `open → investigating → resolved` transitions, and exposes incident statistics.
- **Testing:** Includes API and agent tests, a live Gemini function-calling check, real pgvector integration tests, and a small labeled retrieval evaluation.

## Architecture

```text
Alert / Incident
      |
      v
  FastAPI + Pydantic
      |
      +---- POST /incident --------> Keyword context + structured Gemini analysis
      |
      +---- POST /incident/agent --> Gemini function calling
                                            |
                                            v
                                  Historical knowledge search
                                    |                  |
                              PostgreSQL/pgvector   Keyword fallback
                                    |                  |
                                    +--------+---------+
                                             |
                                   Validated investigation
                                             |
                                   JSON incident store
```

The agent does **not** access live production telemetry or execute remediation commands. Suggested causes are hypotheses, not confirmed root causes.

## Quick start

Requires Python 3.11+ and a Gemini API key.

```bash
git clone https://github.com/shraoshree/ai-incident-copilot.git
cd ai-incident-copilot
python -m pip install -r requirements.txt
```

Create a local `.env` file (never commit it):

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Start the API:

```bash
uvicorn main:app --reload --port 8000
```

Interactive API docs: **http://127.0.0.1:8000/docs**

### Investigate an incident

```bash
curl -X POST http://127.0.0.1:8000/incident/agent \
  -H "Content-Type: application/json" \
  -d '{
    "service": "payment-service",
    "alert": "High API latency",
    "severity": "critical",
    "description": "p99 latency increased to 2800ms after a traffic spike"
  }'
```

The response contains the saved incident, its structured investigation, `analysis_source` (`gemini_agent` or `fallback`), and a tool-call trace.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/incident/agent` | Investigate with bounded, read-only tool calling |
| POST | `/incident` | Analyze an incident using retrieved keyword context |
| GET | `/incidents` | List incidents; optionally filter by status |
| GET | `/incidents/{incident_id}` | Get an incident |
| PATCH | `/incidents/{incident_id}/status` | Update incident status |
| GET | `/stats` | Incident counts and severity statistics |
| GET | `/health` | Service health |
| GET | `/knowledge` | View sample historical knowledge |

## Semantic retrieval (optional)

The agent uses the local knowledge file by default. To enable semantic retrieval, start a PostgreSQL database with the `vector` extension and configure `DATABASE_URL`. The repository includes a local Docker Compose setup.

```bash
# Set GEMINI_API_KEY and POSTGRES_PASSWORD in your environment first
docker compose up -d --build
docker compose exec api python scripts/index_knowledge.py
```

The indexing command embeds the records in `knowledge/incidents.txt` and stores them in pgvector. When `DATABASE_URL` is set, the agent searches those embeddings; database errors fall back to keyword matching. Embedding calls require API access and may incur usage charges.

## Tests and evaluation

```bash
python -m pytest -q -m "not live and not pgvector"
python evals/evaluate.py
```

GitHub Actions additionally runs integration tests against a **real pgvector-enabled PostgreSQL service**, covering similarity ranking, idempotent indexing, query limits, and fallback behavior. A separate opt-in workflow exercises Gemini function calling with a repository secret.

The retrieval evaluation reports **Hit@3** and **MRR@3** over **44 synthetic queries**: 36 labeled positive examples (15 direct and 21 paraphrased), four out-of-domain negatives and four multi-label ambiguous examples. The knowledge base contains **21 historical incident records**, including 18 distractors. Report **Hit@1, Hit@3 and MRR@3** by category, plus negative-query abstention. These author-written examples are a development benchmark, not held-out or production accuracy. Semantic scores require separately indexing the full corpus and running `python evals/evaluate.py --backend both` with a configured database and embedding API; the default offline run measures keyword retrieval only.

## Tech stack

**Backend:** Python, FastAPI, Pydantic  
**Model integration:** Google Gemini SDK, structured output, function calling  
**Retrieval:** Keyword search, Gemini embeddings, PostgreSQL/pgvector  
**Testing:** pytest, GitHub Actions, labeled retrieval evaluations  
**Packaging:** Docker, Docker Compose

## Scope and limitations

This is a locally runnable engineering project, not a production deployment. Incident records still use a JSON file; PostgreSQL stores **knowledge embeddings only**. The JSON store is unsuitable for concurrent production writes and is ephemeral inside an unmounted container. The service has no authentication or live monitoring integrations.

AWS ECS task-definition files are included as a reference, but **the application has not been deployed to AWS**. For deployment notes and operational considerations, see [Deployment and evaluation notes](docs/DEPLOYMENT_AND_EVALUATION.md).
