# AI Incident Analysis Copilot

A backend system that helps investigate production incidents by retrieving similar historical incidents and using that context to generate a structured AI-assisted investigation.

I built this project around a problem I encountered while working in production support: when an incident occurs, engineers often need to search logs, previous incidents, monitoring data, and operational knowledge before they can form a useful hypothesis.

This project explores how semantic retrieval and an LLM can reduce that initial investigation time while still treating the AI as a **copilot rather than the source of truth**.

---

## What It Does

When an incident arrives, the application:

1. Validates the incident through FastAPI and Pydantic.
2. Converts the incident into a semantic query embedding.
3. Searches historical incidents stored in PostgreSQL using pgvector.
4. Retrieves the most similar historical incident.
5. Gives that context to Gemini.
6. Requests a structured investigation containing:
   - likely cause
   - investigation steps
   - immediate action
7. Falls back to deterministic guidance if AI generation fails.
8. Persists the incident and its analysis in PostgreSQL.
9. Tracks the incident through its lifecycle.

---

## Architecture

```text
Incoming Incident
       │
       ▼
┌───────────────────┐
│      FastAPI      │
│ Pydantic Validation
└─────────┬─────────┘
          │
          ▼
┌────────────────────────┐
│ Gemini Embedding       │
│ RETRIEVAL_QUERY        │
│ 768 dimensions         │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ PostgreSQL + pgvector  │
│ Cosine Similarity      │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Historical Incident    │
│ Context                │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Gemini Analysis        │
│ Structured Output      │
└─────────┬──────────────┘
          │
     ┌────┴────┐
     │         │
   Success   Failure
     │         │
     ▼         ▼
  Gemini    Fallback
 Analysis   Analysis
     │         │
     └────┬────┘
          │
          ▼
┌────────────────────────┐
│ PostgreSQL             │
│ Incident Persistence   │
└────────────────────────┘
```

---

## Semantic Retrieval

Historical incidents are embedded using:

```text
Model: gemini-embedding-001
Task type: RETRIEVAL_DOCUMENT
Dimensions: 768
```

Incoming incidents use:

```text
Model: gemini-embedding-001
Task type: RETRIEVAL_QUERY
Dimensions: 768
```

The embeddings are normalized before storage or comparison.

PostgreSQL with the `pgvector` extension performs cosine-distance search using:

```sql
embedding <=> query_embedding
```

This allows the system to retrieve incidents based on semantic similarity instead of requiring exact keyword matches.

For example, an incoming description such as:

```text
DB connections appear saturated and requests are
waiting for available connections.
```

successfully retrieves historical knowledge describing:

```text
Incident: Payment Service High Latency

Possible cause:
Database connection pool exhaustion.
```

---

## AI Analysis

The retrieved historical incident is included as context in the Gemini prompt.

Gemini is asked to return structured output matching the application's Pydantic model:

```json
{
  "likely_cause": "Possible cause of the incident",
  "investigation_steps": [
    "Investigation step 1",
    "Investigation step 2"
  ],
  "immediate_action": "Recommended immediate action"
}
```

The application records whether the final analysis came from Gemini or from the fallback path:

```text
analysis_source = "gemini"
```

or:

```text
analysis_source = "fallback"
```

---

## Failure Handling

The application treats Gemini as an external dependency that can fail.

For Gemini server errors:

- maximum generation attempts: **3**
- delay between retries: **3 seconds**
- configured Gemini HTTP timeout: **30 seconds**

Client-side Gemini failures such as quota exhaustion return the fallback analysis without retrying the request three times.

If generation fails, the application produces deterministic investigation guidance and continues to incident persistence.

As long as PostgreSQL is available, a Gemini generation failure therefore does not cause the incident itself to be lost.

### Retrieval Failure

Embedding generation and semantic retrieval are also protected by failure handling.

If that stage raises an exception, the application currently substitutes:

```text
No relevant past incident found.
```

and continues processing.

There is an important limitation in the current design: **a retrieval dependency failure and a successful retrieval with no available historical record are not represented separately**.

A planned improvement is to explicitly record retrieval state, for example:

```text
matched
no_match
failed
```

so degraded retrieval can be distinguished from a genuine lack of relevant knowledge.

If PostgreSQL itself is unavailable, final persistence cannot succeed and the API returns:

```text
503 Service Unavailable
```

---

## Synchronous Latency Trade-Off

Incident analysis currently happens synchronously inside:

```http
POST /incident
```

This keeps the prototype simple, but it has an important latency consequence.

For retried Gemini server failures, the configured generation path can consume up to approximately:

```text
3 × 30-second request timeout
+
2 × 3-second retry delay
=
~96 seconds
```

of timeout/retry budget in the worst case, plus embedding, database, network, and application overhead.

This is one reason a larger production version should separate incident ingestion from AI analysis using asynchronous processing rather than keeping the client request open while external AI work completes.

---

## Incident Lifecycle

Every new incident starts as:

```text
open
```

Supported transitions are:

```text
open → investigating
open → resolved
investigating → resolved
```

Backward transitions are rejected.

For example:

```text
resolved → open
```

returns:

```text
400 Bad Request
```

This lifecycle is enforced by the API rather than being only descriptive metadata.

---

## Persistence

PostgreSQL stores both the retrieval knowledge base and operational incident records.

### Historical Incidents

The `historical_incidents` table contains:

```text
id
content
embedding VECTOR(768)
```

### Operational Incidents

The `incidents` table contains:

```text
incident_id
timestamp
service
alert
severity
description
status
relevant_knowledge
analysis_source
ai_analysis
```

Structured AI analysis is stored as PostgreSQL `JSONB`.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic service response |
| `GET` | `/health` | Application process health |
| `POST` | `/incident` | Analyze and persist an incident |
| `GET` | `/incidents` | List stored incidents |
| `GET` | `/incidents/{incident_id}` | Retrieve one incident |
| `PATCH` | `/incidents/{incident_id}/status` | Update incident lifecycle status |
| `GET` | `/stats` | Incident statistics |
| `GET` | `/knowledge` | View historical knowledge |

The current `/health` endpoint reports application process health. It does **not** yet verify PostgreSQL or Gemini availability.

### Example Incident

```http
POST /incident
```

```json
{
  "service": "checkout-backend",
  "alert": "Requests slowing down",
  "severity": "critical",
  "description": "DB connections appear saturated and requests are waiting for available connections"
}
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8001/docs
```

when the application is running locally.

---

## Historical Knowledge

Example historical incidents are stored in:

```text
knowledge/incidents.txt
```

The seed script:

```text
seed_knowledge.py
```

creates 768-dimensional `RETRIEVAL_DOCUMENT` embeddings and inserts the incidents into PostgreSQL.

Before inserting them, the script runs:

```sql
DELETE FROM historical_incidents;
```

This means reseeding is **repeatable but destructive**.

Running the script again does not keep adding duplicate seed records; instead, the existing historical seed data is cleared and recreated.

This is suitable for the current prototype but would need a proper update/migration strategy for a production knowledge base.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Validation | Pydantic |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Embeddings | Gemini Embeddings |
| AI Analysis | Gemini |
| PostgreSQL Driver | psycopg |
| Testing | pytest |
| API Testing | FastAPI TestClient |
| Configuration | python-dotenv |

---

## Project Structure

```text
ai-incident-copilot/
│
├── knowledge/
│   └── incidents.txt
│
├── main.py
├── seed_knowledge.py
├── test_main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Secrets and database credentials are supplied through a local `.env` file, which is excluded from Git.

---

# Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/shraoshree/ai-incident-copilot.git
cd ai-incident-copilot
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a local `.env` file:

```text
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=your_postgresql_connection_string
```

Do not commit this file.

---

## 5. Configure PostgreSQL

Enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create the historical knowledge table:

```sql
CREATE TABLE historical_incidents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
```

Create the operational incident table:

```sql
CREATE TABLE incidents (
    incident_id VARCHAR(20) PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    service TEXT NOT NULL,
    alert TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    relevant_knowledge TEXT NOT NULL,
    analysis_source TEXT NOT NULL,
    ai_analysis JSONB NOT NULL
);
```

A repository-managed `schema.sql` is planned so this setup can be reduced to a single reproducible database initialization command.

---

## 6. Seed Historical Knowledge

```bash
python seed_knowledge.py
```

Expected output:

```text
Stored: Incident: Payment Service High Latency
Stored: Incident: High API Error Rate
Stored: Incident: High CPU Usage
Knowledge base seeded successfully with 768-dimensional embeddings!
```

---

## 7. Start the API

```bash
uvicorn main:app --reload --port 8001
```

Then open:

```text
http://127.0.0.1:8001/docs
```

---

## 8. Run Tests

```bash
pytest -v
```

Current result:

```text
7 passed
```

External services are mocked where appropriate so the automated unit/API tests do not consume Gemini quota or modify the real PostgreSQL database.

The project has also been manually tested end-to-end against PostgreSQL/pgvector.

---

# Testing

The current automated suite covers:

- health endpoint
- root endpoint
- semantic retrieval
- retrieval dependency failure
- fallback AI-analysis flow
- incident persistence path using mocked storage
- lifecycle transition behavior, including rejection of backward transitions

Unit/API tests provide fast and deterministic validation, while manual end-to-end testing verifies that the actual external integrations work together.

---

# Evaluation

The project currently demonstrates that semantic retrieval works functionally, but **retrieval quality has not yet been formally measured**.

The current knowledge base contains only a small number of manually created historical incidents, and the nearest vector is selected without a similarity threshold.

Therefore, the project does not currently claim a measured retrieval accuracy.

A useful evaluation set would contain labeled examples such as:

```text
Incoming incident
Expected historical incident
Expected cause/category
```

Retrieval can then be evaluated using metrics such as:

```text
Top-1 retrieval accuracy
Top-k retrieval accuracy
```

The observed similarity-distance distributions for correct and incorrect matches can then be used to choose a defensible similarity threshold.

This is preferable to selecting an arbitrary threshold before there is evaluation data.

As the evaluation set grows, it can also be used to detect retrieval regressions when the embedding configuration or knowledge base changes.

---

# Reliability

```text
Incident
   │
   ▼
Semantic Retrieval
   │
   ├── success ──► Historical Context
   │
   └── failure ──► Degraded / No Context
   │
   ▼
Gemini Analysis
   │
   ├── success ──► Structured Analysis
   │
   └── failure ──► Deterministic Fallback
   │
   ▼
PostgreSQL Persistence
   │
   ├── success ──► HTTP 200
   │
   └── failure ──► HTTP 503
```

The current implementation deliberately allows AI failure to degrade the quality of the response rather than automatically making incident ingestion fail.

Persistent storage is treated differently: if the application cannot safely store the incident, the request reports the storage failure instead of pretending the incident was persisted.

---

# Current Limitations

The current version is a backend prototype rather than a production incident-management platform.

Known limitations include:

- small manually curated historical knowledge base
- only one historical incident retrieved per query
- no similarity threshold
- retrieval failure and genuine no-match are not yet represented separately
- synchronous AI processing can produce high request latency
- no authentication or authorization
- no Alertmanager/Grafana webhook ingestion yet
- no automated knowledge-ingestion pipeline
- no database migration framework
- process-only health endpoint
- no formal retrieval-quality benchmark yet

---

# Potential Future Improvements

These are possible extensions identified during development rather than requirements for the current prototype.

### Retrieval observability

Persist an explicit retrieval state:

```text
matched
no_match
failed
```

This will distinguish a successful semantic search from a degraded retrieval dependency.

### Reproducible database setup

Move the PostgreSQL extension and table definitions into:

```text
schema.sql
```

so a fresh database can be initialized from version-controlled schema definitions.

### Continuous Integration

Add a small GitHub Actions workflow that runs:

```bash
pytest -v
```

on pushes and pull requests.

Because the automated tests mock external dependencies, CI should not require production Gemini or PostgreSQL credentials.

### Retrieval evaluation

Build a labeled incident set, measure retrieval quality, and use those results to determine whether a similarity threshold and top-k retrieval improve the system.

### Asynchronous incident processing

If the system needs to handle real alert traffic, separate fast incident ingestion from slower embedding/LLM processing.

A queue or event stream would allow the API to acknowledge ingestion without holding a client connection open during AI processing.

### External alert ingestion

Integrate an actual alert source such as Alertmanager/Grafana so incidents can enter the system through a webhook rather than only through manual API requests.

Infrastructure such as containers, orchestration, or additional messaging components should be added when the deployment or scaling requirements provide a concrete reason for them.

---

## Why This Project Exists

Production incident investigation is rarely just a matter of generating text.

Useful investigation requires context:

```text
What happened before?
What incidents looked similar?
What was the previous cause?
What should the engineer check first?
```

The project therefore focuses on combining backend engineering, persistent operational state, semantic retrieval, and failure-aware AI integration.

The AI output is intended to provide an investigation starting point. The engineer remains responsible for validating the retrieved context and determining the actual cause of the incident.