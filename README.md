# AI Incident Analysis Copilot

A backend service that helps engineers investigate production incidents by combining historical incident knowledge with LLM-generated troubleshooting guidance.

I built this project around a problem I encountered in production support: an alert tells an engineer that something is wrong, but identifying what to investigate often requires correlating the alert with previous incidents, runbooks, logs, metrics, and recent changes.

This prototype explores whether historical incident context can be retrieved automatically and supplied to an LLM to produce a more useful starting point for investigation.

## How It Works

```text
Incoming Incident
       |
       v
   FastAPI API
       |
       v
Pydantic Validation
       |
       v
Historical Knowledge Retrieval
       |
       | retrieved incident context
       v
 Prompt Construction
       |
       | current incident + historical context
       v
 Gemini LLM
       |
       v
Structured Analysis
       |
       v
Incident Storage
```

The LLM is used as an investigation assistant, not as the source of truth. Historical incident information is retrieved first and included as context in the model request.

## Retrieval

Historical incident knowledge currently lives in:

```text
knowledge/incidents.txt
```

The file contains incident records separated into blocks. Each block contains information such as:

```text
Incident: Payment Service High Latency
Symptoms: p99 API latency above 2000ms.
Possible cause: Database connection pool exhaustion.
Recommended checks:
1. Check active database connections.
2. Check slow database queries.
3. Check application CPU and memory.
4. Check recent deployments.
```

For an incoming incident, the service combines:

```text
service + alert + description
```

and performs simple keyword matching against the historical incident blocks.

Words shorter than five characters are ignored to reduce matches on common words. The first historical block containing a matching keyword is returned as context.

For example:

```text
Incoming alert:
High API latency

Description:
p99 latency increased to 2800ms
```

can retrieve:

```text
Incident: Payment Service High Latency
Possible cause: Database connection pool exhaustion
```

This retrieval mechanism is intentionally simple. It demonstrates the retrieval → context → generation flow without introducing a vector database yet.

Its main limitation is semantic matching. For example:

```text
"database connections exhausted"
```

and:

```text
"DB pool saturation"
```

may describe the same failure but share few useful keywords.

A future version would use embeddings with PostgreSQL/pgvector so incidents can be retrieved based on semantic similarity rather than exact word overlap.

## Prompt and Grounding

The model does not receive only the incoming alert.

The prompt contains two sections:

```text
CURRENT INCIDENT
- service
- alert
- severity
- description

RELEVANT HISTORICAL KNOWLEDGE
- retrieved incident/runbook context
```

The model is then asked to produce:

```text
1. Likely cause
2. Investigation steps
3. Immediate action
```

This keeps the current incident separate from retrieved historical information and gives the model concrete operational context before it generates troubleshooting guidance.

The current implementation uses one retrieved historical incident. Expanding retrieval to ranked top-k results is part of the next retrieval iteration.

## Structured LLM Output

The project currently uses:

```text
gemini-3.6-flash
```

Instead of asking Gemini for arbitrary text and manually parsing JSON, the application provides a Pydantic response schema:

```python
class IncidentAnalysis(BaseModel):
    likely_cause: str
    investigation_steps: list[str]
    immediate_action: str
```

The Gemini request specifies:

```python
response_mime_type = "application/json"
response_schema = IncidentAnalysis
```

The parsed response is therefore validated against the expected structure before being stored.

If Gemini returns a response that cannot be parsed into the expected structure, the application returns predefined investigation guidance rather than allowing malformed model output into the normal response path.

## External API Failure Handling

Gemini is an external dependency, so an AI provider failure should not prevent the incident-management API from continuing to operate.

The application retries Gemini server-side failures up to three times.

```text
Gemini request
     |
     v
 attempt 1
     |
   failure
     |
     v
 wait 3 seconds
     |
     v
 attempt 2
     |
   failure
     |
     v
 wait 3 seconds
     |
     v
 attempt 3
     |
   failure
     |
     v
fallback investigation guidance
```

The current implementation retries `google.genai.errors.ServerError`, which covers temporary server-side failures such as the `503 Service Unavailable` responses encountered during development.

Retries currently use a fixed three-second delay. Exponential backoff and jitter would be preferable at larger scale.

Most importantly, an unavailable LLM does not cause the incident API itself to fail. The incident can still be recorded and investigated using retrieved historical knowledge and fallback guidance.

## End-to-End Example

### Request

```bash
curl -X POST "http://127.0.0.1:8001/incident" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "payment-service",
    "alert": "High API latency",
    "severity": "critical",
    "description": "p99 latency increased to 2800ms"
  }'
```

### Retrieved Historical Context

For this incident, the current keyword retriever selects the historical record:

```text
Incident: Payment Service High Latency
Symptoms: p99 API latency above 2000ms.
Possible cause: Database connection pool exhaustion.
Recommended checks:
1. Check active database connections.
2. Check slow database queries.
3. Check application CPU and memory.
4. Check recent deployments.
```

### Stored Incident

A successful request produces a record with this shape:

```json
{
  "message": "Incident analyzed and saved",
  "incident": {
    "incident_id": "INC-23b667ec",
    "timestamp": "2026-09-21T17:30:00+00:00",
    "service": "payment-service",
    "alert": "High API latency",
    "severity": "critical",
    "description": "p99 latency increased to 2800ms",
    "status": "open",
    "relevant_knowledge": "Incident: Payment Service High Latency\nSymptoms: p99 API latency above 2000ms.\nPossible cause: Database connection pool exhaustion...",
    "ai_analysis": {
      "likely_cause": "Database connection pool exhaustion",
      "investigation_steps": [
        "Check active database connections",
        "Inspect slow database queries",
        "Check application CPU and memory",
        "Review recent deployments"
      ],
      "immediate_action": "Check database connection saturation and recent application changes."
    }
  }
}
```

The incident ID above is an example; IDs are generated dynamically for new incidents.

## Incident Lifecycle

Every new incident receives a unique ID and begins with:

```text
status = open
```

The API currently supports:

```text
open
investigating
resolved
```

Status can be changed through:

```text
PATCH /incidents/{incident_id}/status
```

The current prototype validates that the requested status belongs to this allowed set.

It does not yet enforce transition rules. For example, a resolved incident could currently be changed back to open. A production version would define and validate allowed state transitions.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/incident` | Retrieve context, analyze and store an incident |
| `GET` | `/incidents` | List/filter incidents |
| `GET` | `/incidents/{incident_id}` | Retrieve one incident |
| `PATCH` | `/incidents/{incident_id}/status` | Update incident lifecycle status |
| `GET` | `/stats` | Incident statistics |
| `GET` | `/health` | Service health check |
| `GET` | `/knowledge` | Inspect the prototype knowledge base |

FastAPI automatically exposes interactive API documentation through Swagger.

## Reliability and Observability

The application logs important stages of incident processing:

```text
Incident received
Historical knowledge retrieved
AI analysis requested
AI request retry
AI provider unavailable
Incident saved
Incident status updated
```

During testing, Gemini returned temporary `503 Service Unavailable` responses. These were useful for validating that retries and graceful degradation work as expected.

Incident ingestion and persistence continue even when AI generation is temporarily unavailable.

## Evaluation

The current version does not yet have a formal LLM evaluation suite.

Before treating the generated root-cause suggestions as reliable, the next step is to build a small labeled incident dataset containing:

```text
incident
expected failure category
expected investigation area
historical context
```

The system can then measure whether retrieval selects the correct historical context and whether the generated analysis identifies the expected investigation area.

This is intentionally listed as unfinished rather than claiming accuracy based on a small number of manual examples.

## Current Limitations

### Retrieval

Keyword matching does not capture semantic similarity and returns only the first matching historical incident.

### Storage

Incidents are stored in a JSON file.

This is appropriate for a local prototype but is not safe for concurrent production writes and does not provide transactions, indexing, or scalable querying.

### AI dependency

The application depends on an external LLM provider. Retry and fallback behavior reduce the impact of temporary failures, but the AI analysis itself is unavailable during an extended provider outage.

### Fallback provenance

The current response structure does not explicitly record whether `ai_analysis` came from Gemini or from fallback guidance.

This is an important limitation because downstream users should be able to distinguish model-generated analysis from predefined fallback advice.

Adding an `analysis_source` field such as:

```text
gemini
fallback
```

is planned for the next iteration.

## Next Steps

The immediate priorities are deliberately small:

**1. Automated tests and evaluation**

Add API tests, retrieval tests, failure-path tests, and a labeled incident evaluation set.

**2. Semantic retrieval with pgvector**

Replace keyword matching with embeddings so operationally similar incidents can match even when they use different terminology.

For example:

```text
DB pool exhausted
```

should be able to match:

```text
database connection saturation
```

without depending on shared keywords.

**3. Alertmanager webhook ingestion**

Replace manual incident submission with alerts received from a monitoring system, allowing the Copilot to analyze real alert payloads automatically.

Additional infrastructure such as Kafka, Redis, Slack integration, and Kubernetes would only be introduced when the workload or architecture provides a concrete reason for them.

## Project Structure

```text
ai-incident-copilot/
│
├── data/
│   └── incidents.json
├── knowledge/
│   └── incidents.txt
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Secrets such as the Gemini API key are loaded from `.env`. The `.env` file should never be committed to the repository.