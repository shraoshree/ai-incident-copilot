# Agent investigation, vector retrieval, evaluations and deployment

## Status
The agent workflow has a passing live Gemini function-calling test. PostgreSQL/pgvector, container packaging, and AWS Fargate configuration are implemented as **opt-in development assets**; no hosted AWS deployment or live database integration test is claimed. The original JSON incident storage remains in use. The vector database stores knowledge chunks only, not incident records.

## Local development
```bash
python -m pip install -r requirements.txt
python -m pytest -q -m "not live"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Use `POST /incident/agent` for bounded, read-only function calling. Set `GEMINI_API_KEY` in your local environment (never commit it). The agent defaults to `gemini-3.8-flash` when `GEMINI_MODEL` is not supplied. The legacy `POST /incident` endpoint is preserved.

## Vector search (optional)
Start PostgreSQL with pgvector enabled. Set `DATABASE_URL` to a valid PostgreSQL connection string and `GEMINI_API_KEY` for embeddings. Run:
```bash
python scripts/index_knowledge.py
```
This installs the `vector` extension (database privileges required), creates a 768-dimensional knowledge table, and embeds historical knowledge. When `DATABASE_URL` is configured, agent knowledge search uses cosine similarity with top-3 results. If the vector service is unavailable, the agent falls back to local keyword matching. Without `DATABASE_URL`, the application does not contact PostgreSQL. Embedding requests may incur charges.

**Security:** Use TLS for production database connections, restrict database access, and do not embed private production incident data without permission and appropriate handling. For large collections, add a pgvector HNSW index and benchmark retrieval latency.

## Evaluations
```bash
python evals/evaluate.py
python -m pytest -q -m "not live"
```
The deterministic retrieval fixture measures hit@3 on a small sample and is **not** a statistically meaningful quality benchmark. Expand the dataset with labeled incidents, hard negatives, and synonyms before claiming retrieval accuracy. The live Gemini test is separate and manually triggered; it consumes API quota.

## Container
```bash
docker build -t incident-copilot .
docker run --rm -p 8000:8000 --env-file .env incident-copilot
```
The local JSON incident store inside this container is **ephemeral**. Do not use this configuration as durable production incident storage. Use a persistent data store before production deployment.

## AWS deployment template
`deploy/ecs-task-definition.json` is a starting point for ECS Fargate. Replace placeholders with your own ECR image URI, IAM role ARNs, Secrets Manager ARN, and CloudWatch Logs region. Create an ECR repository, build/push the image, provision an ECS cluster/service, VPC networking and security groups, a log group, health checks, and a secure public entry point if needed. Restrict IAM permissions, configure secret access, and budget for AWS resources. **No AWS resources are provisioned by this repository.** A production launch additionally requires persistent incident storage, operational alarms, authentication, and load testing.
