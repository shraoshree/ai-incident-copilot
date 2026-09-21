import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import errors, types
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from pydantic import BaseModel


load_dotenv()


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("incident-copilot")


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=30000
    )
)

DATABASE_URL = os.getenv("DATABASE_URL")


# --------------------------------------------------
# Models
# --------------------------------------------------

class Incident(BaseModel):
    service: str
    alert: str
    severity: str
    description: str


class IncidentAnalysis(BaseModel):
    likely_cause: str
    investigation_steps: list[str]
    immediate_action: str


class StatusUpdate(BaseModel):
    status: str


# --------------------------------------------------
# Semantic Knowledge Retrieval
# --------------------------------------------------

def normalize_embedding(values):
    magnitude = math.sqrt(
        sum(value * value for value in values)
    )

    if magnitude == 0:
        return values

    return [
        value / magnitude
        for value in values
    ]


def create_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": 768,
        },
    )

    embedding = response.embeddings[0].values

    return normalize_embedding(embedding)


def find_relevant_knowledge(incident: Incident):
    search_text = (
        f"Service: {incident.service}\n"
        f"Alert: {incident.alert}\n"
        f"Description: {incident.description}"
    )

    try:
        query_embedding = Vector(
            create_embedding(search_text)
        )

        with psycopg.connect(DATABASE_URL) as connection:
            register_vector(connection)

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT content,
                           embedding <=> %s AS distance
                    FROM historical_incidents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT 1;
                    """,
                    (
                        query_embedding,
                        query_embedding
                    ),
                )

                result = cursor.fetchone()

        if result is None:
            logger.warning(
                "No historical knowledge found for service=%s",
                incident.service
            )

            return "No relevant past incident found."

        content, distance = result

        logger.info(
            "Semantic knowledge retrieved service=%s distance=%s",
            incident.service,
            distance
        )

        return content

    except Exception as error:
        logger.error(
            "Semantic retrieval failed service=%s error=%s",
            incident.service,
            error
        )

        return "No relevant past incident found."


# --------------------------------------------------
# AI Analysis
# --------------------------------------------------

def build_fallback_analysis(reason: str):
    return IncidentAnalysis(
        likely_cause=reason,
        investigation_steps=[
            "Review the retrieved historical incident.",
            "Check application logs and monitoring metrics.",
            "Review recent deployments and infrastructure changes."
        ],
        immediate_action=(
            "Continue investigation using the retrieved historical "
            "knowledge and retry AI analysis later."
        )
    )


def analyze_incident(
    incident: Incident,
    relevant_knowledge: str
):
    prompt = f"""
You are an AI incident analysis copilot helping a software engineer
investigate a production incident.

CURRENT INCIDENT:
Service: {incident.service}
Alert: {incident.alert}
Severity: {incident.severity}
Description: {incident.description}

RELEVANT HISTORICAL KNOWLEDGE:
{relevant_knowledge}

Analyze the incident using the current incident and the historical
knowledge provided.

Return:
- the most likely cause
- practical investigation steps
- the recommended immediate action

Keep the analysis concise and useful for an on-call engineer.
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            logger.info(
                "Requesting AI analysis attempt=%s service=%s",
                attempt + 1,
                incident.service
            )

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": IncidentAnalysis,
                },
            )

            if response.parsed:
                logger.info(
                    "AI analysis completed for service=%s",
                    incident.service
                )

                return response.parsed, "gemini"

            logger.warning(
                "AI response could not be parsed for service=%s",
                incident.service
            )

            return build_fallback_analysis(
                "Unable to parse the AI response."
            ), "fallback"

        except errors.ServerError as error:
            logger.warning(
                "Gemini server unavailable attempt=%s error=%s",
                attempt + 1,
                error
            )

            if attempt < max_attempts - 1:
                time.sleep(3)

            else:
                return build_fallback_analysis(
                    "AI analysis is temporarily unavailable because "
                    "the Gemini service is experiencing high demand."
                ), "fallback"

        except errors.ClientError as error:
            logger.warning(
                "Gemini request rejected service=%s error=%s",
                incident.service,
                error
            )

            return build_fallback_analysis(
                "AI analysis is temporarily unavailable because "
                "the Gemini API request limit or quota was reached."
            ), "fallback"

        except Exception:
            logger.exception(
                "Unexpected AI analysis failure service=%s",
                incident.service
            )

            return build_fallback_analysis(
                "AI analysis is temporarily unavailable due to "
                "an unexpected service error."
            ), "fallback"


# --------------------------------------------------
# PostgreSQL Incident Storage
# --------------------------------------------------

def generate_incident_id():
    return f"INC-{uuid.uuid4().hex[:8]}"


def row_to_incident(row):
    if row is None:
        return None

    return {
        "incident_id": row[0],
        "timestamp": row[1].isoformat(),
        "service": row[2],
        "alert": row[3],
        "severity": row[4],
        "description": row[5],
        "status": row[6],
        "relevant_knowledge": row[7],
        "analysis_source": row[8],
        "ai_analysis": row[9],
    }


def save_incident(
    incident: Incident,
    relevant_knowledge: str,
    ai_analysis: IncidentAnalysis,
    analysis_source: str
):
    incident_id = generate_incident_id()
    timestamp = datetime.now(timezone.utc)

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO incidents (
                    incident_id,
                    timestamp,
                    service,
                    alert,
                    severity,
                    description,
                    status,
                    relevant_knowledge,
                    analysis_source,
                    ai_analysis
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING
                    incident_id,
                    timestamp,
                    service,
                    alert,
                    severity,
                    description,
                    status,
                    relevant_knowledge,
                    analysis_source,
                    ai_analysis;
                """,
                (
                    incident_id,
                    timestamp,
                    incident.service,
                    incident.alert,
                    incident.severity,
                    incident.description,
                    "open",
                    relevant_knowledge,
                    analysis_source,
                    Jsonb(ai_analysis.model_dump()),
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    logger.info(
        "Incident saved incident_id=%s service=%s severity=%s "
        "analysis_source=%s",
        incident_id,
        incident.service,
        incident.severity,
        analysis_source
    )

    return row_to_incident(row)


def load_incidents(status: str | None = None):
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:

            if status:
                cursor.execute(
                    """
                    SELECT
                        incident_id,
                        timestamp,
                        service,
                        alert,
                        severity,
                        description,
                        status,
                        relevant_knowledge,
                        analysis_source,
                        ai_analysis
                    FROM incidents
                    WHERE status = %s
                    ORDER BY timestamp DESC;
                    """,
                    (status,),
                )

            else:
                cursor.execute(
                    """
                    SELECT
                        incident_id,
                        timestamp,
                        service,
                        alert,
                        severity,
                        description,
                        status,
                        relevant_knowledge,
                        analysis_source,
                        ai_analysis
                    FROM incidents
                    ORDER BY timestamp DESC;
                    """
                )

            rows = cursor.fetchall()

    return [
        row_to_incident(row)
        for row in rows
    ]


def find_incident_by_id(incident_id: str):
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    incident_id,
                    timestamp,
                    service,
                    alert,
                    severity,
                    description,
                    status,
                    relevant_knowledge,
                    analysis_source,
                    ai_analysis
                FROM incidents
                WHERE incident_id = %s;
                """,
                (incident_id,),
            )

            row = cursor.fetchone()

    return row_to_incident(row)


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Incident Copilot is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Incident Copilot"
    }


@app.post("/incident")
def receive_incident(incident: Incident):
    logger.info(
        "Incident received service=%s alert=%s severity=%s",
        incident.service,
        incident.alert,
        incident.severity
    )

    relevant_knowledge = find_relevant_knowledge(
        incident
    )

    ai_analysis, analysis_source = analyze_incident(
        incident,
        relevant_knowledge
    )

    try:
        saved_incident = save_incident(
            incident,
            relevant_knowledge,
            ai_analysis,
            analysis_source
        )

    except Exception:
        logger.exception(
            "Failed to save incident service=%s",
            incident.service
        )

        raise HTTPException(
            status_code=503,
            detail="Incident storage is temporarily unavailable"
        )

    return {
        "message": "Incident analyzed and saved",
        "incident": saved_incident
    }


@app.get("/incidents")
def get_incidents(status: str | None = None):
    try:
        incidents = load_incidents(status)

    except Exception:
        logger.exception(
            "Failed to load incidents"
        )

        raise HTTPException(
            status_code=503,
            detail="Incident storage is temporarily unavailable"
        )

    return {
        "count": len(incidents),
        "incidents": incidents
    }


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    try:
        incident = find_incident_by_id(
            incident_id
        )

    except Exception:
        logger.exception(
            "Failed to retrieve incident incident_id=%s",
            incident_id
        )

        raise HTTPException(
            status_code=503,
            detail="Incident storage is temporarily unavailable"
        )

    if incident is None:
        logger.warning(
            "Incident not found incident_id=%s",
            incident_id
        )

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )

    return {
        "incident": incident
    }


@app.patch("/incidents/{incident_id}/status")
def update_incident_status(
    incident_id: str,
    status_update: StatusUpdate
):
    allowed_statuses = [
        "open",
        "investigating",
        "resolved"
    ]

    if status_update.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. Use open, investigating, "
                "or resolved."
            )
        )

    allowed_transitions = {
        "open": ["investigating", "resolved"],
        "investigating": ["resolved"],
        "resolved": []
    }

    try:
        with psycopg.connect(DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status
                    FROM incidents
                    WHERE incident_id = %s
                    FOR UPDATE;
                    """,
                    (incident_id,),
                )

                row = cursor.fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Incident not found"
                    )

                current_status = row[0]
                new_status = status_update.status

                if new_status == current_status:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Incident is already in status "
                            f"'{current_status}'."
                        )
                    )

                if new_status not in allowed_transitions[current_status]:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid status transition: "
                            f"{current_status} -> {new_status}"
                        )
                    )

                cursor.execute(
                    """
                    UPDATE incidents
                    SET status = %s
                    WHERE incident_id = %s
                    RETURNING
                        incident_id,
                        timestamp,
                        service,
                        alert,
                        severity,
                        description,
                        status,
                        relevant_knowledge,
                        analysis_source,
                        ai_analysis;
                    """,
                    (
                        new_status,
                        incident_id
                    ),
                )

                updated_row = cursor.fetchone()

            connection.commit()

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to update incident status incident_id=%s",
            incident_id
        )

        raise HTTPException(
            status_code=503,
            detail="Incident storage is temporarily unavailable"
        )

    logger.info(
        "Incident status updated incident_id=%s %s -> %s",
        incident_id,
        current_status,
        new_status
    )

    return {
        "message": "Incident status updated",
        "incident": row_to_incident(updated_row)
    }


@app.get("/stats")
def get_incident_stats():
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_incidents,
                        COUNT(*) FILTER (
                            WHERE status = 'open'
                        ) AS open_count,
                        COUNT(*) FILTER (
                            WHERE status = 'investigating'
                        ) AS investigating_count,
                        COUNT(*) FILTER (
                            WHERE status = 'resolved'
                        ) AS resolved_count,
                        COUNT(*) FILTER (
                            WHERE LOWER(severity) = 'critical'
                        ) AS critical_count
                    FROM incidents;
                    """
                )

                row = cursor.fetchone()

        return {
            "total_incidents": row[0],
            "open": row[1],
            "investigating": row[2],
            "resolved": row[3],
            "critical": row[4],
        }

    except Exception:
        logger.exception(
            "Failed to calculate incident statistics"
        )

        raise HTTPException(
            status_code=503,
            detail="Incident storage is temporarily unavailable"
        )


@app.get("/knowledge")
def get_knowledge():
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, content
                    FROM historical_incidents
                    ORDER BY id;
                    """
                )

                rows = cursor.fetchall()

        return {
            "count": len(rows),
            "knowledge": [
                {
                    "id": row[0],
                    "content": row[1]
                }
                for row in rows
            ]
        }

    except Exception as error:
        logger.error(
            "Failed to load historical knowledge error=%s",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load historical knowledge"
        )