import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import errors
from pydantic import BaseModel, ValidationError
from agent_tools import investigate_with_tools


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

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INCIDENTS_FILE = "data/incidents.json"


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
# Knowledge Retrieval
# --------------------------------------------------

def load_knowledge():
    with open("knowledge/incidents.txt", "r") as file:
        return file.read()


def find_relevant_knowledge(incident: Incident):
    knowledge = load_knowledge()

    incidents = knowledge.split("\n\n")

    search_text = (
        incident.service + " " +
        incident.alert + " " +
        incident.description
    ).lower()

    for past_incident in incidents:
        words = search_text.split()

        for word in words:
            if len(word) > 4 and word in past_incident.lower():

                logger.info(
                    "Relevant historical knowledge found for service=%s",
                    incident.service
                )

                return past_incident

    logger.warning(
        "No relevant historical knowledge found for service=%s",
        incident.service
    )

    return "No relevant past incident found."


# --------------------------------------------------
# AI Analysis
# --------------------------------------------------

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

            fallback_analysis = IncidentAnalysis(
                likely_cause="Unable to parse the AI response.",
                investigation_steps=[
                    "Review the retrieved historical incident.",
                    "Check application logs and monitoring metrics.",
                    "Review recent deployments and infrastructure changes."
                ],
                immediate_action=(
                    "Continue manual investigation using the retrieved "
                    "historical knowledge."
                )
            )

            return fallback_analysis, "fallback"

        except errors.ServerError as error:

            logger.warning(
                "Gemini unavailable attempt=%s error=%s",
                attempt + 1,
                error
            )

            if attempt < max_attempts - 1:
                time.sleep(3)

            else:

                logger.error(
                    "AI analysis unavailable after %s attempts",
                    max_attempts
                )

                fallback_analysis = IncidentAnalysis(
                    likely_cause=(
                        "AI analysis is temporarily unavailable because "
                        "the Gemini service is experiencing high demand."
                    ),
                    investigation_steps=[
                        "Review the retrieved historical incident.",
                        "Check application logs and monitoring metrics.",
                        "Review recent deployments and infrastructure changes."
                    ],
                    immediate_action=(
                        "Continue investigation using the retrieved knowledge "
                        "and retry AI analysis shortly."
                    )
                )

                return fallback_analysis, "fallback"


# --------------------------------------------------
# Incident Storage
# --------------------------------------------------

def load_incidents():
    with open(INCIDENTS_FILE, "r") as file:
        return json.load(file)


def save_incidents(incidents):
    with open(INCIDENTS_FILE, "w") as file:
        json.dump(incidents, file, indent=4)


def generate_incident_id():
    return f"INC-{uuid.uuid4().hex[:8]}"


def save_incident(
    incident: Incident,
    relevant_knowledge: str,
    ai_analysis: IncidentAnalysis,
    analysis_source: str
):
    incidents = load_incidents()

    incident_record = {
        "incident_id": generate_incident_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": incident.service,
        "alert": incident.alert,
        "severity": incident.severity,
        "description": incident.description,
        "status": "open",
        "relevant_knowledge": relevant_knowledge,
        "analysis_source": analysis_source,
        "ai_analysis": ai_analysis.model_dump()
    }

    incidents.append(incident_record)

    save_incidents(incidents)

    logger.info(
        "Incident saved incident_id=%s service=%s severity=%s "
        "analysis_source=%s",
        incident_record["incident_id"],
        incident.service,
        incident.severity,
        analysis_source
    )

    return incident_record


def find_incident_by_id(incident_id: str):
    incidents = load_incidents()

    for incident in incidents:
        if incident.get("incident_id") == incident_id:
            return incident

    return None


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

    relevant_knowledge = find_relevant_knowledge(incident)

    ai_analysis, analysis_source = analyze_incident(
        incident,
        relevant_knowledge
    )

    saved_incident = save_incident(
        incident,
        relevant_knowledge,
        ai_analysis,
        analysis_source
    )

    return {
        "message": "Incident analyzed and saved",
        "incident": saved_incident
    }



@app.post("/incident/agent")
def receive_agent_incident(incident: Incident):
    """Investigate with bounded read-only tools, then persist validated guidance."""
    relevant_knowledge = find_relevant_knowledge(incident)
    tool_trace = []
    try:
        result = investigate_with_tools(
            client=client,
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            incident=incident.model_dump(),
            knowledge_text=load_knowledge(),
        )
        tool_trace = result["tool_trace"]
        raw = result["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        analysis = IncidentAnalysis.model_validate_json(raw)
        source = "gemini_agent"
    except (ValueError, RuntimeError, ValidationError, errors.APIError) as exc:
        logger.warning("Agent investigation fallback: %s", type(exc).__name__)
        analysis = IncidentAnalysis(
            likely_cause="Root cause not confirmed; agent analysis unavailable.",
            investigation_steps=[
                "Review historical incident knowledge.",
                "Inspect application logs and monitoring metrics.",
                "Check recent deployments and infrastructure changes.",
            ],
            immediate_action="Continue manual investigation; do not perform automated remediation.",
        )
        source = "fallback"

    saved = save_incident(incident, relevant_knowledge, analysis, source)
    return {
        "message": "Incident investigated and saved",
        "incident": saved,
        "tool_trace": tool_trace,
    }


@app.get("/incidents")
def get_incidents(status: str | None = None):

    incidents = load_incidents()

    if status:
        incidents = [
            incident
            for incident in incidents
            if incident.get("status") == status
        ]

    return {
        "count": len(incidents),
        "incidents": incidents
    }


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):

    incident = find_incident_by_id(incident_id)

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

    incidents = load_incidents()

    for incident in incidents:
        if incident.get("incident_id") == incident_id:

            current_status = incident.get("status", "open")
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

            incident["status"] = new_status

            save_incidents(incidents)

            logger.info(
                "Incident status updated incident_id=%s %s -> %s",
                incident_id,
                current_status,
                new_status
            )

            return {
                "message": "Incident status updated",
                "incident": incident
            }

    raise HTTPException(
        status_code=404,
        detail="Incident not found"
    )


@app.get("/stats")
def get_incident_stats():

    incidents = load_incidents()

    stats = {
        "total_incidents": len(incidents),
        "open": 0,
        "investigating": 0,
        "resolved": 0,
        "critical": 0
    }

    for incident in incidents:

        status = incident.get("status")

        if status in stats:
            stats[status] += 1

        if incident.get("severity", "").lower() == "critical":
            stats["critical"] += 1

    return stats


@app.get("/knowledge")
def get_knowledge():
    return {
        "knowledge": load_knowledge()
    }