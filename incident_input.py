"""Shared API incident validation and redaction."""
from pydantic import BaseModel, Field

from security import redact_sensitive


class Incident(BaseModel):
    service: str = Field(min_length=1, max_length=120)
    alert: str = Field(min_length=1, max_length=300)
    severity: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=10000)

def prepare_incident(incident: Incident) -> Incident:
    """Redact after validation, then revalidate exactly as the API does."""
    return Incident(**{
        name: redact_sensitive(getattr(incident, name))
        for name in ("service", "alert", "severity", "description")
    })
