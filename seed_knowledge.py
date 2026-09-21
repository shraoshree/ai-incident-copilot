import math
import os

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector import Vector
from pgvector.psycopg import register_vector


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


def load_historical_incidents():
    with open("knowledge/incidents.txt", "r") as file:
        knowledge = file.read()

    return [
        incident.strip()
        for incident in knowledge.split("\n\n")
        if incident.strip()
    ]


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


def create_document_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 768,
        },
    )

    embedding = response.embeddings[0].values

    return normalize_embedding(embedding)


def seed_database():
    incidents = load_historical_incidents()

    with psycopg.connect(DATABASE_URL) as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM historical_incidents;"
            )

            for incident in incidents:
                embedding = Vector(
                    create_document_embedding(incident)
                )

                cursor.execute(
                    """
                    INSERT INTO historical_incidents (
                        content,
                        embedding
                    )
                    VALUES (%s, %s)
                    """,
                    (
                        incident,
                        embedding
                    ),
                )

                print(
                    "Stored:",
                    incident.split("\n")[0]
                )

        connection.commit()

    print(
        "Knowledge base seeded successfully "
        "with 768-dimensional embeddings!"
    )


if __name__ == "__main__":
    seed_database()