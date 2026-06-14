import uuid
from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    question_title: str
    question_id: int
    score: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str

    @field_validator("session_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")
        return v


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str
