from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., example="Как восстановить пароль?")
    k: int = Field(default=5, ge=1, le=20, example=5)


class Source(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
