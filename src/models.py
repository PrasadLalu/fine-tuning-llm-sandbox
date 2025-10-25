from typing import Optional
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    model: str
    query: str
    response: str
    tokens_generated: int
