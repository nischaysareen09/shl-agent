"""
Step 4: FastAPI service.

Exposes exactly the two endpoints the assignment's automated evaluator expects:
  GET  /health -> {"status": "ok"}
  POST /chat   -> {"reply": str, "recommendations": [...] | null, "end_of_conversation": bool}

This file is intentionally thin -- all actual decision-making lives in app/agent.py,
which is unit-tested independently of the HTTP layer. main.py's only jobs are:
  1. Validate the request/response shape with pydantic (schema compliance is a
     hard-eval; a malformed response fails the whole submission).
  2. Load the Agent (and its Retriever/embedding model) ONCE at startup, not per
     request -- important given the 30s per-call timeout the evaluator enforces.
  3. Never let an unhandled exception produce a non-JSON or 500 response with the
     wrong shape -- always fall back to a schema-valid error reply instead.
"""
from __future__ import annotations
from typing import List, Optional, Literal

import app.env  # noqa: F401 -- loads .env before Agent reads GROQ_API_KEY

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import Agent

app = FastAPI(title="SHL Assessment Recommender")

# Open CORS since the evaluator harness calls this from an external server, not a browser --
# but harmless either way and saves debugging a CORS failure at submission time.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at process startup (includes building the embedding index over the
# catalog) so individual /chat calls don't pay that cost -- critical for the 30s
# per-call budget and the "up to 2 minutes" first-health-check cold-start allowance.
agent = Agent()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    end_of_conversation: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if not messages:
        return ChatResponse(
            reply="Hi! Tell me about the role or skills you're hiring for and I'll suggest SHL assessments.",
            recommendations=None,
            end_of_conversation=False,
        )

    try:
        result = agent.handle_turn(messages)
    except Exception:
        # Absolute last resort: the evaluator scores schema compliance on EVERY
        # response, so even a total internal crash must still return valid shape.
        return ChatResponse(
            reply="Sorry, something went wrong on my end. Could you try again?",
            recommendations=None,
            end_of_conversation=False,
        )

    return ChatResponse(
        reply=result["reply"],
        recommendations=result.get("recommendations"),
        end_of_conversation=result.get("end_of_conversation", False),
    )
