"""
FastAPI service for "The Tank" — a Shark-Tank-style multi-agent pitch
review bot.

Run with:
    uvicorn main:app --reload

Requires ANTHROPIC_API_KEY to be set in the environment (see .env.example).
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from graph import PERSONAS, run_tank, iter_tank

app = FastAPI(title="The Tank", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdviseRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The pitch/plan/idea to put in the tank")
    personas: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of shark keys to consult (defaults to all five). "
        "See GET /personas for valid keys.",
    )


class AdviseResponse(BaseModel):
    topic: str
    feedback: dict


@app.get("/personas")
def list_personas():
    """Return the available sharks and their descriptions."""
    return {
        key: {
            "display_name": p["display_name"],
            "system_prompt": p["system_prompt"],
            "species": p.get("species"),
            "photo_url": p.get("photo_url"),
            "credit_url": p.get("credit_url"),
        }
        for key, p in PERSONAS.items()
    }


def _validate_request(req: AdviseRequest):
    if req.personas:
        unknown = set(req.personas) - set(PERSONAS.keys())
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown persona key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(PERSONAS.keys())}",
            )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set on the server. "
            "Copy .env.example to .env and add your key.",
        )


@app.post("/advise", response_model=AdviseResponse)
def advise(req: AdviseRequest):
    """Blocking variant: runs every shark sequentially server-side and
    returns the full set at once. Prefer /advise/stream for the UI's
    one-shark-at-a-time reveal — this exists for simple API callers."""
    _validate_request(req)
    try:
        feedback = run_tank(req.topic, req.personas)
    except Exception as e:  # surface a clean error to the frontend
        raise HTTPException(status_code=500, detail=str(e))

    return AdviseResponse(topic=req.topic, feedback=feedback)


@app.post("/advise/stream")
def advise_stream(req: AdviseRequest):
    """Streams newline-delimited JSON events as each shark finishes, in
    the fixed nicest-to-meanest order — one LLM call at a time. This is
    what the UI consumes for the sequential "shark enters the tank" reveal.

    Each line is one JSON object:
      {"type": "start",  "key", "display_name", "index", "total"}
      {"type": "result", "key", "display_name", "index", "total", "text"}
      {"type": "error",  "key", "display_name", "index", "total", "error"}
      {"type": "done"}                                   (always last)
    A single shark erroring doesn't kill the stream — it's reported as an
    "error" event and the next shark still runs.
    """
    _validate_request(req)

    def event_stream():
        try:
            for event in iter_tank(req.topic, req.personas):
                yield json.dumps(event) + "\n"
        except Exception as e:
            # only reachable for setup errors before the per-shark
            # try/except in iter_tank takes over (e.g. bad persona keys
            # slipping past validation) — per-shark failures are already
            # "error" events, not exceptions.
            yield json.dumps({"type": "fatal", "error": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Serve the simple frontend at "/"
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
def favicon():
    # Avoids a noisy 404 in server logs / browser console — browsers
    # request this automatically even though nothing in the app links it.
    return Response(status_code=204)
