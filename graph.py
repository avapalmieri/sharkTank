"""
LangGraph / LangChain orchestration for "The Tank" — a Shark-Tank-style
pitch review bot.

Five shark personas review the same pitch independently — they do not see
each other's answers and there is no synthesis step. Each one evaluates
from a different angle (people, brand, scale, product instinct, numbers)
and ends with their own verdict. The point is deliberately unresolved,
sometimes contradictory feedback that the human has to weigh themselves.

Two ways to run it:
  - `iter_tank()` — the default. Runs sharks ONE AT A TIME, in a fixed
    nicest-to-meanest order, yielding an event as each one starts and
    finishes. This is what powers the sequential reveal in the UI (the
    "one shark at a time" experience) — total latency is roughly 5x a
    single call since nothing runs concurrently, which is the deliberate
    trade for a real, non-faked progressive reveal.
  - `build_graph()` / the old parallel path — still here if you want low
    latency instead of drama: all five sharks fan out from LangGraph's
    START node and run concurrently, same personas, same prompts.
"""

import os
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# --- Shark definitions ---------------------------------------------------
# Fixed order: NICEST FIRST, MEANEST LAST. This order is load-bearing —
# it drives both the sequential reveal (who goes 1st vs 5th) and the tone
# gradient in each system prompt below. Reorder with intent; don't shuffle
# casually. These are original fictional personas (not based on any real
# person) — archetypes of what a tough investor panel tends to probe for,
# with an added "how do they deliver the news" dial from warm to savage.

VERDICT_INSTRUCTION = (
    " End with a single verdict line on its own, starting with 'VERDICT:' "
    "followed by exactly one of: \"I'm in.\" / \"I'm out.\" / \"I'm in, "
    "but only if...\" (fill in your real condition)."
)

# The tank is scoped to business and financial decisions only — pricing,
# spending, hiring, fundraising, market entry, investments, and similar.
# Prepended to every persona so a shark redirects (briefly, still in
# character) rather than earnestly reviewing an unrelated life decision.
SCOPE_INSTRUCTION = (
    "This tank only evaluates BUSINESS AND FINANCIAL decisions — pricing, "
    "spending, hiring, fundraising, market entry, investments, and the "
    "like. If what you're handed isn't one of those, say so briefly, in "
    "character, and decline to give a real verdict rather than forcing a "
    "business lens onto something personal. "
)

PERSONAS = {
    "people_shark": {  # 1. nicest
        "display_name": "The People Shark",
        "species": "Whale Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Whale_shark_Maldives.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Whale_shark_Maldives.jpg",
        "system_prompt": SCOPE_INSTRUCTION + (
            "You are the People Shark in the tank — the closest thing to a "
            "warm mentor at this table. You believe in people first, ideas "
            "second, and you genuinely want to see the person in front of "
            "you succeed. You evaluate the plan through what it demands of "
            "the person running it: skills, grit, time, network. Be warm "
            "and encouraging in tone, but still honest — name the real "
            "execution risk you see, delivered like a mentor's note, not a "
            "critic's jab. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "brand_shark": {  # 2.
        "display_name": "The Brand Shark",
        "species": "Nurse Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Nurse_Shark_4472.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Nurse_Shark_4472.jpg",
        "system_prompt": SCOPE_INSTRUCTION + (
            "You are the Brand Shark in the tank — friendly, curious, "
            "coach-like. You care about the story, not just the "
            "spreadsheet: is this memorable, ownable, worth talking about? "
            "Deliver your notes the way a good creative director would — "
            "constructive, a little playful, genuinely rooting for a "
            "better version of this. Name the single biggest branding or "
            "positioning weakness, and the one change that would make this "
            "pitch stick in someone's head. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "scale_shark": {  # 3. — the tonal midpoint
        "display_name": "The Scale Shark",
        "species": "Shortfin Mako Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Shortfin_mako.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Shortfin_mako.jpg",
        "system_prompt": SCOPE_INSTRUCTION + (
            "You are the Scale Shark in the tank — direct, matter-of-fact, "
            "no wasted words, but fair. You're not here to make friends or "
            "make enemies, just to size up whether this can get big. Point "
            "out the one thing standing between this idea and 10x growth, "
            "plainly, without softening it and without any personal edge. "
            "3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "product_shark": {  # 4.
        "display_name": "The Product Shark",
        "species": "Bull Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bullshark_Beqa_Fiji_2007.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Bullshark_Beqa_Fiji_2007.jpg",
        "system_prompt": SCOPE_INSTRUCTION + (
            "You are the Product Shark in the tank — blunt, and visibly "
            "unimpressed until proven otherwise. You've seen a thousand "
            "pitches and have no patience for ones that sound good but "
            "haven't been tested on a real customer. Call out the gap "
            "between the pitch and what a real customer would actually do "
            "— tersely, with little warmth. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "numbers_shark": {  # 5. meanest
        "display_name": "The Numbers Shark",
        "species": "Great White Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Guadalupe_Island_Great_White_Shark_Face_On.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Guadalupe_Island_Great_White_Shark_Face_On.jpg",
        "system_prompt": SCOPE_INSTRUCTION + (
            "You are the Numbers Shark in the tank — the harshest voice at "
            "the table, and you know it. You have zero patience for anyone "
            "who hasn't done their homework on unit economics, margins, or "
            "valuation, and you say so with a sharp, cutting one-liner. "
            "Find the weakest financial assumption in the pitch and go "
            "after it directly — biting, impatient, borderline dismissive. "
            "Stay ruthless about the MATH, never personal or demeaning "
            "about who they are. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
}


def _llm():
    return ChatAnthropic(model=MODEL_NAME, temperature=0.7, max_tokens=400)


def _resolve_keys(persona_keys):
    persona_keys = persona_keys or list(PERSONAS.keys())
    unknown = set(persona_keys) - set(PERSONAS.keys())
    if unknown:
        raise ValueError(f"Unknown persona(s): {unknown}")
    # Always walk PERSONAS' own nicest->meanest order, regardless of the
    # order the caller's subset arrived in — the reveal order is fixed.
    return [k for k in PERSONAS if k in set(persona_keys)]


def iter_tank(topic: str, persona_keys=None):
    """Yield sequential events as each shark reviews the pitch, nicest to
    meanest. One LLM call at a time — this is what makes the one-shark-
    at-a-time reveal real rather than a client-side animation over data
    that was already all there.

    Yields dicts:
      {"type": "start",  "key", "display_name", "index", "total"}
      {"type": "result", "key", "display_name", "index", "total", "text"}
      {"type": "error",  "key", "display_name", "index", "total", "error"}
    """
    keys = _resolve_keys(persona_keys)
    total = len(keys)
    for i, key in enumerate(keys):
        persona = PERSONAS[key]
        yield {
            "type": "start",
            "key": key,
            "display_name": persona["display_name"],
            "index": i,
            "total": total,
        }
        try:
            messages = [
                SystemMessage(content=persona["system_prompt"]),
                HumanMessage(content=topic),
            ]
            response = _llm().invoke(messages)
            yield {
                "type": "result",
                "key": key,
                "display_name": persona["display_name"],
                "index": i,
                "total": total,
                "text": response.content,
            }
        except Exception as e:
            yield {
                "type": "error",
                "key": key,
                "display_name": persona["display_name"],
                "index": i,
                "total": total,
                "error": str(e),
            }


def run_tank(topic: str, persona_keys=None) -> dict:
    """Blocking helper for non-streaming callers: run the whole tank
    sequentially and return {display_name: feedback}. Errors on individual
    sharks are dropped rather than raised, matching the streaming path's
    resilience — a single shark failing shouldn't blank the whole response."""
    feedback = {}
    for event in iter_tank(topic, persona_keys):
        if event["type"] == "result":
            feedback[event["display_name"]] = event["text"]
    return feedback


# --- Parallel path (optional, faster, no drama) --------------------------
# Kept for anyone who wants low latency over a sequential reveal — same
# personas, same prompts, all five fan out from START and run concurrently.


class TankState(TypedDict):
    topic: str
    feedback: Annotated[dict, operator.or_]


def _make_persona_node(persona_key: str):
    persona = PERSONAS[persona_key]
    llm = _llm()

    def node(state: TankState) -> dict:
        messages = [
            SystemMessage(content=persona["system_prompt"]),
            HumanMessage(content=state["topic"]),
        ]
        response = llm.invoke(messages)
        return {"feedback": {persona_key: response.content}}

    return node


def build_graph(persona_keys=None):
    """Build (and compile) the tank's PARALLEL review graph."""
    keys = _resolve_keys(persona_keys)
    workflow = StateGraph(TankState)
    for key in keys:
        workflow.add_node(key, _make_persona_node(key))
        workflow.add_edge(START, key)
        workflow.add_edge(key, END)
    return workflow.compile()


def run_tank_parallel(topic: str, persona_keys=None) -> dict:
    """Run all sharks concurrently via LangGraph and return
    {display_name: feedback}. Faster than run_tank(), no sequential drama."""
    graph = build_graph(persona_keys)
    result = graph.invoke({"topic": topic, "feedback": {}})
    return {
        PERSONAS[key]["display_name"]: text
        for key, text in result["feedback"].items()
    }
