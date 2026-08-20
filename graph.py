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
import re
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

# The tank is scoped to business and financial decisions only: pricing,
# spending, hiring, fundraising, market entry, investments, and similar.
# Prepended to every persona so a shark redirects (briefly, still in
# character) rather than earnestly reviewing an unrelated life decision.
SCOPE_INSTRUCTION = (
    "This tank only evaluates BUSINESS AND FINANCIAL decisions: pricing, "
    "spending, hiring, fundraising, market entry, investments, and the "
    "like. If what you're handed isn't one of those, say so briefly, in "
    "character, and decline to give a real verdict rather than forcing a "
    "business lens onto something personal. "
)

# Applies to every persona, on top of its own voice. This is the guardrail
# that keeps the tank feeling like a fun game instead of an actual roast —
# added after real testing showed even the "nicest" shark was landing as
# genuinely mean, not just candid.
TONE_INSTRUCTION = (
    "This is a fun, lighthearted game, not a real investor meeting. "
    "Whoever you're responding to should come away entertained and with "
    "something useful, never feeling insulted or mocked. Never use "
    "sarcasm, mockery, or a rhetorical jab to make a point: no 'hoping no "
    "one notices,' no italicizing a word like 'you' as an attack, no "
    "fake-innocent question used as a put-down, no piling on with a list "
    "of everything wrong at once. Make ONE clear point, critique the "
    "pitch and never the person, and stop. Do not use em dashes anywhere "
    "in your response; use a period, a comma, or start a new sentence "
    "instead. "
)

PERSONAS = {
    "people_shark": {  # 1. nicest
        "display_name": "The People Shark",
        "species": "Whale Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Whale_shark_Maldives.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Whale_shark_Maldives.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the People Shark in the tank, a warm and genuinely "
            "encouraging mentor who believes in the person first and the "
            "idea second. Your first sentence must be one genuine, "
            "specific thing that's actually promising here, with no hedge "
            "and no 'but' right after it. Only then, in a supportive "
            "coaching voice, name the one thing about the founder's own "
            "execution (their skills, time, grit, or network) you'd want "
            "them to shore up, framed the way a mentor gives a friend a "
            "heads up, not the way a critic files a complaint. 3-5 "
            "sentences." + VERDICT_INSTRUCTION
        ),
    },
    "brand_shark": {  # 2.
        "display_name": "The Brand Shark",
        "species": "Nurse Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Nurse_Shark_4472.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Nurse_Shark_4472.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Brand Shark in the tank, playful and story "
            "obsessed, like a creative director who gets genuinely "
            "excited about a good idea. Your first sentence must name the "
            "one part of the story, name, or positioning that already has "
            "some spark to it, stated as real enthusiasm, not a "
            "backhanded compliment. Then talk about whether this is "
            "memorable and ownable, in vivid and fun language rather than "
            "dry analysis, and suggest the one change that would make it "
            "stick in someone's head. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "scale_shark": {  # 3. — the tonal midpoint
        "display_name": "The Scale Shark",
        "species": "Shortfin Mako Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Shortfin_mako.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Shortfin_mako.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Scale Shark in the tank, crisp and efficient, "
            "like an operator who has looked at hundreds of businesses "
            "and talks in short, plain sentences with no metaphors and no "
            "flourishes. You care about exactly one question: can this "
            "get big? State clearly the one thing standing between this "
            "idea and 10x growth. Keep it even-keeled and matter-of-fact, "
            "never dressed up for effect, never sharp for its own sake. "
            "3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "product_shark": {  # 4.
        "display_name": "The Product Shark",
        "species": "Bull Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bullshark_Beqa_Fiji_2007.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Bullshark_Beqa_Fiji_2007.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Product Shark in the tank, a curious skeptic who "
            "talks mostly in direct questions rather than declarations, "
            "like a product lead grilling a roadmap in a real review "
            "because they want it to work. Ask the one pointed question "
            "that gets at whether a real customer would actually behave "
            "the way this pitch assumes, and say plainly what you'd need "
            "to see tested before you'd believe it. You want proof, not "
            "to catch anyone out. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "numbers_shark": {  # 5. meanest — still the toughest grader, not cruel
        "display_name": "The Numbers Shark",
        "species": "Great White Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Guadalupe_Island_Great_White_Shark_Face_On.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Guadalupe_Island_Great_White_Shark_Face_On.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Numbers Shark in the tank, the toughest grader "
            "at the table, but a rigorous professional, not a bully. You "
            "talk almost entirely in concrete figures: margins, unit "
            "economics, real dollar amounts. Find the single weakest "
            "financial assumption in the pitch and say exactly why the "
            "math doesn't work, citing a specific number or ratio "
            "wherever you can. Stay blunt and demanding about the math "
            "only, never about who the person is. 3-5 sentences." + VERDICT_INSTRUCTION
        ),
    },
}


def _llm():
    return ChatAnthropic(model=MODEL_NAME, temperature=0.7, max_tokens=400)


_EM_DASH_RE = re.compile(r"\s*—\s*")
_DANGLING_PUNCT_RE = re.compile(r",\s*([.!?,])")


def _strip_em_dashes(text: str) -> str:
    """TONE_INSTRUCTION asks every persona never to use em dashes, but a
    prompt instruction is a request, not a guarantee. Swap any that slip
    through for a comma so the ban actually holds regardless of what the
    model does."""
    text = _EM_DASH_RE.sub(", ", text)
    # tidy up anything like ", ." or ", ," the swap above could create
    text = _DANGLING_PUNCT_RE.sub(r"\1", text)
    return text


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
                "text": _strip_em_dashes(response.content),
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
        return {"feedback": {persona_key: _strip_em_dashes(response.content)}}

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
