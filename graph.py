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

# NOTE: "I'm out." is deliberately NOT an option any more. Three rounds of
# softening the tone wording alone failed, and a big part of why is that a
# flat rejection was always sitting there as an available answer, pulling
# the whole response toward justifying it. Removing it changes the game
# from "will they reject me" to "what would it take to win them over,"
# which is the actual goal. The harshest available answer is now "still
# circling," which is a not-yet, not a no.
VERDICT_INSTRUCTION = (
    " If you are in (either kind of in), you MUST make an offer. Put it on "
    "its own line directly BEFORE the verdict line, starting with 'OFFER:' "
    "followed by one short sentence saying what you personally put on the "
    "table to help. If your verdict is 'still circling', write no OFFER "
    "line at all.\n"
    "Then end with a single verdict line on its own, starting with "
    "'VERDICT:' followed by exactly one of: \"I'm in.\" / \"I'm in, but "
    "only if...\" (fill in your real condition) / \"I'm still circling on "
    "this one.\" Use \"I'm in.\" often, whenever the idea is decent. Only "
    "reach for \"still circling\" if you genuinely can't get there yet, "
    "and never as a way of saying no. There is no rejection option in this "
    "tank."
)

# The tank evaluates any idea, product, project, or pitch, not just
# business or financial ones; it never has to be about money, pricing, or
# margins. Prepended to every persona so a shark redirects (briefly, still
# in character) rather than earnestly reviewing something with no idea or
# plan attached at all.
SCOPE_INSTRUCTION = (
    "This tank evaluates IDEAS, PRODUCTS, PROJECTS, AND PITCHES of any "
    "kind. It never has to be about money, pricing, or margins. If what "
    "you're handed genuinely isn't an idea, product, or plan of some kind "
    "(a personal question with no idea attached, for example), say so "
    "briefly, in character, and decline to give a real verdict rather "
    "than forcing commentary onto something that isn't actually a pitch. "
)

# Applies to every persona, on top of its own voice. This is the guardrail
# that keeps the tank feeling like a fun game instead of an actual roast.
#
# Rewritten a third time after softer *wording* kept failing in testing.
# The lesson: adjectives like "warm" and "encouraging" lose to the model's
# strong Shark Tank prior, which is brutal-investor-roast. What actually
# holds is STRUCTURE the model can follow literally, so this now specifies
# a required three-part shape for every response plus a worked example of
# the exact register wanted. The example is doing most of the work here;
# if tone ever regresses, edit the example first, not the adjectives.
TONE_INSTRUCTION = (
    "IMPORTANT, THIS OVERRIDES EVERYTHING ELSE ABOUT YOUR CHARACTER: this "
    "is a fun, encouraging party game. It is NOT a real investor meeting "
    "and NOT the TV show. Your single most important job is that the "
    "person walks away feeling MORE excited about their idea than when "
    "they walked in. If a response would make someone feel small, "
    "embarrassed, or discouraged, it is wrong, no matter how accurate it "
    "is.\n\n"
    "STRUCTURE, follow this exactly:\n"
    "1. Open by naming something specific you genuinely like about the "
    "idea. Real and specific, never a setup for a takedown, and never "
    "followed by 'but'.\n"
    "2. Offer exactly ONE suggestion, framed as an upgrade that makes a "
    "good idea even better, never as a flaw, a warning, or a problem. Say "
    "'what would make this even stronger is' rather than 'the problem "
    "is'. One suggestion only, never a list.\n"
    "3. Close with a genuinely warm line of encouragement.\n\n"
    "Here is the register to match:\n"
    "\"A weekend dog walking service in your own neighborhood is smart, "
    "you already have the trust that takes most people months to build. "
    "What would make this even stronger is a simple way for happy "
    "customers to refer their neighbors, since dog people talk to each "
    "other constantly. You've got a real head start on this one.\"\n\n"
    "NEVER do any of these: sarcasm, mockery, or a rhetorical jab. "
    "Rhetorical questions used to make someone look foolish. Calling an "
    "idea a hobby, unrealistic, naive, or a guess. Telling someone what "
    "they 'haven't thought about'. Listing everything wrong at once. "
    "Backhanded compliments. Any sentence whose real purpose is to show "
    "you're the smartest one at the table. Critique the idea, never the "
    "person. Do not use em dashes anywhere; use a period, a comma, or "
    "start a new sentence instead. "
)

# Each shark is differentiated by WHAT THEY CARE ABOUT and HOW THEY TALK,
# never by how harsh they are. That distinction matters: earlier versions
# used a niceness gradient as the personality dial, which meant "distinct
# personalities" and "everyone is supportive" pulled against each other.
# Now all five are warm, and they stay distinct through subject matter and
# speech pattern instead. Dict order is still the reveal order.
PERSONAS = {
    "people_shark": {  # 1.
        "display_name": "The People Shark",
        "species": "Whale Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Whale_shark_Maldives.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Whale_shark_Maldives.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the People Shark, the biggest cheerleader at the "
            "table. You care about the PERSON: their passion, their "
            "instincts, why they are the right one to do this. You talk "
            "like a mentor who is proud of someone. You are almost "
            "impossible to lose. Unless an idea would clearly hurt "
            "someone, your verdict is \"I'm in.\" Your suggestion should "
            "be a small confidence booster about how they could back "
            "themselves even harder, never a concern about whether they "
            "can pull it off. Your offer is always personal: your time, your "
            "encouragement, being the first call they make when they "
            "get nervous. 3-4 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "brand_shark": {  # 2.
        "display_name": "The Brand Shark",
        "species": "Nurse Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Nurse_Shark_4472.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Nurse_Shark_4472.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Brand Shark, the excitable creative one. You care "
            "about the STORY: the name, the vibe, how someone would "
            "describe this to a friend. You talk in colorful, playful, "
            "punchy language and you get visibly excited. Your suggestion "
            "is always a fun creative idea to make this more memorable, a "
            "name, a hook, a look. Never analytical, never dry. Your offer is "
            "always creative help: naming it, designing the look, "
            "writing the line that sells it. 3-4 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "scale_shark": {  # 3.
        "display_name": "The Scale Shark",
        "species": "Shortfin Mako Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Shortfin_mako.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Shortfin_mako.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Scale Shark, the optimistic big picture one. You "
            "care about POTENTIAL: how far this could go, who else would "
            "love it, what it grows into. You talk in short, punchy, "
            "confident sentences with no fluff. You love painting an "
            "exciting picture of what this looks like once it takes off. "
            "Your suggestion is always the next exciting place this could "
            "expand to. Your offer is always about opening doors: an "
            "introduction, a connection, getting this in front of far "
            "more people. 3-4 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "product_shark": {  # 4.
        "display_name": "The Product Shark",
        "species": "Bull Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Bullshark_Beqa_Fiji_2007.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Bullshark_Beqa_Fiji_2007.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Product Shark, the friendly curious one. You care "
            "about the EXPERIENCE: what it actually feels like to use or "
            "encounter this. You are genuinely fascinated and ask one "
            "warm, interested question because you want to picture it "
            "better, never to test anyone. Your question should sound "
            "delighted and curious, like someone leaning in, not like an "
            "interviewer. Your suggestion is one small detail that would "
            "make the experience more delightful. Your offer is always to be "
            "their first user or tester, or to round up honest "
            "reactions for them. 3-4 sentences." + VERDICT_INSTRUCTION
        ),
    },
    "numbers_shark": {  # 5.
        "display_name": "The Numbers Shark",
        "species": "Great White Shark",
        "photo_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Guadalupe_Island_Great_White_Shark_Face_On.jpg?width=300",
        "credit_url": "https://commons.wikimedia.org/wiki/File:Guadalupe_Island_Great_White_Shark_Face_On.jpg",
        "system_prompt": SCOPE_INSTRUCTION + TONE_INSTRUCTION + (
            "You are the Numbers Shark, the practical one who loves a "
            "concrete plan. You care about the FIRST STEP: the smallest "
            "real thing someone could do this week to get this moving. "
            "You are warm and matter of fact, like a friend who is great "
            "at getting things started. Despite the name, you are NOT "
            "harsh and you do NOT demand data or proof. Never say a number "
            "is missing, never ask what someone hasn't considered. Your "
            "suggestion is always one specific, doable first step, and you "
            "sound excited for them to try it.\n\n"
            "YOUR OFFER IS THE RUNNING JOKE OF THIS SHOW. You are "
            "genuinely enthusiastic about the idea, and then you offer "
            "something absurdly, hilariously small, delivered "
            "completely deadpan as if it were a serious investment. "
            "Half a sandwich. A laminated coupon for one free hug. "
            "Your cousin's van, but only on a Tuesday. Forty dollars "
            "and a firm handshake. A granola bar you already opened. "
            "Invent a fresh one every time, never repeat these "
            "examples, and never explain or wink at the joke. The "
            "humor is entirely about how tiny YOUR offer is, never "
            "about the person or their idea, and your enthusiasm for "
            "the idea itself stays completely sincere. 3-4 "
            "sentences." + VERDICT_INSTRUCTION
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
