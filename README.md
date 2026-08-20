# The Tank

A multi-agent bot built with **LangGraph** + **LangChain** + **FastAPI** —
Shark-Tank-style pitch reviews for **business and financial decisions**.

You give it a pricing call, a spending decision, a hiring plan, a
fundraise, a market-entry idea, an investment — anything business or
financial. (It's scoped on purpose: each shark's prompt is instructed to
call out and decline anything that isn't, in character, rather than
earnestly reviewing e.g. a personal relationship question. It's not
licensed financial, legal, or investment advice — it's for brainstorming
and entertainment.) Five shark agents review it **one at a time, nicest to
meanest**: The People Shark → The Brand Shark → The Scale Shark → The
Product Shark → The Numbers Shark. Each reviews independently — they don't
see each other's answers, and there's no synthesis step that averages them
into a single "correct" answer — and each ends with their own verdict:
*I'm in* / *I'm out* / *I'm in, but only if...* On the frontend, that plays
out as a real sequential reveal (not a fake animation over data that was
already all there): a shark swims into the tank, thinks, delivers its
verdict, and the next one enters. The water gets visibly bloodier with
every rejection, and a rejecting shark keeps circling in the background for
the rest of the session.

Each shark's avatar is a real photo of a real shark species (not an
illustration, and not any real person from the actual show) — species
chosen to roughly match that shark's personality tier, from a gentle giant
to an apex predator:

| Shark persona   | Species              |
|------------------|----------------------|
| The People Shark  | Whale Shark           |
| The Brand Shark   | Nurse Shark            |
| The Scale Shark   | Shortfin Mako Shark    |
| The Product Shark | Bull Shark             |
| The Numbers Shark | Great White Shark      |

Photos are hotlinked from Wikimedia Commons and tinted with each shark's
accent color. Every avatar links through to its Commons file page for full
license/photographer credit (see the footer in the app) rather than this
project asserting attribution text itself.

## How it works

- `graph.py` — `iter_tank()` is the default path: a generator that calls
  each shark's LLM one at a time, in the fixed nicest→meanest order, and
  yields a `start`/`result`/`error` event around each call. This is what
  makes the one-at-a-time reveal real — the backend is actually only
  talking to one shark at a time, not faking the pacing client-side. The
  old fully-parallel version is still there (`build_graph()` /
  `run_tank_parallel()`) if you ever want low latency instead of drama.
- `main.py` — `POST /advise/stream` streams newline-delimited JSON events
  as each shark finishes (what the UI uses). `POST /advise` is a blocking
  variant that runs the same sequence and returns everything at once, for
  simple API callers. `GET /personas` lists the sharks.
- `static/index.html` — the real frontend, wired to `/advise/stream`.
- `static/demo.html` — a standalone preview with the identical UI/JS but a
  simulated stream (canned responses + a fake thinking delay) so you can
  see the whole experience without a server or API key.

## Setup

```bash
cd shark-tank-bot
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

Get a key at https://console.anthropic.com/ if you don't have one.

## Run

```bash
uvicorn main:app --reload
```

Then open http://localhost:8000 in a browser.

## Deploying to Vercel

`vercel.json` and `.vercelignore` are already set up — Vercel auto-detects
the FastAPI `app` instance in `main.py` and serves `static/` correctly
since it's mounted with `StaticFiles`, so this should be close to a
zero-config import.

1. Push this repo to GitHub (see `push_to_github.sh` / `push_to_github.ps1`).
2. On [vercel.com](https://vercel.com), New Project → import the repo.
3. In the project's Environment Variables, add `ANTHROPIC_API_KEY` (and
   `ANTHROPIC_MODEL` if you're overriding the default). Nothing in the repo
   itself contains the key — it's git-ignored via `.env`.
4. Deploy.

**One real risk worth testing first, not assuming works:** the app's whole
"one shark at a time" experience depends on `POST /advise/stream` actually
streaming newline-delimited JSON as each shark finishes, rather than
buffering the full response and sending it all at once at the end. Vercel's
Python serverless runtime has had reported issues with exactly this
pattern — FastAPI `StreamingResponse` calls that stream fine locally but
arrive all at once (or 500) once deployed there. `maxDuration` is set to
120s in `vercel.json` as headroom either way (Hobby's default/max is 300s,
so there's room to raise it), but duration isn't the risk here — buffering
is. After deploying, load the real page (not just curl the endpoint) and
watch whether sharks actually appear one at a time or all dump in at once
at the end. If it buffers, the app still works, it just loses the reveal
pacing — and a platform built for long-lived server processes rather than
serverless functions (Render, Fly.io, Railway) would be a more reliable
fit for this specific streaming pattern.

## API

**GET /personas** — list available sharks and their prompts, in reveal order.

**POST /advise/stream** — newline-delimited JSON, one event per line:
```json
{"type": "start",  "key": "people_shark", "display_name": "The People Shark", "index": 0, "total": 5}
{"type": "result", "key": "people_shark", "display_name": "The People Shark", "index": 0, "total": 5, "text": "... VERDICT: I'm in."}
{"type": "error",  "key": "...", "display_name": "...", "index": 2, "total": 5, "error": "..."}
{"type": "done"}
```
A single shark erroring doesn't kill the stream — you get an `error` event and the next shark still runs.

**POST /advise** — same request body, blocking:
```json
{
  "topic": "I'm thinking of quitting my job to freelance full-time.",
  "personas": ["people_shark", "numbers_shark"]   // optional, omit for all five, order doesn't matter — always replayed nicest→meanest
}
```
Returns `{"topic": "...", "feedback": {"The People Shark": "...VERDICT: I'm in.", ...}}`.

## The fun stuff

- **Sequential reveal** with a progress track of five fin icons (pending → pulsing "current" → colored by verdict).
- **Typewriter text** for each shark's response — click a card mid-type to skip straight to the end.
- **Sound effects**, all synthesized in-browser via the Web Audio API (no
  asset files): a whoosh as each shark enters, a rising chime for "I'm in",
  a neutral ding for conditional, and a two-note ominous bass sting — an
  original synthesized motif, not a sample of anything — for "I'm out".
  There's a mute button in the pitch panel.
- **Blood in the water**: a red tint over the whole tank that gets stronger
  with every rejection (5 opacity levels), plus a screen shake and a small
  bite-mark/drip decoration on any card that goes "I'm out."
- **Swimming sharks in the background**: a fast one-shot pass while a shark
  is actively "thinking," and a persistent looping silhouette — tinted to
  match that shark's own color — added for every shark that ends up OUT.
- **Confetti** on an "I'm in," and a slammed rubber stamp (APPROVED /
  MIXED VERDICT / REJECTED) once all sharks have weighed in, based on
  majority verdict among the ones that actually responded.

## Customizing / extending

- **Add or edit sharks**: edit the `PERSONAS` dict in `graph.py`. Dict order
  IS the reveal order (nicest first) — reorder entries to change who goes
  when. Each entry needs a `display_name` and `system_prompt`;
  `VERDICT_INSTRUCTION` is appended to every prompt so each shark reliably
  ends with a parseable verdict line.
- **Adjust the tone gradient**: each system prompt has a "how mean are they"
  dial baked into its wording (mentor-warm for People, savage one-liner for
  Numbers). Tune to taste — the frontend's verdict classification
  (`classifyVerdict` in the JS) just looks for "I'm out" / "but only if" /
  else-is-"I'm in" in the text, so it doesn't care about tone.
- **Add a synthesis step**: if you want a "producer" agent that reads all
  five verdicts and writes a summary, add a step after `iter_tank()` (or a
  final LangGraph node in the parallel path) that calls the LLM with all
  collected feedback as context. Left out on purpose so the sharks keep
  disagreeing.
- **Swap the model**: set `ANTHROPIC_MODEL` in `.env`.
- **Change the sound effects**: they're small synthesized-tone functions in
  the `<script>` block (`playChime`, `playJawsSting`, etc.) — no audio
  files to manage, just tweak frequencies/durations.
- **Deploy**: this is a plain FastAPI app — deploy behind `uvicorn`/`gunicorn`
  on any host (Render, Fly.io, a VM, etc.). Tighten the CORS policy in
  `main.py` (`allow_origins=["*"]`) before putting it on the public internet.

## Notes / things worth knowing before you rely on this

- **Sequential is slower.** Five LLM calls one after another instead of in
  parallel means total latency is roughly 5x a single call — expect the
  full run to take significantly longer than the old parallel version did.
  That's the deliberate trade for a real (not faked) one-at-a-time reveal.
  If you want speed back, `run_tank_parallel()` in `graph.py` still exists.
- There's no conversation memory — each request is stateless. If you want
  the tank to remember prior pitches, add a checkpointer (LangGraph
  supports this natively) and a session/thread id.
- No auth on the API as shipped. Add something (API key header, etc.)
  before exposing this beyond localhost.
- No rate limiting or cost guard — every full run is 5 LLM calls. Worth
  adding before sharing a public link to this.
- These five sharks are original fictional personas built around common
  investor-evaluation angles (people, brand, scale, product instinct,
  numbers) — not based on any real person.
- **Scope is enforced by the prompt, not by code.** `SCOPE_INSTRUCTION` in
  `graph.py` tells every shark to redirect (in character) if what it's
  handed isn't a business/financial decision, but nothing blocks an
  off-topic request from reaching the LLM — it's a steer, not a filter. If
  you want a hard gate, add a cheap pre-check (keyword match, or a small
  classification call) in `main.py` before `iter_tank()` runs.
- **The avatar photo hotlinks were not verified rendering in the
  environment they were built in** — that sandbox's network access was
  restricted to package registries, so the Wikimedia Commons image URLs
  (`Special:FilePath/...`) could be confirmed as real, existing file pages
  via search, but the actual images couldn't be fetched to preview. Open
  the app and check the five avatars load correctly; if any don't, there's
  a built-in fallback (an `onerror` handler swaps in a solid tinted circle
  with the shark's icon badge, so nothing looks broken either way) — but a
  stale filename is possible and worth a quick visual check. To swap a
  photo, edit the matching `photo_url` in `graph.py`'s `PERSONAS` dict
  *and* the matching `photo` field in `SHARK_META` in both
  `static/index.html` and `static/demo.html` (three places, kept in sync
  manually rather than fetched from one source).
