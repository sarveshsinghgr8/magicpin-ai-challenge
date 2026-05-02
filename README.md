# Vera Enhanced — magicpin AI Challenge Submission

## Approach

**Architecture**: LLM-guided composition with a strict **Composition Guide** that acts as the single source of truth for all output. The guide enforces deterministic, consistent responses through:

1. **Temperature=0 + Seed=42** — Eliminates randomness. Same input → same output.
2. **Structured JSON output** — LLM returns only `{body, cta, send_as, rationale}`. No free-form text.
3. **RAG-style grounding** — LLM receives ONLY the pushed contexts as its knowledge base. Cannot hallucinate beyond what the judge provides.
4. **Few-shot examples** — 5 golden examples embedded in the guide showing exactly what "good" looks like.
5. **Negative constraints** — Explicit "NEVER do X" rules prevent common failure modes.
6. **Rule-based fallback** — If LLM fails or is unavailable, deterministic templates ensure the bot never times out.

### The Composition Guide (the "brain")

The guide is a ~2500-token system prompt that defines:
- Output JSON schema (mandatory)
- 10 absolute constraints (never fabricate, never use taboo words, etc.)
- Voice rules per category (dentist=clinical-peer, salon=warm-practical, etc.)
- CTA selection logic
- Compulsion levers catalog
- Hindi-English code-mix rules
- 5 few-shot examples scoring 47-50/50

### Multi-turn Conversation Handling

The Reply Guide handles conversations with priority-ordered detection:
1. **Auto-reply** → wait/end (pattern matching, no LLM needed)
2. **Hostile** → graceful exit (safety-critical, no LLM needed)
3. **Commitment** → action mode (LLM for natural response)
4. **Question** → answer with context
5. **Default** → acknowledge + next step

## Deployment

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Set environment variables
export LLM_PROVIDER=gemini          # openai | gemini | groq | ollama | deepseek
export LLM_API_KEY=your_key_here
export LLM_MODEL=                   # optional, uses provider default

# Run
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Works without an API key (falls back to rule-based), but scores significantly higher with an LLM.

## Testing

```bash
# Start bot
uvicorn bot:app --host 0.0.0.0 --port 8080

# Run judge (separate terminal)
python judge_simulator.py
```

## Tradeoffs

| Decision | Why |
|----------|-----|
| Gemini Flash as default | Fast (sub-2s), free tier generous, JSON mode native |
| Rule-based fallback | Never timeout — bot responds even if LLM is down |
| Pattern-match for auto-reply/hostile | Safety-critical detection shouldn't depend on LLM latency |
| temperature=0, seed=42 | Reproducible output for the same context inputs |
| No conversation memory across ticks | Avoids stale state; each tick uses fresh pushed contexts |

## What Additional Context Would Have Helped

1. A/B test results from production Vera — which levers actually convert per category
2. Real slot availability API — for confirmed bookable time slots
3. Competitor pricing live data — for sharper competitive positioning
4. Merchant response rate by trigger kind — to prioritize high-engagement triggers

## Files

| File | Purpose |
|------|---------|
| `bot.py` | FastAPI app with LLM-guided composer + conversation handler |
| `submission.jsonl` | 25 pre-composed messages for seed test pairs |
| `generate_submission.py` | Helper script to regenerate submission.jsonl |
| `README.md` | This file |
