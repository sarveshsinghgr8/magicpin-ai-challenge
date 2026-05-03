"""
Vera AI Merchant Assistant — magicpin AI Challenge Bot
======================================================

LLM-powered bot with a strict Composition Guide that controls all output.
The LLM responds ONLY based on the guide — no hallucination, no creativity beyond constraints.

Architecture:
- FastAPI server with 5 endpoints
- Composition Guide (system prompt) defines output format, constraints, voice rules
- LLM called with temperature=0, fixed seed for deterministic responses
- RAG-style: contexts pushed by judge are the ONLY grounding data
- Fallback to rule-based if LLM fails

Supported LLM providers: openai, gemini, groq, ollama, deepseek
Configure via environment variables.

Run: uvicorn bot:app --host 0.0.0.0 --port 8080

Environment variables:
    LLM_PROVIDER=gemini          (openai|gemini|groq|ollama|deepseek)
    LLM_API_KEY=your_key_here
    LLM_MODEL=                   (optional, uses provider default)
    OLLAMA_URL=http://localhost:11434  (ollama only)
"""

import os
import time
import json
import uuid
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib import request as urlrequest, error as urlerror

# Load .env file
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

app = FastAPI(title="Vera AI Bot", version="2.0.0")
START = time.time()

# Self-ping to prevent Render free-tier sleep (pings /v1/healthz every 13 min)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

def _self_ping():
    """Background thread that pings own healthz to prevent idle shutdown."""
    import time as _time
    while True:
        _time.sleep(780)  # 13 minutes
        if RENDER_URL:
            try:
                urlrequest.urlopen(f"{RENDER_URL}/v1/healthz", timeout=10)
            except Exception:
                pass

if RENDER_URL:
    _ping_thread = threading.Thread(target=_self_ping, daemon=True)
    _ping_thread.start()

# =============================================================================
# CONFIGURATION — Set via environment variables
# =============================================================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_openrouter_base = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
OPENROUTER_URL = _openrouter_base.rstrip("/") + "/chat/completions" if not _openrouter_base.endswith("/chat/completions") else _openrouter_base
LLM_TIMEOUT = 45  # seconds — generous for free-tier models

# =============================================================================
# IN-MEMORY STORES
# =============================================================================

contexts: dict[tuple[str, str], dict] = {}  # (scope, context_id) -> {version, payload}
conversations: dict[str, list] = {}  # conversation_id -> [turns]

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# =============================================================================
# THE COMPOSITION GUIDE — Controls ALL LLM output
# =============================================================================
# This is the single source of truth for how Vera composes messages.
# The LLM MUST follow it strictly. Temperature=0 + this guide = consistent output.

COMPOSITION_GUIDE = """# VERA COMPOSITION GUIDE — STRICT RULES

You are Vera, magicpin's merchant AI assistant. You compose WhatsApp messages.

## OUTPUT FORMAT — MANDATORY JSON (no markdown, no explanation, no preamble)
{
  "body": "<the WhatsApp message text, max 500 chars>",
  "cta": "<binary_yes_no | open_ended | multi_choice | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "rationale": "<1-2 sentence explanation of compulsion levers used>"
}

## ABSOLUTE CONSTRAINTS — NEVER VIOLATE
1. ONLY use data present in the provided context. NEVER invent facts, offers, prices, names, citations.
2. NEVER use words from the category vocab_taboo list.
3. MAX 500 characters for body.
4. ONE primary call-to-action only.
5. NO promotional/salesy language — peer tone.
6. NO greetings like "Hope you're doing well" or filler.
7. NO internal system terminology exposed to merchant.
8. ALWAYS anchor on at least TWO verifiable facts from the context (numbers, dates, peer comparisons, sources). Include a merchant-vs-peer gap when peer_stats differ. If only ONE fact is available in the context, use that one — NEVER invent a second fact.
9. MATCH language preference (hindi-english mix if indicated).
10. If scope=customer → send_as MUST be "merchant_on_behalf".
11. If scope=merchant → send_as MUST be "vera".
12. ALWAYS end with effort externalization when offering work ("I'll draft X — ready in 5 min", "Live in 10 min").

## VOICE RULES BY CATEGORY
| Category | Tone | Salutation | Key rules |
|----------|------|-----------|-----------|
| dentists | clinical-peer | Dr. {first_name} | Technical terms OK. No health claims. Source-cite research. |
| salons | warm-practical | Hi {first_name} | Friendly expert. Emojis OK (1-2 max). |
| restaurants | fellow-operator | Hi {first_name} | Business metrics (covers, footfall). Operator shorthand. |
| gyms | coach-to-operator | Hi {first_name} | Energetic. Data-driven. No hype. |
| pharmacies | trustworthy-precise | Hi {first_name} | Regulatory awareness. Molecule names OK. |

## CTA SELECTION RULES
- binary_yes_no → Action triggers requiring commitment (recall booking, renewal, draft approval)
- open_ended → Curiosity/knowledge triggers (research digest, curious ask, competitor alert)
- multi_choice → Slot/option selection (Reply 1 for X, 2 for Y)
- none → Pure information with no ask needed

## COMPULSION LEVERS (use 2-3 per message, NEVER just 1)
- Specificity: Real numbers/dates/sources from context
- Loss aversion: What they're missing (-X% visibility, Y lapsed customers)
- Curiosity: Tease useful information
- Social proof: Peer stats, benchmarks (MANDATORY when peer_stats provided — e.g., "your CTR 2.1% vs category avg 3.0%")
- Reciprocity: "Want me to draft/pull/create..."
- Effort externalization: "Takes 5 min", "Live in 10 min" (MANDATORY when offering to do work)
- Single-binary CTA: Make responding trivially easy

## MANDATORY LEVER RULES
- ALWAYS use Social Proof when peer_stats show a gap (merchant metric < peer avg)
- ALWAYS use Loss Aversion when customer_aggregate shows lapsed customers (e.g., "78 patients haven't returned in 6mo")
- ALWAYS end with Effort Externalization when offering a deliverable (draft, checklist, post)

## HINDI-ENGLISH CODE-MIX (when language_pref includes "hi")
- Mix naturally: "Apke liye 2 slots ready hain"
- Keep technical/business terms in English
- Keep numbers and prices in English format

## FEW-SHOT EXAMPLES

### Example 1: Research Digest → Dentist (merchant-facing)
Context: Dr. Meera, JIDA Oct 2026, 2100-patient trial, 38% caries reduction, high-risk adults
{
  "body": "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14",
  "cta": "open_ended",
  "send_as": "vera",
  "rationale": "Source citation + merchant-specific anchor (high-risk cohort) + reciprocity + specificity"
}

### Example 2: Recall Due → Customer (hi-en mix, customer-facing)
Context: Priya, lapsed_soft, 5 months, cleaning due, slots Wed 6pm / Thu 5pm, ₹299
{
  "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
  "cta": "multi_choice",
  "send_as": "merchant_on_behalf",
  "rationale": "Name personalization + hi-en mix + specific slots/price + low-friction CTA"
}

### Example 3: IPL Match → Restaurant (merchant-facing, counter-intuitive)
Context: Suresh, SK Pizza, DC vs MI Saturday, -12% Saturday covers, BOGO active
{
  "body": "Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). Skip the match-night promo today; instead push your BOGO pizza (already active) as a delivery-only Saturday special. Want me to draft the Swiggy banner + an Insta story? Live in 10 min.",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Counter-intuitive data (-12%) + saves merchant from bad decision + leverages existing offer + 10-min effort cap"
}

### Example 4: Lapsed Customer Winback → Gym (customer-facing)
Context: Rashmi, 57 days lapsed, weight loss focus, free trial available
{
  "body": "Hi Rashmi 👋 Karthik from PowerHouse here. It's been about 8 weeks — happens to most members at some point, no judgment. We've added a Tue/Thu evening HIIT class that fits weight-loss goals well (45 min, 6:30pm). Want me to hold a free trial spot for you next Tue? Reply YES — no commitment, no auto-charge.",
  "cta": "binary_yes_no",
  "send_as": "merchant_on_behalf",
  "rationale": "No-shame framing + addresses past goal + specific new class + no-commitment barrier removal"
}

### Example 5: Supply Alert → Pharmacy (merchant-facing, urgent)
Context: Ramesh, Apollo Pharmacy, atorvastatin recall, batches AT2024-1102/1108, 240 chronic-Rx customers
{
  "body": "Hi Ramesh, urgent — voluntary recall on atorvastatin batches AT2024-1102 + AT2024-1108 by MfrZ. Action needed:\\n1. Pull affected stock immediately\\n2. Check your repeat-Rx list for patients on atorvastatin\\n3. Your 240 chronic-Rx customers may include affected ones\\nWant me to filter your customer list for atorvastatin prescriptions so you can WhatsApp them directly?",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Urgency + specific batch numbers + chronic-Rx count anchor + reciprocity (I'll filter the list)"
}
"""

# =============================================================================
# REPLY GUIDE — For multi-turn conversation handling
# =============================================================================

REPLY_GUIDE = """# VERA REPLY HANDLER — STRICT RULES

You handle a merchant's or customer's reply in an ongoing WhatsApp conversation.

## OUTPUT FORMAT — MANDATORY JSON ONLY
Choose ONE action:

### SEND (continue conversation)
{"action": "send", "body": "<reply, max 300 chars>", "cta": "<type or null>", "rationale": "<reason>"}

### WAIT (back off)
{"action": "wait", "wait_seconds": <600-3600>, "rationale": "<reason>"}

### END (close conversation)
{"action": "end", "body": "<optional farewell, max 100 chars or empty>", "rationale": "<reason>"}

## VERA'S CAPABILITIES (NEVER claim actions outside this list)
Vera CAN: draft messages, pull data, create checklists, filter customer lists, suggest content, share information
Vera CANNOT: book appointments, process payments, access external systems, confirm reservations, make calls
- For booking/slot requests: "Noted — [slot details]. The team will confirm your booking shortly."
- For action requests: "On it — I'll [specific deliverable] and share here in [X] min."
- NEVER say "I have booked" / "Done, confirmed" / "Your appointment is set" — Vera is an assistant, not a booking system.

## DETECTION PRIORITY (apply in order):

1. AUTO-REPLY: "thank you for contacting" / "our team will respond" / generic automated text
   → 1st time: wait 1800s | 2nd+ time: end

2. HOSTILE: "stop" / "spam" / "useless" / "block" / "leave me alone"
   → end with brief apology. NEVER argue.

3. COMMITMENT / ACTION REQUEST: "yes" / "let's do it" / "go ahead" / "sure" / "book me" / "please do X"
   → send. Switch to ACTION mode immediately.
   → Body = acknowledgment + what YOU will do next + timeline.
   → NEVER ask qualifying questions after commitment.
   → NEVER fabricate completed actions.

   CORRECT examples:
   - Merchant says "yes, draft it": {"action":"send","body":"On it. I'll draft the checklist based on DCI's new limits — sharing here in 5 min for your review.","rationale":"Commitment detected. Action mode."}
   - Customer says "book me for Wed 6pm": {"action":"send","body":"Noted — Wed 6pm slot. The clinic team will confirm your booking shortly.","rationale":"Booking request. Acknowledged without fabricating confirmation."}
   - Merchant says "go ahead": {"action":"send","body":"Done — working on it now. Will share the draft here in 5 min for approval.","rationale":"Commitment. Action mode with timeline."}

   WRONG (NEVER do this):
   - "To get started, could you tell me..." ← qualifying after commitment
   - "I have booked your appointment" ← fabricating an action Vera cannot do
   - "Great! What time works?" ← re-qualifying when slot was already given

4. QUESTION: contains "?"
   → send. Answer using available context. If not available: "Let me check and get back."

5. DEFAULT:
   → send. Brief acknowledgment + offer next step.

## CONSTRAINTS
- MAX 300 chars in reply body
- NEVER re-pitch after hostility
- NEVER qualify after commitment
- NEVER fabricate completed actions (bookings, payments, confirmations)
- After turn 5: lean toward end
- When customer picks a slot/option: acknowledge the choice + say team will confirm
"""

# =============================================================================
# LLM CLIENT — Deterministic, multi-provider
# =============================================================================


class LLMClient:
    """
    Calls LLM with strict determinism:
    - temperature=0 (most likely token always chosen)
    - seed=42 (fixed random state)
    - Pinned model version
    """

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL or self._default_model()
        self.available = bool(self.api_key) or self.provider == "ollama"

    def _default_model(self) -> str:
        return {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.1-70b-versatile",
            "ollama": "llama3",
            "deepseek": "deepseek-chat",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }.get(self.provider, "gpt-4o-mini")

    def complete(self, system: str, user_prompt: str) -> Optional[str]:
        """Call LLM. Returns response text or None on failure."""
        if not self.available:
            return None
        try:
            dispatch = {
                "openai": self._call_openai,
                "gemini": self._call_gemini,
                "groq": self._call_groq,
                "ollama": self._call_ollama,
                "deepseek": self._call_deepseek,
                "openrouter": self._call_openrouter,
            }
            fn = dispatch.get(self.provider)
            return fn(system, user_prompt) if fn else None
        except Exception as e:
            print(f"[LLM ERROR] {self.provider}: {e}")
            return None

    def _call_openai(self, system: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }).encode()
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    def _call_gemini(self, system: str, user_prompt: str) -> str:
        full = f"{system}\n\n---\n\n{user_prompt}"
        body = json.dumps({
            "contents": [{"parts": [{"text": full}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 800,
                                 "responseMimeType": "application/json"}
        }).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        req = urlrequest.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["candidates"][0]["content"]["parts"][0]["text"]

    def _call_groq(self, system: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_prompt}],
            "temperature": 0, "seed": 42, "max_tokens": 800
        }).encode()
        req = urlrequest.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    def _call_ollama(self, system: str, user_prompt: str) -> str:
        full = f"{system}\n\n---\n\n{user_prompt}"
        body = json.dumps({
            "model": self.model, "prompt": full, "stream": False,
            "format": "json", "options": {"temperature": 0, "seed": 42}
        }).encode()
        req = urlrequest.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["response"]

    def _call_deepseek(self, system: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_prompt}],
            "temperature": 0, "seed": 42, "max_tokens": 800
        }).encode()
        req = urlrequest.Request(
            "https://api.deepseek.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    def _call_openrouter(self, system: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_prompt}],
            "temperature": 0, "seed": 42, "max_tokens": 800,
            "provider": {"order": ["Anthropic", "Google", "OpenAI"]}
        }).encode()
        req = urlrequest.Request(
            OPENROUTER_URL, data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://magicpin.com",
                     "X-Title": "Vera AI Bot"})
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        return json.loads(resp.read())["choices"][0]["message"]["content"]


# Initialize LLM client
llm = LLMClient()

# =============================================================================
# CONTEXT HELPERS
# =============================================================================


def get_context(scope: str, context_id: str) -> Optional[dict]:
    key = (scope, context_id)
    return contexts[key]["payload"] if key in contexts else None


def get_category_for_merchant(merchant: dict) -> dict:
    return get_context("category", merchant.get("category_slug", "")) or {}


def get_merchant(merchant_id: str) -> dict:
    return get_context("merchant", merchant_id) or {}


def get_customer(customer_id: str) -> dict:
    return get_context("customer", customer_id) or {}


def get_trigger(trigger_id: str) -> dict:
    return get_context("trigger", trigger_id) or {}


# =============================================================================
# COMPOSER — LLM-guided with fallback
# =============================================================================


class VeraComposer:
    """Composes messages: LLM + COMPOSITION_GUIDE → structured JSON output."""

    def compose(self, category: dict, merchant: dict, trigger: dict,
                customer: Optional[dict] = None) -> dict:
        """Build context prompt → call LLM → parse. Falls back to rules if LLM fails."""
        user_prompt = self._build_prompt(category, merchant, trigger, customer)

        # Primary: LLM with guide
        if llm.available:
            result = self._call_llm(user_prompt)
            if result:
                # Inject suppression_key from trigger
                result["suppression_key"] = trigger.get("suppression_key", "")
                return result

        # Fallback: rule-based
        return self._fallback(category, merchant, trigger, customer)

    def _build_prompt(self, category: dict, merchant: dict, trigger: dict,
                      customer: Optional[dict]) -> str:
        """Build structured context for the LLM (the 'retrieved data')."""
        voice = category.get("voice", {})
        peer_stats = category.get("peer_stats", {})
        digest = category.get("digest", [])
        seasonal = category.get("seasonal_beats", [])
        identity = merchant.get("identity", {})
        perf = merchant.get("performance", {})
        m_offers = merchant.get("offers", [])
        signals = merchant.get("signals", [])
        cust_agg = merchant.get("customer_aggregate", {})
        review_themes = merchant.get("review_themes", [])
        conv_history = merchant.get("conversation_history", [])
        payload = trigger.get("payload", {})

        # Find relevant digest item
        ref_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id") or ""
        relevant_digest = next((d for d in digest if d.get("id") == ref_id), None)

        # Pre-compute social proof gaps for the LLM
        peer_ctr = peer_stats.get('avg_ctr')
        merchant_ctr = perf.get('ctr')
        ctr_gap = ""
        if peer_ctr and merchant_ctr and isinstance(peer_ctr, (int, float)) and isinstance(merchant_ctr, (int, float)):
            gap_pp = round((merchant_ctr - peer_ctr) * 100, 1)
            ctr_gap = f"CTR gap: merchant {merchant_ctr} vs peer avg {peer_ctr} ({gap_pp:+}pp)"

        peer_views = peer_stats.get('avg_views_30d')
        merchant_views = perf.get('views')
        views_gap = ""
        if peer_views and merchant_views and isinstance(peer_views, (int, float)) and isinstance(merchant_views, (int, float)):
            views_gap = f"Views gap: merchant {merchant_views} vs peer avg {peer_views} ({merchant_views - peer_views:+})"

        lapsed_count = cust_agg.get('lapsed_180d', 0) or cust_agg.get('lapsed', 0)
        active_count = cust_agg.get('active', 0)

        prompt = f"""## COMPOSE A MESSAGE FOR THIS CONTEXT

### CATEGORY: {category.get('slug', 'unknown')}
- Voice: {voice.get('tone', '?')} / {voice.get('register', '?')}
- Code-mix: {voice.get('code_mix', 'english')}
- Vocab TABOO: {voice.get('vocab_taboo', [])}
- Peer stats: rating={peer_stats.get('avg_rating', '?')}, views_30d={peer_stats.get('avg_views_30d', '?')}, ctr={peer_stats.get('avg_ctr', '?')}, calls={peer_stats.get('avg_calls_30d', '?')}
- Seasonal: {json.dumps(seasonal, ensure_ascii=False)[:200]}
{f'- RELEVANT DIGEST ITEM: {json.dumps(relevant_digest, ensure_ascii=False)}' if relevant_digest else ''}

### MERCHANT
- Name: {identity.get('name', '?')}
- Owner: {identity.get('owner_first_name', '?')}
- City/Locality: {identity.get('city', '?')}, {identity.get('locality', '?')}
- Languages: {identity.get('languages', ['en'])}
- Subscription: {merchant.get('subscription', {}).get('status', '?')}, plan={merchant.get('subscription', {}).get('plan', '?')}, days_left={merchant.get('subscription', {}).get('days_remaining', '?')}
- Performance 30d: views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, ctr={perf.get('ctr', '?')}, directions={perf.get('directions', '?')}
- 7d delta: {json.dumps(perf.get('delta_7d', {}), ensure_ascii=False)}
- Active offers: {json.dumps([o for o in m_offers if o.get('status') == 'active'], ensure_ascii=False)}
- Signals: {signals}
- Customer aggregate: {json.dumps(cust_agg, ensure_ascii=False)}
- Review themes: {json.dumps(review_themes[:3], ensure_ascii=False)}
- Last conversation: {json.dumps(conv_history[-2:], ensure_ascii=False)[:300]}

### SOCIAL PROOF ANCHORS (USE these in the message when relevant)
{f'- {ctr_gap}' if ctr_gap else '- No CTR gap data'}
{f'- {views_gap}' if views_gap else '- No views gap data'}
{f'- Lapsed customers: {lapsed_count} (use for loss aversion)' if lapsed_count else '- No lapsed data'}
{f'- Active customers: {active_count}' if active_count else ''}

### TRIGGER
- Kind: {trigger.get('kind', '?')}
- Scope: {trigger.get('scope', '?')}
- Source: {trigger.get('source', '?')}
- Urgency: {trigger.get('urgency', 1)}/5
- Payload: {json.dumps(payload, ensure_ascii=False)}
"""
        if customer:
            ci = customer.get("identity", {})
            cr = customer.get("relationship", {})
            cp = customer.get("preferences", {})
            prompt += f"""
### CUSTOMER (this is CUSTOMER-FACING)
- Name: {ci.get('name', '?')}
- Language pref: {ci.get('language_pref', 'english')}
- Age: {ci.get('age_band', '?')}
- State: {customer.get('state', '?')}
- Visits: {cr.get('visits_total', '?')}, last_visit: {cr.get('last_visit', '?')}
- Services: {cr.get('services_received', [])}
- Prefs: {json.dumps(cp, ensure_ascii=False)}
- Consent: {customer.get('consent', {}).get('scope', [])}
- IMPORTANT: send_as MUST be "merchant_on_behalf"
"""
        else:
            prompt += "\n### This is MERCHANT-FACING. send_as MUST be \"vera\".\n"

        prompt += "\nCompose the message. Return ONLY the JSON object.\n"
        return prompt

    def _call_llm(self, user_prompt: str) -> Optional[dict]:
        """Call LLM with guide, parse JSON."""
        response = llm.complete(COMPOSITION_GUIDE, user_prompt)
        if not response:
            return None

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                return None
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        if "body" not in result:
            return None

        return {
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": "",
            "rationale": result.get("rationale", "")
        }

    def _fallback(self, category: dict, merchant: dict, trigger: dict,
                  customer: Optional[dict]) -> dict:
        """Deterministic rule-based fallback."""
        owner = merchant.get("identity", {}).get("owner_first_name", "there")
        cat_slug = category.get("slug", "")
        kind = trigger.get("kind", "update")
        payload = trigger.get("payload", {})
        salutation = f"Dr. {owner}" if cat_slug == "dentists" else f"Hi {owner}"

        # Customer-facing
        if customer:
            cust_name = customer.get("identity", {}).get("name", "there")
            merchant_name = merchant.get("identity", {}).get("name", "")
            body = f"Hi {cust_name}, {merchant_name} here. "
            if kind == "recall_due":
                slots = payload.get("available_slots", [])
                slot_text = " or ".join(s.get("label", "") for s in slots[:2])
                body += f"Your {payload.get('service_due', 'checkup').replace('_', ' ')} is due. Available: {slot_text}. Reply to book."
            elif kind == "customer_lapsed_hard":
                weeks = payload.get("days_since_last_visit", 30) // 7
                body += f"It's been {weeks} weeks — no pressure. Reply YES for a free trial spot."
            elif kind == "chronic_refill_due":
                meds = ", ".join(payload.get("molecule_list", [])[:3])
                body += f"Your refill for {meds} is due. Reply OK to confirm."
            elif kind == "wedding_package_followup":
                days = payload.get("days_to_wedding", 0)
                body += f"💍 {days} days to your wedding! Perfect time to start skin-prep. Want to book?"
            elif kind == "trial_followup":
                sessions = payload.get("next_session_options", [])
                slot = sessions[0].get("label", "this week") if sessions else "soon"
                body += f"Enjoyed the trial? Next session: {slot}. Reply YES to book."
            else:
                body += "Quick update — reply if you'd like to know more."
            return {"body": body.strip(), "cta": "binary_yes_no",
                    "send_as": "merchant_on_behalf",
                    "suppression_key": trigger.get("suppression_key", ""),
                    "rationale": f"Fallback: {kind} (customer-facing)"}

        # Merchant-facing
        digest = category.get("digest", [])
        ref_id = payload.get("top_item_id") or payload.get("digest_item_id") or ""
        digest_item = next((d for d in digest if d.get("id") == ref_id), None)
        body = f"{salutation}, "

        if kind == "research_digest" and digest_item:
            body += f"{digest_item.get('source', 'New research')} — {digest_item.get('title', '')}. Want me to pull the details?"
        elif kind == "regulation_change" and digest_item:
            body += f"Compliance: {digest_item.get('title', '')}. Deadline: {payload.get('deadline_iso', '')[:10]}. Want a checklist?"
        elif kind == "perf_dip":
            body += f"your {payload.get('metric', 'views')} dropped {abs(int(payload.get('delta_pct', 0)*100))}% this week. Want me to diagnose?"
        elif kind == "seasonal_perf_dip":
            body += f"views down {abs(int(payload.get('delta_pct', 0)*100))}% — normal seasonal dip. Focus retention. Want a plan?"
        elif kind == "perf_spike":
            body += f"nice — {payload.get('metric', 'views')} up {int(payload.get('delta_pct', 0)*100)}%! Want me to amplify?"
        elif kind == "renewal_due":
            body += f"plan expires in {payload.get('days_remaining', 0)} days. Reply YES to renew."
        elif kind == "ipl_match_today":
            body += f"{payload.get('match', 'IPL')} tonight. Sat matches = -12% covers. Push delivery. Want me to draft?"
        elif kind == "supply_alert":
            body += f"URGENT — {payload.get('molecule', '')} recall, batches {'+'.join(payload.get('affected_batches', [])[:2])}. Pull stock. Want affected customer list?"
        elif kind == "active_planning_intent":
            body += f"here's a draft for {payload.get('intent_topic', '').replace('_', ' ')}. Want me to refine?"
        elif kind == "curious_ask_due":
            body += f"Quick check — what's most in-demand this week? I'll make it into a Google post."
        elif kind == "review_theme_emerged":
            body += f'review pattern: "{payload.get("theme", "").replace("_", " ")}" ({payload.get("occurrences_30d", 0)}x). Want me to draft replies?'
        elif kind == "milestone_reached":
            body += f"you're at {payload.get('value_now', 0)} {payload.get('metric', '').replace('_', ' ')} — {payload.get('milestone_value', 0) - payload.get('value_now', 0)} from milestone! Want a post ready?"
        elif kind == "competitor_opened":
            body += f"{payload.get('competitor_name', 'competitor')} opened {payload.get('distance_km', 0)}km away. Want to see their profile?"
        elif kind == "winback_eligible":
            body += f"it's been {payload.get('days_since_expiry', 0)} days. Visibility down {abs(int(payload.get('perf_dip_pct', 0)*100))}%. Reply SHOW for comparison."
        elif kind == "gbp_unverified":
            body += f"GBP not verified — missing ~{int(payload.get('estimated_uplift_pct', 0)*100)}% of searches. 5-min fix. Want to do it now?"
        elif kind == "cde_opportunity" and digest_item:
            body += f"CDE tonight: {digest_item.get('title', '')}. {digest_item.get('credits', 0)} credits."
        elif kind == "dormant_with_vera":
            body += f"been {payload.get('days_since_last_merchant_message', 0)} days — anything I can help with? Even a quick review reply takes 2 min."
        elif kind == "category_seasonal":
            trends = payload.get("trends", [])[:3]
            body += f"seasonal shift: {', '.join(t.replace('_demand_', ' ').replace('+','↑').replace('-','↓') for t in trends)}. Want a customer broadcast?"
        elif kind == "festival_upcoming":
            body += f"{payload.get('festival', '')} in {payload.get('days_until', 0)} days. Want themed content?"
        else:
            body += f"update on {kind.replace('_', ' ')}. Want help acting on it?"

        return {"body": body.strip(), "cta": "open_ended", "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Fallback: {kind}"}


# Initialize composer
composer = VeraComposer()

# =============================================================================
# CONVERSATION HANDLER — LLM-guided with fast-path detection
# =============================================================================


class ConversationHandler:
    """Multi-turn handling: fast pattern detection + LLM for nuanced replies."""

    AUTO_REPLY = ["thank you for contacting", "our team will respond",
                  "we will get back to you", "your message has been received",
                  "this is an automated", "we are currently unavailable"]
    HOSTILE = ["stop", "unsubscribe", "spam", "don't message", "leave me alone",
               "useless", "waste of time", "block", "shut up"]
    COMMIT = ["yes", "ok let", "lets do it", "let's do it", "go ahead", "sure",
              "proceed", "do it", "confirm", "whats next", "what's next",
              "sounds good", "perfect", "ok done"]

    def handle_reply(self, conversation_id: str, merchant_id: str,
                     message: str, turn_number: int, from_role: str) -> dict:
        conversations.setdefault(conversation_id, [])
        conversations[conversation_id].append(
            {"turn": turn_number, "from": from_role, "message": message})

        msg = message.lower().strip()

        # 1. Auto-reply (fast, no LLM)
        if any(p in msg for p in self.AUTO_REPLY):
            count = sum(1 for t in conversations[conversation_id]
                        if any(p in t.get("message", "").lower() for p in self.AUTO_REPLY))
            if count >= 2:
                return {"action": "end",
                        "rationale": f"Auto-reply detected {count}x. Exiting."}
            return {"action": "wait", "wait_seconds": 1800,
                    "rationale": "Auto-reply detected. Waiting for human."}

        # 2. Hostile (fast, safety-critical)
        if any(h in msg for h in self.HOSTILE):
            return {"action": "end",
                    "body": "Understood — won't message again unless you reach out. Apologies for the inconvenience.",
                    "rationale": "Hostile detected. Graceful exit."}

        # 3. Commitment → action mode (use stricter action-mode prompt)
        if any(c in msg for c in self.COMMIT) or self._is_slot_pick(msg):
            if llm.available:
                result = self._llm_reply_action(conversation_id, merchant_id, message)
                if result:
                    return result
            # Detect if it's a booking/slot pick
            if self._is_slot_pick(msg):
                return {"action": "send",
                        "body": "Noted — your preferred slot has been shared with the team. They'll confirm shortly.",
                        "cta": None,
                        "rationale": "Slot pick detected. Acknowledged without fabricating booking."}
            return {"action": "send",
                    "body": "On it — I'll prepare this and share here in 5 min for your review.",
                    "cta": None,
                    "rationale": "Commitment detected. Action mode."}

        # 4. Try LLM for nuanced reply
        if llm.available:
            result = self._llm_reply(conversation_id, merchant_id, message)
            if result:
                return result

        # 5. Fallback
        if "?" in message:
            return {"action": "send",
                    "body": "Good question — let me check and get back to you shortly.",
                    "cta": None, "rationale": "Question detected."}
        return {"action": "send",
                "body": "Got it, thanks. Anything else I can help with?",
                "cta": None, "rationale": "Default acknowledgment."}

    def _llm_reply(self, conversation_id: str, merchant_id: str, message: str) -> Optional[dict]:
        history = conversations.get(conversation_id, [])
        merchant = get_merchant(merchant_id)
        hist_text = "\n".join(
            f"  Turn {t['turn']} ({t['from']}): {t['message'][:100]}"
            for t in history[-5:])

        prompt = f"""## CONTEXT
Merchant: {merchant.get('identity', {}).get('name', '?')} ({merchant.get('identity', {}).get('owner_first_name', '?')})
Category: {merchant.get('category_slug', '?')}
Turn: {len(history)}

## HISTORY
{hist_text}

## LATEST MESSAGE
"{message}"

Respond following the Reply Handler Guide. ONLY JSON."""

        response = llm.complete(REPLY_GUIDE, prompt)
        if not response:
            return None
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*?\}', response)
            if not match:
                return None
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return result if "action" in result else None

    def _llm_reply_action(self, conversation_id: str, merchant_id: str, message: str) -> Optional[dict]:
        """Stricter action-mode prompt that prevents qualifying and fabrication."""
        history = conversations.get(conversation_id, [])
        merchant = get_merchant(merchant_id)
        hist_text = "\n".join(
            f"  Turn {t['turn']} ({t['from']}): {t['message'][:100]}"
            for t in history[-5:])

        action_prompt = f"""## CONTEXT
Merchant: {merchant.get('identity', {}).get('name', '?')} ({merchant.get('identity', {}).get('owner_first_name', '?')})
Category: {merchant.get('category_slug', '?')}
Turn: {len(history)}

## HISTORY
{hist_text}

## LATEST MESSAGE (COMMITMENT/ACTION REQUEST)
"{message}"

## INSTRUCTIONS — ACTION MODE
The user has committed or requested an action. Respond with ACTION confirmation.
- Say what YOU will do next + give a timeline
- If they picked a slot/time: acknowledge it + say "team will confirm"
- NEVER ask qualifying questions
- NEVER say "I have booked" or "confirmed" — Vera cannot book/transact
- Keep under 300 chars

Return ONLY: {{"action": "send", "body": "<your reply>", "cta": null, "rationale": "<reason>"}}"""

        response = llm.complete(REPLY_GUIDE, action_prompt)
        if not response:
            return None
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*?\}', response)
            if not match:
                return None
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        if not result or "action" not in result:
            return None
        # Safety check: strip fabricated booking confirmations
        body = result.get("body", "")
        fabrication_phrases = ["i have booked", "appointment is confirmed", "booking confirmed",
                               "your appointment is set", "reservation confirmed"]
        if any(p in body.lower() for p in fabrication_phrases):
            result["body"] = "Noted — your preference is recorded. The team will confirm shortly."
        return result

    @staticmethod
    def _is_slot_pick(msg: str) -> bool:
        """Detect if message is a slot/time selection."""
        slot_patterns = [
            r'\b(book|slot|wed|thu|fri|sat|sun|mon|tue)\b.*\d',
            r'\breply\s*[12]\b', r'^\s*[12]\s*$',
            r'\b\d{1,2}\s*(am|pm)\b',
            r'\bbook\s+me\b', r'\bplease\s+book\b',
        ]
        return any(re.search(p, msg, re.IGNORECASE) for p in slot_patterns)


conversation_handler = ConversationHandler()

# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START),
            "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Enhanced",
        "team_members": ["AI Engineer"],
        "model": f"{llm.provider}/{llm.model}" if llm.available else "rule-based-fallback",
        "approach": "LLM-guided composition with strict Composition Guide (temp=0, seed=42). Deterministic. RAG-grounded on pushed contexts only.",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        raise HTTPException(status_code=400, detail={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"})

    key = (body.scope, body.context_id)
    if key in contexts:
        existing = contexts[key]["version"]
        if body.version < existing:
            return {"accepted": False, "reason": "stale_version",
                    "current_version": existing}
        if body.version == existing:
            return {"accepted": True,
                    "ack_id": f"ack_{body.context_id}_{body.version}",
                    "stored_at": datetime.utcnow().isoformat() + "Z"}

    contexts[key] = {"version": body.version, "payload": body.payload,
                     "delivered_at": body.delivered_at}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trigger_id in body.available_triggers:
        trigger = get_trigger(trigger_id)
        if not trigger:
            continue
        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")
        merchant = get_merchant(merchant_id)
        if not merchant:
            continue
        category = get_category_for_merchant(merchant)
        customer = get_customer(customer_id) if customer_id else None

        result = composer.compose(category, merchant, trigger, customer)
        if result and result.get("body"):
            conv_id = f"conv_{trigger_id}_{uuid.uuid4().hex[:8]}"
            actions.append({
                "conversation_id": conv_id,
                "trigger_id": trigger_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "body": result["body"],
                "cta": result.get("cta"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", "")
            })
            conversations[conv_id] = [{"turn": 1, "from": "bot",
                                       "message": result["body"]}]
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    return conversation_handler.handle_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        message=body.message,
        turn_number=body.turn_number,
        from_role=body.from_role)


# =============================================================================
# STANDALONE COMPOSE (for generating submission.jsonl)
# =============================================================================


def compose(category: dict, merchant: dict, trigger: dict,
            customer: dict | None = None) -> dict:
    return composer.compose(category, merchant, trigger, customer)
