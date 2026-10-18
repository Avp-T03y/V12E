"""Granite explanation layer: model numbers -> grounded plain-language summary + action.

explain_order(payload) always returns a valid result:
  * Granite (llama-server, OpenAI-style /v1/chat/completions) writes the wording;
  * a validator rejects ungrounded or overconfident output;
  * on timeout, bad JSON or a failed check, a template built from the numbers is used.
The recommended action is chosen in code from the runway tier; Granite only rephrases it.

payload = {
  "risk_score": 0.62, "risk_band": "high", "risk_percentile": 94.0,
  "runway_days": 74, "runway_tier": "medium",
  "typical_days_late_if_late": {"median": 12, "q25": 6, "q75": 25},
  "drivers": ["this vendor (X) has a late rate over the past year 9 points above the overall rate", ...],
  "context": {"country": ..., "vendor": ..., "fulfillment_route": ..., "product_group": ..., "quantity": ...}
}
"""
import json
import os
import re
import time

import requests

GRANITE_URL = os.environ.get("GRANITE_URL", "http://granite:8081")
GRANITE_TIMEOUT = float(os.environ.get("GRANITE_TIMEOUT", "20"))
GRANITE_MODEL = os.environ.get("GRANITE_MODEL", "granite")

ACTIONS = {
    "long": "Review sourcing: consider an alternative supplier or route.",
    "medium": "Expedite with the vendor or pre-position buffer stock.",
    "short": "Expedite now: pre-position from the nearest depot and warn the receiving facility.",
}
CAVEAT = ("Risk is a ranking against recent orders, not a guarantee; the delay range is a "
          "historical estimate.")
# causes the model never sees; mentioning them would be invented
BANNED_CAUSES = ["weather", "storm", "flood", "strike", "customs", "port", "holiday", "war",
                 "conflict", "fuel", "pandemic", "covid", "political", "shortage", "congestion"]
CERTAINTY = ["will be late", "will arrive late", "certainly", "guaranteed", "definitely", "100%"]

SYSTEM = """You explain delivery-delay risk to a logistics officer for a health supply chain.
Rules:
- Use ONLY the facts listed under DRIVERS and CONTEXT. Do not add any other cause
  (no weather, strikes, customs, ports, holidays, shortages, or anything not listed).
- Never state certainty: the risk is a ranking, and the delay range is a historical estimate.
- If a fact says "not recorded", say that it is unknown. Never guess a value.
- Keep risk_summary to 1-2 short sentences. Rephrase ACTION without changing its meaning.
Return JSON only, exactly: {"risk_summary": "...", "recommended_action": "...", "caveat": "..."}"""


def tier_for(runway_days):
    if runway_days is None:
        return None
    return "long" if runway_days > 120 else "medium" if runway_days >= 45 else "short"


def user_prompt(p):
    d = p["typical_days_late_if_late"]
    drivers = p.get("drivers") or ["no individual driver stood out for this order"]
    return (f"RISK: {p['risk_band']} (score {p['risk_score']:.2f}, above {p.get('risk_percentile', 0):.0f}% "
            f"of recent orders)\n"
            f"RUNWAY: {p['runway_days']} days until the scheduled delivery ({p['runway_tier']})\n"
            f"ACTION: {ACTIONS[p['runway_tier']]}\n"
            f"TYPICAL DELAY IF LATE: {d['median']:.0f} days (usually {d['q25']:.0f}-{d['q75']:.0f})\n"
            "DRIVERS:\n- " + "\n- ".join(drivers) + "\n"
            f"CONTEXT: {json.dumps(p.get('context', {}))}")


def ask_granite(p, url=None, timeout=None):
    r = requests.post(f"{url or GRANITE_URL}/v1/chat/completions", timeout=timeout or GRANITE_TIMEOUT,
                      json={"model": GRANITE_MODEL,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": user_prompt(p)}],
                            "temperature": 0, "max_tokens": 220,
                            "response_format": {"type": "json_object"}})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)          # tolerate stray text around the JSON
    return json.loads(m.group(0) if m else text)


def validate(out, p):
    """Raise ValueError if the output breaks the contract or the guardrails."""
    if not isinstance(out, dict) or set(out) != {"risk_summary", "recommended_action", "caveat"}:
        raise ValueError("wrong keys")
    if not all(isinstance(v, str) and v.strip() for v in out.values()):
        raise ValueError("empty field")
    text = " ".join(out.values()).lower()
    allowed = (" ".join(p.get("drivers") or []) + " " + json.dumps(p.get("context", {}))).lower()
    invented = [w for w in BANNED_CAUSES                      # whole words only ("war" != "warn")
                if re.search(rf"\b{w}(s|es|ing|ed)?\b", text) and not re.search(rf"\b{w}", allowed)]
    if invented:
        raise ValueError(f"invented cause: {invented}")
    if any(w in text for w in CERTAINTY):
        raise ValueError("claims certainty")
    if any("not recorded" in d for d in (p.get("drivers") or [])):
        if not any(w in text for w in ("unknown", "not recorded", "missing", "not available")):
            raise ValueError("did not say the lead time is unknown")
    action = out["recommended_action"].lower()
    keys = {"long": ["sourcing", "supplier", "route"], "medium": ["expedite", "buffer", "pre-position"],
            "short": ["expedite", "depot", "warn", "pre-position"]}[p["runway_tier"]]
    if not any(k in action for k in keys):
        raise ValueError("action does not match the runway tier")
    if len(out["risk_summary"]) > 400:
        raise ValueError("summary too long")
    return out


def template(p):
    drivers = [d for d in (p.get("drivers") or []) if "(lowers risk)" not in d]   # only reasons for concern
    why = ("Main factors: " + "; ".join(d.replace(" (raises risk)", "") for d in drivers[:2]) + ".") \
        if drivers else "No single factor stood out."
    return {"risk_summary": f"This order ranks {p['risk_band']} risk (above {p.get('risk_percentile', 0):.0f}% "
                            f"of recent orders). {why}",
            "recommended_action": ACTIONS[p["runway_tier"]],
            "caveat": CAVEAT}


_cache = {}


def explain_order(p, use_granite=True):
    """Always returns {risk_summary, recommended_action, caveat, source, latency_ms[, fallback_reason]}."""
    key = json.dumps(p, sort_keys=True, default=str)
    if key in _cache:
        return _cache[key]
    t0 = time.time()
    if use_granite:
        try:
            out = {**validate(ask_granite(p), p), "source": "granite"}
        except Exception as e:                       # timeout, connection, JSON, guardrail
            out = {**template(p), "source": "template", "fallback_reason": f"{type(e).__name__}: {e}"[:200]}
    else:
        out = {**template(p), "source": "template", "fallback_reason": "granite disabled"}
    out["latency_ms"] = round((time.time() - t0) * 1000)
    if len(_cache) > 2000:
        _cache.clear()
    _cache[key] = out
    return out
