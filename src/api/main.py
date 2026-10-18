"""V12 Engines scoring API (FastAPI).

  GET  /health    model loaded? version, backend
  POST /score     one order -> risk score, band, runway, drivers, Granite explanation
  GET  /alerts    scored orders still open, highest risk first (shorter runway first among ties)
  GET  /stream    Server-Sent Events: every scored order as it happens (the replay feed)
  GET  /metrics   the model's metrics.json (written by `v12e_model.py evaluate`)
  POST /deliveries  record a completed delivery, so later orders' past-performance features use it

A new order only needs its own fields (dates, quantity, country, vendor, route, ...). The
past-performance features (prior_excess_*, prior_known_*) and the planned lead time are computed
here from the delivery history, with the preprocessing notebook's exact rules
(src/features/history.py; tests/test_feature_parity.py proves they match the training tables).
Alerts use the live rolling rule: top 10% of orders placed in the previous 30 days.

Env: MODEL_DIR (dir written by train), MODEL_FILE (v12e_eval_model.pkl for the 2014-2015 replay,
v12e_model.pkl for new orders), HISTORY_CSV (raw SCMS CSV), GRANITE_URL, V12E_API_KEY (optional:
when set, every endpoint except /health requires the X-API-Key header; nginx adds it).
"""
import asyncio
import json
import os
import platform
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import v12e_model as vm  # noqa: E402
from src.features.history import DeliveryHistory, planned_lead_time  # noqa: E402
from src.genai.explainer import explain_order, tier_for  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
MODEL_FILE = os.environ.get("MODEL_FILE", "v12e_eval_model.pkl")
HISTORY_CSV = os.environ.get("HISTORY_CSV", str(ROOT / "preprocessing" / "SCMS_Delivery_History_Dataset_20150929.csv"))
STATE_DIR = os.environ.get("STATE_DIR", "")              # writable; the review log survives restarts here
# Snap ML only: convert the loaded model's trees to the fastest inference format ("auto" picks the
# NNPA/zDNN format when the accelerator is available). Off by default; /health reports the result.
OPTIMIZE_TREES = os.environ.get("OPTIMIZE_TREES", "0") == "1"
API_KEY = os.environ.get("V12E_API_KEY", "")
USE_GRANITE = os.environ.get("USE_GRANITE", "1") == "1"
STATE = {"art": None, "error": None, "alerts": {}, "subscribers": set(), "history": None, "policy": None,
         "lat_model": deque(maxlen=5000), "lat_total": deque(maxlen=5000), "lat_granite": deque(maxlen=5000),
         "granite_check": (0.0, None), "scored": 0, "started": time.time(), "reviews": {}}


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def system_facts():
    """Hardware and runtime facts readable inside the container (no Docker socket). Cached."""
    if STATE.get("facts") is not None:
        return STATE["facts"]
    import re
    import shutil
    from importlib import metadata
    cpuinfo = _read("/proc/cpuinfo")
    machine_type, nnpa = None, None
    if cpuinfo is not None:
        m = re.search(r"machine\s*[=:]\s*(\w+)", cpuinfo)              # s390x: "... machine = 9175"
        machine_type = m.group(1) if m else None
        flags = set()
        for line in cpuinfo.splitlines():
            key = line.split(":", 1)[0].strip()
            if key in ("features", "flags") and ":" in line:            # s390x "features", x86 "flags"
                flags.update(line.split(":", 1)[1].split())
        nnpa = "nnpa" in flags
    meminfo = _read("/proc/meminfo") or ""
    mm = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
    cpu_max = (_read("/sys/fs/cgroup/cpu.max") or "").split()          # cgroup v2: "<quota> <period>" or "max ..."
    mem_max = (_read("/sys/fs/cgroup/memory.max") or "").strip()
    try:
        snapml_version = metadata.version("snapml")
    except Exception:
        snapml_version = None
    try:
        disk_total_gb = round(shutil.disk_usage("/").total / 1024 ** 3, 1)
    except OSError:
        disk_total_gb = None
    STATE["facts"] = {
        "machine_type": machine_type,                                   # 9175 = IBM z17
        "nnpa": nnpa,                                                   # accelerator AVAILABLE (CPU flag)
        "cpus": os.cpu_count(),
        "cpu_limit": round(int(cpu_max[0]) / int(cpu_max[1]), 2) if len(cpu_max) == 2 and cpu_max[0] != "max" else None,
        "mem_total_gb": round(int(mm.group(1)) / 1024 ** 2, 1) if mm else None,
        "mem_limit_gb": round(int(mem_max) / 1024 ** 3, 2) if mem_max.isdigit() else None,
        "disk_total_gb": disk_total_gb,                                 # root filesystem as seen by the container
        "snapml_version": snapml_version,
    }
    return STATE["facts"]


def review_log():
    return Path(STATE_DIR) / "reviews.jsonl" if STATE_DIR else None


def load_reviews():
    """Officer reviews from earlier runs (last entry per order wins)."""
    path = review_log()
    if not path or not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            STATE["reviews"][r["order_id"]] = {k: r[k] for k in ("status", "reviewed_by", "reviewed_at")}
        except Exception:
            continue


def latency_stats(values):
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 2)  # noqa: E731
    return {"n": len(v), "median_ms": pick(0.5), "p95_ms": pick(0.95), "max_ms": round(v[-1], 2)}


def granite_status():
    """Is the Granite server reachable? Checked at most every 30 s."""
    checked, status = STATE["granite_check"]
    if time.time() - checked < 30 and status is not None:
        return status
    import requests
    from src.genai import explainer
    try:
        r = requests.get(f"{explainer.GRANITE_URL}/health", timeout=1.5)
        status = "ok" if r.ok else f"http {r.status_code}"
    except Exception as e:
        status = f"unreachable ({type(e).__name__})"
    STATE["granite_check"] = (time.time(), status)
    return status


@asynccontextmanager
async def lifespan(app):
    try:
        STATE["art"] = vm.load_artifact(MODEL_DIR, MODEL_FILE)
        STATE["policy"] = vm.policy_from_artifact(STATE["art"])
        STATE["history"] = DeliveryHistory.from_scms(HISTORY_CSV)
        load_reviews()
        if OPTIMIZE_TREES and STATE["art"]["backend"] == "snapml":
            STATE["art"]["tree_format"] = STATE["art"]["model"].optimize_for_inference()
    except Exception as e:                  # keep serving /health with the reason
        STATE["error"] = f"{type(e).__name__}: {e}"
        STATE["art"] = None
    yield


app = FastAPI(title="V12 Engines scoring API", lifespan=lifespan)


def error(status, code, message):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    code, msg = exc.detail if isinstance(exc.detail, tuple) else ("error", str(exc.detail))
    return error(exc.status_code, code, msg)


@app.exception_handler(Exception)
async def unhandled(request, exc):
    return error(500, "internal_error", f"{type(exc).__name__}")


@app.middleware("http")
async def api_key(request: Request, call_next):
    if API_KEY and request.url.path != "/health" and request.headers.get("x-api-key") != API_KEY:
        return error(401, "unauthorized", "missing or wrong X-API-Key")
    return await call_next(request)


def model():
    if STATE["art"] is None:
        raise HTTPException(503, ("model_not_loaded", STATE["error"] or "model not loaded"))
    return STATE["art"]


class Order(BaseModel):
    order_id: str
    decision_date: Optional[str] = None
    scheduled_delivery_date: str
    quantity: float
    planned_lead_time_days: Optional[float] = None      # computed from the dates when omitted
    # past-performance features: computed from the delivery history when omitted
    prior_excess_country: Optional[float] = None
    prior_excess_vendor: Optional[float] = None
    prior_excess_fulfillment_route: Optional[float] = None
    prior_known_country: Optional[int] = None
    prior_known_vendor: Optional[int] = None
    prior_known_fulfillment_route: Optional[int] = None
    country: str
    vendor: str
    fulfillment_route: str
    shipment_mode: Optional[str] = None
    product_group: Optional[str] = None
    as_of: Optional[str] = None             # "today" for the runway; the replay sends the order date
    explain: bool = True


@app.get("/health")
def health():
    art = STATE["art"]
    if art is None:
        return error(503, "model_not_loaded", STATE["error"] or "loading")
    return {"status": "ok", "model_version": art.get("model_version"), "role": art.get("role"),
            "backend": art["backend"], "trained_on": art["trained_on"], "open_alerts": len(STATE["alerts"]),
            "history_deliveries": len(STATE["history"].d), "alert_rule": "top 10% of the previous 30 days",
            "machine": platform.machine(), "python": platform.python_version(),
            "model_file": MODEL_FILE, "granite": granite_status() if USE_GRANITE else "disabled",
            "scored_since_start": STATE["scored"], "uptime_s": round(time.time() - STATE["started"]),
            "latency": {"model_ms": latency_stats(STATE["lat_model"]),
                        "granite_ms": latency_stats(STATE["lat_granite"]),
                        "score_total_ms": latency_stats(STATE["lat_total"])},
            **system_facts(),
            # inference tree format of the LOADED model, one per seed member. None = Snap ML's default
            # format; the accelerator counts as used only when the format is zDNN (NNPA) tensors.
            "tree_format": art.get("tree_format"),
            "nnpa_used_by_model": any("zdnn" in str(f).lower() for f in (art.get("tree_format") or []))}


@app.post("/score")
async def score(order: Order):
    t0 = time.perf_counter()
    art = model()
    try:
        sched = pd.Timestamp(order.scheduled_delivery_date)
        decided = pd.Timestamp(order.decision_date) if order.decision_date else None
    except Exception:
        raise HTTPException(422, ("invalid_date", "dates must be YYYY-MM-DD"))
    if decided is None or decided > sched:
        raise HTTPException(422, ("invalid_decision_date",
                                  "decision_date is required and must not be after scheduled_delivery_date"))
    today = pd.Timestamp(order.as_of) if order.as_of else pd.Timestamp(date.today())
    # In the source data some orders (all Direct Drop) have the promised date equal to the order date,
    # so a runway is meaningless for them: report it as unavailable rather than 0 days.
    runway_known = decided.normalize() != sched.normalize()
    runway = int((sched - today).days) if runway_known else None

    df = pd.DataFrame([order.model_dump(exclude={"explain", "as_of"})])
    lane_cols = [c for c in vm.RELATIVE + vm.PRIOR_FLAGS]
    if any(pd.isna(df.at[0, c]) for c in lane_cols):
        live = STATE["history"].features(df)                 # cutoff = decision date (<= scheduled)
        for c in lane_cols:
            df[c] = live[c].values
        features_source = "computed_from_history"
    else:
        features_source = "provided"
    if pd.isna(df.at[0, "planned_lead_time_days"]):
        df["planned_lead_time_days"] = planned_lead_time(df).values
    s = float(vm.score_frame(art, df)[0])
    band, pct, window_n, basis = STATE["policy"].rank(s, decided)
    STATE["policy"].add(order.order_id, decided, s)
    drivers = vm.explain(art, df)[0]
    d = art["days_late_if_late"]
    tier = "unknown" if runway is None else tier_for(runway) if runway >= 0 else "short"
    result = {
        "order_id": order.order_id, "model_version": art.get("model_version"),
        "risk_score": round(s, 4), "risk_band": band, "risk_percentile": pct,
        "alert_basis": {"rule": basis, "window_orders": window_n},
        "features": {c: (None if pd.isna(df.at[0, c]) else float(df.at[0, c]))
                     for c in ["planned_lead_time_days"] + vm.RELATIVE + vm.PRIOR_FLAGS},
        "features_source": features_source,
        "runway_days": runway, "runway_tier": tier,
        "runway_note": None if runway_known else "not available: promised date equals the order date in the source data",
        "typical_days_late_if_late": d,
        "top_drivers": drivers,
        "context": {k: getattr(order, k) for k in ("country", "vendor", "fulfillment_route", "shipment_mode",
                                                   "product_group", "quantity")},
        "decision_date": order.decision_date, "scheduled_delivery_date": order.scheduled_delivery_date,
        "as_of": str(today.date()),
    }
    STATE["lat_model"].append((time.perf_counter() - t0) * 1000)   # features + score + rank + drivers
    if order.explain and band != "normal":     # Granite only for orders worth an officer's time
        payload = {k: result[k] for k in ("risk_score", "risk_band", "risk_percentile", "runway_days",
                                          "runway_tier", "typical_days_late_if_late", "context")}
        payload["drivers"] = [x["text"] if x["direction"] == "unknown" else f"{x['text']} ({x['direction']})"
                              for x in drivers]
        result["explanation"] = await asyncio.to_thread(explain_order, payload, USE_GRANITE)
        if result["explanation"].get("source") == "granite":
            STATE["lat_granite"].append(float(result["explanation"].get("latency_ms", 0)))
    else:
        result["explanation"] = None
    if band != "normal":
        prev = STATE["alerts"].get(order.order_id) or STATE["reviews"].get(order.order_id, {})
        STATE["alerts"][order.order_id] = {**result, "status": prev.get("status", "open"),
                                           **{k: prev[k] for k in ("reviewed_by", "reviewed_at") if k in prev}}
    STATE["scored"] += 1
    STATE["lat_total"].append((time.perf_counter() - t0) * 1000)
    await publish(result)
    return result


class Delivery(BaseModel):
    order_id: str
    country: str
    vendor: str
    fulfillment_route: str
    delivered_date: str
    scheduled_delivery_date: str


@app.post("/deliveries")
def record_delivery(d: Delivery):
    model()
    late = int(pd.Timestamp(d.delivered_date) > pd.Timestamp(d.scheduled_delivery_date))
    STATE["history"].add(d.order_id, d.country, d.vendor, d.fulfillment_route, d.delivered_date, late)
    return {"recorded": d.order_id, "late": late, "history_deliveries": len(STATE["history"].d)}


@app.get("/alerts")
def alerts(limit: int = 50, band: Optional[str] = None):
    rows = [a for a in STATE["alerts"].values() if band is None or a["risk_band"] == band]
    rows.sort(key=lambda a: (-a["risk_score"], a["runway_days"] if a["runway_days"] is not None else 10 ** 6))
    return {"count": len(rows), "alerts": rows[:limit]}


@app.post("/alerts/{order_id}/status")
def set_status(order_id: str, status: str, officer: str = "officer"):
    if status not in ("acknowledged", "action_taken", "dismissed", "open"):
        raise HTTPException(422, ("invalid_status", "status must be acknowledged/action_taken/dismissed/open"))
    if order_id not in STATE["alerts"]:
        raise HTTPException(404, ("unknown_alert", order_id))
    review = {"status": status, "reviewed_by": officer,
              "reviewed_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    STATE["alerts"][order_id].update(review)
    STATE["reviews"][order_id] = review
    path = review_log()
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps({"order_id": order_id, **review}) + "\n")
        except OSError:
            pass                                  # read-only or missing dir: keep the review in memory
    return STATE["alerts"][order_id]


@app.get("/metrics")
def metrics():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(503, ("metrics_missing", "run `v12e_model.py evaluate` first"))
    return json.loads(path.read_text())


async def publish(event):
    for q in list(STATE["subscribers"]):
        if q.qsize() < 500:
            q.put_nowait(event)


@app.get("/stream")
async def stream(request: Request):
    q = asyncio.Queue()
    STATE["subscribers"].add(q)

    async def gen():
        try:
            yield ": connected\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: scored\ndata: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            STATE["subscribers"].discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
