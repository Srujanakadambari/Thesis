"""
Thesis — LLM Visualization Benchmark Dashboard
All 18 architectural scenarios + caching + parallel execution.

Run with:
    cd "FINAL FOLDER"
    streamlit run app.py
"""

import os
import sys
import io
import json
import time
import hashlib
import re
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, namedtuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import instructor
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Phoenix tracing (optional — app works without it)
try:
    import phoenix as px
    from phoenix.otel import register as _phoenix_register
    from openinference.instrumentation.openai import OpenAIInstrumentor as _OAIInstrumentor
    _PHOENIX_AVAILABLE = True
except ImportError:
    _PHOENIX_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BRANCH_ROOT  = PROJECT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(BRANCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _env in [PROJECT_ROOT / ".env",
             PROJECT_ROOT.parent / ".env",
             PROJECT_ROOT.parent.parent / ".env"]:
    if _env.exists():
        load_dotenv(_env, override=True)
        break

from retrieve_data import retrieve_data
from prompts.default import (
    DATA_ANALYSIS_PROMPT,
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_TYPE_JUSTIFICATION_PROMPT,
    SYSTEM_PROMPT,
)
from response_models.default import (
    VisualizationConfig, DataAnalysis, ChartTypeJustification, ChartType,
)

# ── Scenario definitions ──────────────────────────────────────────────────────
# ScenarioCfg: router_model | analyze_model | extract_model | justify_model | group label
ScenarioCfg = namedtuple("ScenarioCfg",
    ["router_model", "analyze_model", "extract_model", "justify_model", "group"])

SCENARIOS = OrderedDict([
    # ── S0: Baseline — full 4-step pipeline, single model (o4-mini) ──────────
    ("S0",  ("S0",  ScenarioCfg("o4-mini",     "o4-mini",     "o4-mini",     "o4-mini",     "Baseline · Full Pipeline"))),
    # ── Group 1: No Router, Single Model (o4-mini) ────────────────────────────
    ("S1",  ("S1",  ScenarioCfg(None,           "o4-mini",     "o4-mini",     "o4-mini",     "No-Router · Single-Model"))),
    ("S2",  ("S2",  ScenarioCfg(None,           "o4-mini",     "o4-mini",     None,          "No-Router · Single-Model"))),
    ("S3",  ("S3",  ScenarioCfg(None,           None,          "o4-mini",     None,          "No-Router · Single-Model"))),
    # ── Group 2: No Router, Mixed Models ─────────────────────────────────────
    ("S4",  ("S4",  ScenarioCfg(None,           "o4-mini",     "gpt-4o-mini", "gpt-4o-mini", "No-Router · Mixed-Model"))),
    ("S4b", ("S4b", ScenarioCfg(None,           "gpt-4o",      "gpt-4o-mini", "gpt-4o-mini", "No-Router · Mixed-Model"))),
    ("S5",  ("S5",  ScenarioCfg(None,           "o4-mini",     "gpt-4o-mini", None,          "No-Router · Mixed-Model"))),
    ("S5b", ("S5b", ScenarioCfg(None,           "gpt-4o",      "gpt-4o-mini", None,          "No-Router · Mixed-Model"))),
    ("S6",  ("S6",  ScenarioCfg(None,           None,          "gpt-4o-mini", None,          "No-Router · Mixed-Model"))),
    # ── Group 3: No Router, Reasoning Models ─────────────────────────────────
    ("S7",  ("S7",  ScenarioCfg(None,           "o3",          "gpt-4o-mini", "gpt-4o-mini", "No-Router · Reasoning"))),
    ("S8",  ("S8",  ScenarioCfg(None,           "o3",          "gpt-4o-mini", None,          "No-Router · Reasoning"))),
    ("S9",  ("S9",  ScenarioCfg(None,           None,          "gpt-4o",      None,          "No-Router · Reasoning"))),
    # ── Group 4: With Router ──────────────────────────────────────────────────
    ("SA1", ("SA1", ScenarioCfg("o4-mini",      "o4-mini",     "o4-mini",     "o4-mini",     "Router"))),
    ("SA2", ("SA2", ScenarioCfg("o4-mini",      "o4-mini",     "gpt-4o-mini", "gpt-4o-mini", "Router"))),
    ("SA3", ("SA3", ScenarioCfg("gpt-4o-mini",  "gpt-4o",      "gpt-4o-mini", "gpt-4o-mini", "Router"))),
    ("SA4", ("SA4", ScenarioCfg("gpt-4o-mini",  "gpt-4o",      "gpt-4o-mini", None,          "Router"))),
    ("SA5", ("SA5", ScenarioCfg("gpt-4o-mini",  None,          "gpt-4o-mini", None,          "Router"))),
    ("SA6", ("SA6", ScenarioCfg("gpt-4o",       "gpt-4o",      "gpt-4o-mini", None,          "Router"))),
    # ── Group 5: Qwen (local vLLM) ───────────────────────────────────────────
    ("SQ0", ("SQ0", ScenarioCfg("qwen",         "qwen",        "qwen",        "qwen",        "Qwen · Full Pipeline"))),
    ("SQ1", ("SQ1", ScenarioCfg(None,           "qwen",        "qwen",        "qwen",        "Qwen · No-Router"))),
])

# Custom baseline models available for single-step extract benchmarking
BASELINE_MODELS = [
    ("gpt-3.5-turbo",  "GPT-3.5 Turbo"),
    ("gpt-4o-mini",    "GPT-4o Mini"),
    ("gpt-4o",         "GPT-4o"),
    ("gpt-4.1-mini",   "GPT-4.1 Mini"),
    ("gpt-4.1-nano",   "GPT-4.1 Nano"),
    ("o4-mini",        "o4-mini"),
    ("o3-mini",        "o3-mini"),
]

def scenario_label(sid):
    if sid.startswith("CB-"):
        model_id = sid[3:]
        display  = next((d for m, d in BASELINE_MODELS if m == model_id), model_id)
        return f"CB · Analyze→Extract→Justify[{display}]"
    sid_str, cfg = SCENARIOS[sid]
    steps = []
    if cfg.router_model:
        steps.append(f"Router[{cfg.router_model}]")
    if cfg.analyze_model:
        steps.append(f"Analyze[{cfg.analyze_model}]")
    if cfg.justify_model:
        steps.append(f"Extract[{cfg.extract_model}]→Justify[{cfg.justify_model}]")
    else:
        steps.append(f"Extract[{cfg.extract_model}]")
    return f"{sid} · " + "→".join(steps)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR     = PROJECT_ROOT / "bench_cache"
CHARTS_DIR    = CACHE_DIR / "charts"
LOG_FILE      = CACHE_DIR / "query_log.json"
SEM_THRESHOLD        = 0.88
ROUTER_PREVIEW_CHARS = 500   # chars sent to router when input pruning is ON

QWEN_BASE_URL    = "http://hal9000.skim.th-owl.de:11877/v1"
QWEN_PREFERRED   = [
    "qwen3-30b-thinking", "qwen3-30b",
    "Qwen3-30B-A3B-Instruct", "qwen3-8b", "Qwen3.6-27B",
]
REASONING_MODELS = {"o1", "o1-mini", "o3", "o3-mini", "o4-mini"}

_qwen_model_cache: str | None = None

def get_qwen_model_name() -> str:
    """Discover the available model on the Qwen server; cache the result."""
    global _qwen_model_cache
    if _qwen_model_cache:
        return _qwen_model_cache
    try:
        client    = OpenAI(base_url=QWEN_BASE_URL, api_key="dummy", timeout=8)
        available = [m.id for m in client.models.list().data]
        for m in QWEN_PREFERRED:
            if m in available:
                _qwen_model_cache = m
                return m
        if available:
            _qwen_model_cache = available[0]
            return _qwen_model_cache
    except Exception:
        pass
    _qwen_model_cache = QWEN_PREFERRED[-1]
    return _qwen_model_cache

PALETTE = ["#5b9cf6", "#9c7af7", "#f0a855", "#4caf7d", "#ef6b6b",
           "#64b5f6", "#81c784", "#ffb74d", "#ba68c8"]

STAGE_COLORS = {
    "router":         "#2a3a6a",
    "analyze":        "#9c7af7",
    "extract":        "#f0a855",
    "justify":        "#4caf7d",
    "extract∥justify":"#e8a040",
    "cache_lookup":   "#2a2a5a",
    "judge":          "#ef6b6b",
    "render":         "#555566",
}

RESULTS_DIR   = PROJECT_ROOT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Router tool definition ────────────────────────────────────────────────────
ROUTER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "generate_visualization",
        "description": "Analyze the data and generate a visualization",
        "parameters": {
            "type": "object",
            "properties": {
                "data":       {"type": "string", "description": "The data as a markdown table"},
                "user_query": {"type": "string", "description": "The user's query"},
            },
            "required": ["data", "user_query"],
        },
    },
}]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Viz Benchmark · 18 Scenarios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0f14; color: #e8e8f0; }
.metric-card { background:#1a1a24; border:1px solid #2a2a3a; border-radius:10px; padding:12px 14px; text-align:center; }
.metric-value { font-family:'DM Mono',monospace; font-size:1.4rem; font-weight:500; color:#7eb8f7; line-height:1.1; }
.metric-label { font-size:0.63rem; color:#777; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.badge-hit  { background:#1a3a1a; color:#4caf50; border:1px solid #4caf50; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.75rem; }
.badge-miss { background:#3a1a1a; color:#ef5350; border:1px solid #ef5350; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.75rem; }
.badge-pass { background:#1a3a1a; color:#4caf50; border:1px solid #4caf50; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.75rem; }
.badge-fail { background:#3a1a1a; color:#ef5350; border:1px solid #ef5350; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.75rem; }
.section-title { font-size:0.63rem; text-transform:uppercase; letter-spacing:0.12em; color:#555; margin-bottom:8px; font-family:'DM Mono',monospace; }
.reasoning-box { background:#13131d; border-left:3px solid #7eb8f7; border-radius:0 8px 8px 0; padding:12px 16px; font-size:0.85rem; color:#c0c0d0; line-height:1.6; }
.scenario-header { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:8px; padding:8px 14px; margin-bottom:10px; font-family:'DM Mono',monospace; font-size:0.82rem; color:#aac; }
div[data-testid="stSidebar"] { background:#0c0c12; border-right:1px solid #1e1e2e; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ═════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL = "text-embedding-3-small"

class OpenAIEmbedder:
    """Wraps OpenAI text-embedding-3-small with the same .encode() interface
    previously provided by SentenceTransformer."""
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return np.array([d.embedding for d in resp.data])

def get_clients(api_key: str):
    """Returns (instructor_client, raw_openai_client)."""
    base = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
    return instructor.from_openai(base), base

def load_embedding_model(api_key: str):
    return OpenAIEmbedder(api_key)

def get_qwen_client():
    return OpenAI(base_url=QWEN_BASE_URL, api_key="dummy", timeout=300)


@st.cache_resource
def start_phoenix():
    """Connect to a running Phoenix server and instrument OpenAI calls.
    Reads PHOENIX_COLLECTOR_ENDPOINT from env (default: http://localhost:6006).
    Returns (ui_url, error_msg) — error_msg is None on success."""
    if not _PHOENIX_AVAILABLE:
        return None, "not_installed"
    try:
        base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").strip().rstrip("/")
        traces_endpoint = base + "/v1/traces"
        ui_url = base.replace("/v1/traces", "")

        # Quick reachability check before registering
        import urllib.request
        try:
            urllib.request.urlopen(base, timeout=3)
        except Exception as e:
            return None, f"Phoenix not reachable at {base}\nStart it with: python -m phoenix.server.main serve"

        tp = _phoenix_register(
            project_name="viz-benchmark",
            endpoint=traces_endpoint,
            set_global_tracer_provider=False,
        )
        _OAIInstrumentor().instrument(tracer_provider=tp)
        return ui_url, None
    except Exception as e:
        return None, str(e)


# ═════════════════════════════════════════════════════════════════════════════
# CACHE
# ═════════════════════════════════════════════════════════════════════════════

def load_cache_index():
    p = CACHE_DIR / "cache_index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

def save_cache_index(index):
    (CACHE_DIR / "cache_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

def semantic_cache_lookup(query, encoder, threshold, scenario_id=None):
    index = load_cache_index()
    # Only compare entries from the same scenario so SA5 never returns an S0 hit
    if scenario_id:
        index = [e for e in index if e.get("scenario") == scenario_id]
    if not index:
        return None, 0.0
    q_emb = encoder.encode([query])
    sims  = cosine_similarity(q_emb, np.array([e["embedding"] for e in index]))[0]
    best  = int(np.argmax(sims))
    score = float(sims[best])
    return (index[best], score) if score >= threshold else (None, score)

def save_to_cache(query, cfg, chart_path, verdict, score,
                  latency, scenario_id, encoder, judge_result=None):
    index = load_cache_index()
    emb   = encoder.encode([query])[0].tolist()
    entry = {
        "id":          hashlib.md5(f"{query}::{scenario_id}".encode()).hexdigest()[:10],
        "query":       query,
        "embedding":   emb,
        "config_path": str(chart_path).replace(".png", ".json"),
        "chart_path":  str(chart_path),
        "verdict":     verdict,
        "score":       score,
        "latency_s":   round(latency, 3),
        "scenario":    scenario_id,
        "timestamp":   datetime.now().isoformat(),
        "reasoning":        (judge_result or {}).get("evaluation_statement", ""),
        "reasoning_steps":  (judge_result or {}).get("reasoning_steps", []),
        "strengths":        (judge_result or {}).get("main_strengths", []),
        "weaknesses":       (judge_result or {}).get("main_weaknesses", []),
        "improvement":      (judge_result or {}).get("suggested_improvement", ""),
    }
    Path(entry["config_path"]).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    index.append(entry)
    save_cache_index(index)

def _purge_cache_for_combo(query, sid):
    """Remove the exact cache entry for query+sid so the combined bench gets a genuine miss."""
    entry_id = hashlib.md5(f"{query}::{sid}".encode()).hexdigest()[:10]
    index    = load_cache_index()
    filtered = [e for e in index if e.get("id") != entry_id]
    if len(filtered) < len(index):
        save_cache_index(filtered)

def log_query(query, scenario, cache_hit, similarity, latency, verdict):
    log = json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else []
    log.append({
        "ts":         datetime.now().isoformat()[:16],
        "query":      query[:55],
        "scenario":   scenario,
        "cache_hit":  cache_hit,
        "similarity": round(similarity, 3),
        "latency_s":  round(latency, 3),
        "verdict":    verdict,
    })
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE STEP FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _is_qwen(model: str) -> bool:
    return model.lower().startswith("qwen") or model in QWEN_PREFERRED

def _qwen_kwargs(model: str) -> dict:
    """Return Qwen-specific sampling params; empty dict for all other models."""
    if _is_qwen(model):
        return {
            "temperature":      0.7,
            "top_p":            0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    return {}

def _extract_usage(completion):
    """Pull token counts from a raw OpenAI completion object."""
    u = getattr(completion, "usage", None)
    if not u:
        return {}
    return {
        "prompt_tokens":     getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens":      getattr(u, "total_tokens", 0) or 0,
    }


def step_router(raw_client, model, query, md_table):
    """Router LLM call — returns (tool_was_called, elapsed_s, tokens_dict)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{query}\n\nData (preview):\n{md_table[:ROUTER_PREVIEW_CHARS]}"},
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        tools=ROUTER_TOOLS,
        tool_choice={"type": "function", "function": {"name": "generate_visualization"}},
    )
    if model in REASONING_MODELS or _is_qwen(model):
        kwargs["tool_choice"] = "required"
    kwargs.update(_qwen_kwargs(model))
    t0   = time.perf_counter()
    resp = raw_client.chat.completions.create(**kwargs)
    return bool(resp.choices[0].message.tool_calls), time.perf_counter() - t0, _extract_usage(resp)

def step_analyze(inst_client, model, data, query):
    """analyze_data step — returns (analysis_str, elapsed_s, tokens_dict)."""
    prompt = DATA_ANALYSIS_PROMPT.format(data=data, query=query)
    t0     = time.perf_counter()
    try:
        resp, completion = inst_client.chat.completions.create_with_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=DataAnalysis,
            **_qwen_kwargs(model),
        )
        tokens = _extract_usage(completion)
    except AttributeError:
        resp   = inst_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=DataAnalysis,
            **_qwen_kwargs(model),
        )
        tokens = {}
    return resp.analysis, time.perf_counter() - t0, tokens

def step_extract(inst_client, model, data, analysis):
    """extract_chart_config step — returns (cfg_dict, elapsed_s, tokens_dict)."""
    prompt = CHART_CONFIGURATION_PROMPT.format(data=data, analysis=analysis)
    t0     = time.perf_counter()
    try:
        resp, completion = inst_client.chat.completions.create_with_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=VisualizationConfig,
            **_qwen_kwargs(model),
        )
        tokens = _extract_usage(completion)
    except AttributeError:
        resp   = inst_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=VisualizationConfig,
            **_qwen_kwargs(model),
        )
        tokens = {}
    res = resp.model_dump()
    res["charttype"] = res["charttype"].value
    return res, time.perf_counter() - t0, tokens

def step_justify(inst_client, model, data, analysis):
    """justify_chart_type step — returns (charttype_str, elapsed_s, tokens_dict)."""
    charttypes = {ct.name for ct in ChartType}
    prompt     = CREATE_CHART_TYPE_JUSTIFICATION_PROMPT.format(
        charttypes=charttypes, analysis=analysis, data=data)
    t0   = time.perf_counter()
    try:
        resp, completion = inst_client.chat.completions.create_with_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=ChartTypeJustification,
            **_qwen_kwargs(model),
        )
        tokens = _extract_usage(completion)
    except AttributeError:
        resp   = inst_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=ChartTypeJustification,
            **_qwen_kwargs(model),
        )
        tokens = {}
    return resp.chart_type.value, time.perf_counter() - t0, tokens


def detect_requested_chart_type(query: str):
    """
    Return an explicit chart type if the user named one in the query, else None.
    This overrides model judgment so 'show me a pie chart' always produces pie.
    Order matters: check most-specific phrases first.
    """
    q = query.lower()
    if "stacked bar" in q or "stackedbar" in q:
        return "stackedbar"
    if "pie chart" in q or "pie graph" in q or " pie " in q or q.startswith("pie ") or q.endswith(" pie"):
        return "pie"
    if "scatter plot" in q or "scatter chart" in q or "scatter " in q:
        return "scatter"
    if "bar chart" in q or "bar graph" in q or "bar plot" in q:
        return "bar"
    if "line chart" in q or "line graph" in q or "line plot" in q:
        return "line"
    return None


def _merge_tokens(acc, new):
    """Add token counts from new into accumulator dict."""
    for k, v in new.items():
        acc[k] = acc.get(k, 0) + v


def _inst_for(model, inst_client):
    """Return the right instructor client for the given model name."""
    if _is_qwen(model):
        return instructor.from_openai(get_qwen_client(), mode=instructor.Mode.JSON)
    return inst_client

def warmup_models(scenario_ids: list[str], api_key: str):
    """Send a minimal untimed request to every OpenAI model used across the
    selected scenarios so the server is warm before the real benchmark starts.
    Qwen / local models are skipped — they are always warm on that server."""
    models_to_warm = set()
    for sid in scenario_ids:
        if sid.startswith("CB-"):
            models_to_warm.add(sid[3:])
        elif sid in SCENARIOS:
            _, cfg = SCENARIOS[sid]
            for m in (cfg.router_model, cfg.analyze_model,
                      cfg.extract_model, cfg.justify_model):
                if m and not _is_qwen(m):
                    models_to_warm.add(m)
    if not models_to_warm:
        return
    raw = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
    for model in models_to_warm:
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1,
            )
            if model not in REASONING_MODELS:
                kwargs["max_tokens"] = 1
                del kwargs["max_completion_tokens"]
            raw.chat.completions.create(**kwargs)
        except Exception:
            pass

def _raw_for(model, raw_client):
    """Return the right raw OpenAI client for the given model name."""
    if _is_qwen(model):
        return get_qwen_client()
    return raw_client

def run_pipeline(sid, query, md_table, use_parallel, inst_client, raw_client):
    """
    Generic runner for all 20 scenarios (including Qwen).
    Returns (cfg_dict, errors_list, timings_dict, token_counts_dict).
    """
    _, cfg   = SCENARIOS[sid]
    timings  = {}
    errors   = []
    tok_all  = {}

    # ── Router ───────────────────────────────────────────────────────────────
    if cfg.router_model:
        _, t, tok = step_router(
            _raw_for(cfg.router_model, raw_client), cfg.router_model, query, md_table)
        timings["router"] = t
        _merge_tokens(tok_all, tok)

    # ── Analyze ──────────────────────────────────────────────────────────────
    if cfg.analyze_model:
        analysis, t, tok = step_analyze(
            _inst_for(cfg.analyze_model, inst_client), cfg.analyze_model, md_table, query)
        timings["analyze"] = t
        _merge_tokens(tok_all, tok)
    else:
        analysis = query

    # ── Extract + Justify ─────────────────────────────────────────────────────
    if cfg.justify_model and use_parallel:
        t0 = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f_ext = ex.submit(step_extract,
                                  _inst_for(cfg.extract_model, inst_client),
                                  cfg.extract_model, md_table, analysis)
                f_jus = ex.submit(step_justify,
                                  _inst_for(cfg.justify_model, inst_client),
                                  cfg.justify_model, md_table, analysis)
                result_cfg, _, tok_e = f_ext.result()
                charttype,  _, tok_j = f_jus.result()
            result_cfg["charttype"] = charttype
            timings["extract∥justify"] = time.perf_counter() - t0
            _merge_tokens(tok_all, tok_e)
            _merge_tokens(tok_all, tok_j)
        except Exception as _par_err:
            errors.append(f"Parallel failed ({_par_err}) — fell back to sequential")
            result_cfg, t, tok = step_extract(
                _inst_for(cfg.extract_model, inst_client), cfg.extract_model, md_table, analysis)
            timings["extract"] = t
            _merge_tokens(tok_all, tok)
            charttype, t, tok = step_justify(
                _inst_for(cfg.justify_model, inst_client), cfg.justify_model, md_table, analysis)
            result_cfg["charttype"] = charttype
            timings["justify"] = t
            _merge_tokens(tok_all, tok)

    elif cfg.justify_model:
        result_cfg, t, tok = step_extract(
            _inst_for(cfg.extract_model, inst_client), cfg.extract_model, md_table, analysis)
        timings["extract"] = t
        _merge_tokens(tok_all, tok)
        charttype,  t, tok = step_justify(
            _inst_for(cfg.justify_model, inst_client), cfg.justify_model, md_table, analysis)
        result_cfg["charttype"] = charttype
        timings["justify"] = t
        _merge_tokens(tok_all, tok)

    else:
        result_cfg, t, tok = step_extract(
            _inst_for(cfg.extract_model, inst_client), cfg.extract_model, md_table, analysis)
        timings["extract"] = t
        _merge_tokens(tok_all, tok)

    forced_type = detect_requested_chart_type(query)
    if forced_type and result_cfg:
        result_cfg["charttype"] = forced_type

    return result_cfg, errors, timings, tok_all


# ═════════════════════════════════════════════════════════════════════════════
# RENDERER (VisualizationConfig embedded-data format)
# ═════════════════════════════════════════════════════════════════════════════

def _fig_to_bytes(fig):
    """Return PNG bytes for a matplotlib figure — avoids st.image(path) issues with spaces."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def _style_axes(fig, ax):
    fig.patch.set_facecolor("#1a1a24")
    ax.set_facecolor("#13131d")
    ax.tick_params(colors="#aaa", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a3a")

def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

def _align_group(grp):
    x = list(grp.get("x_data") or [])
    y = [_safe_float(v) for v in (grp.get("y_data") or [])]
    n = min(len(x), len(y))
    return {**grp, "x_data": x[:n], "y_data": y[:n]}


def _resolve_chart_position(ann, chart_type, chart_items, data_groups):
    """Resolve annotation (x, y) from data_id/data_value indices or x_value/y_value."""
    data_id    = ann.get("data_id")
    data_value = ann.get("data_value")
    x_value    = ann.get("x_value")
    y_value    = ann.get("y_value")

    if data_id is not None and data_value is not None:
        try:
            gi = int(round(float(data_id)))
            di = int(round(float(data_value)))
        except (ValueError, TypeError):
            gi = di = -1

        if gi >= 0 and di >= 0:
            if chart_type in ("bar", "column") and gi < len(chart_items):
                bars = list(chart_items[gi])
                if di < len(bars):
                    b = bars[di]
                    return (b.get_x() + b.get_width() / 2, b.get_height())

            elif chart_type == "stackedbar" and gi < len(chart_items):
                bars = list(chart_items[gi])
                if di < len(bars):
                    b = bars[di]
                    return (b.get_x() + b.get_width() / 2, b.get_y() + b.get_height())

            elif chart_type == "line" and gi < len(data_groups):
                grp = data_groups[gi]
                if di < len(grp["y_data"]):
                    return (di, grp["y_data"][di])

            elif chart_type == "scatter" and gi < len(chart_items):
                offs = chart_items[gi].get_offsets()
                if di < len(offs):
                    return (float(offs[di][0]), float(offs[di][1]))

            elif chart_type == "pie" and chart_items:
                patches = chart_items[0][0]
                if gi < len(patches):
                    w = patches[gi]
                    theta = (w.theta1 + w.theta2) / 2 * np.pi / 180
                    r = w.r * 0.55
                    return (r * np.cos(theta), r * np.sin(theta))

    if x_value is not None and y_value is not None:
        try:
            return (float(x_value), float(y_value))
        except (ValueError, TypeError):
            pass
    return None


def render_chart(cfg):
    """Render from VisualizationConfig dict (all 18-scenario output format)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _style_axes(fig, ax)

    chart_type  = cfg.get("charttype", "bar")
    title       = cfg.get("titlename", "Chart")
    xlabel      = cfg.get("xlabel", "")
    ylabel      = cfg.get("ylabel", "")
    data_groups = [_align_group(g) for g in (cfg.get("data") or [])]
    annotations = cfg.get("annotations") or []

    if not data_groups:
        ax.text(0.5, 0.5, "No data returned by model", ha="center", va="center",
                color="#ef5350", fontsize=11, transform=ax.transAxes)
        ax.set_title(title, color="#e8e8f0", fontsize=13, fontweight="bold", pad=16)
        plt.tight_layout()
        return fig

    chart_items = []  # track rendered items for annotation resolution

    if chart_type in ("bar", "column"):
        if len(data_groups) == 1:
            grp    = data_groups[0]
            x_vals = [str(x) for x in grp["x_data"]]
            y_vals = list(grp["y_data"])
            bars   = ax.bar(x_vals, y_vals, color=PALETTE[0], alpha=0.85, width=0.6, zorder=3)
            ax.bar_label(bars, labels=[f"{v:,.0f}" for v in y_vals], padding=4,
                         color="#ccc", fontsize=8, fontfamily="monospace")
            ax.grid(axis="y", color="#2a2a3a", linewidth=0.8, zorder=0)
            chart_items = [bars]
        else:
            n     = len(data_groups)
            lbls  = [str(x) for x in data_groups[0]["x_data"]]
            x_pos = np.arange(len(lbls))
            bw    = 0.8 / n
            for i, grp in enumerate(data_groups):
                bars = ax.bar(x_pos + i*bw - (n-1)*bw/2, grp["y_data"], width=bw,
                              label=grp["label"], color=PALETTE[i % len(PALETTE)],
                              alpha=0.85, zorder=3)
                chart_items.append(bars)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(lbls)
            ax.legend(facecolor="#1a1a24", edgecolor="#2a2a3a", labelcolor="#ccc", fontsize=8)
            ax.grid(axis="y", color="#2a2a3a", linewidth=0.8, zorder=0)

    elif chart_type == "stackedbar":
        lbls    = [str(x) for x in data_groups[0]["x_data"]]
        bottoms = [0.0] * len(lbls)
        for i, grp in enumerate(data_groups):
            bars = ax.bar(lbls, grp["y_data"], bottom=bottoms, label=grp["label"],
                          color=PALETTE[i % len(PALETTE)], alpha=0.85, zorder=3)
            chart_items.append(bars)
            bottoms = [b + v for b, v in zip(bottoms, grp["y_data"])]
        ax.legend(facecolor="#1a1a24", edgecolor="#2a2a3a", labelcolor="#ccc", fontsize=8)
        ax.grid(axis="y", color="#2a2a3a", linewidth=0.8, zorder=0)

    elif chart_type == "line":
        x_vals = []
        for i, grp in enumerate(data_groups):
            x_vals = [str(x) for x in grp["x_data"]]
            lines  = ax.plot(x_vals, grp["y_data"], color=PALETTE[i % len(PALETTE)],
                             linewidth=2.5, marker="o", markersize=6,
                             label=grp["label"], zorder=3)
            ax.fill_between(range(len(x_vals)), grp["y_data"],
                            alpha=0.08, color=PALETTE[i % len(PALETTE)])
            chart_items.append(lines)
        if x_vals:
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals)
        ax.grid(color="#2a2a3a", linewidth=0.8, zorder=0)
        if len(data_groups) > 1:
            ax.legend(facecolor="#1a1a24", edgecolor="#2a2a3a", labelcolor="#ccc", fontsize=8)

    elif chart_type == "pie":
        # Multiple groups → one wedge per group (label from group.label)
        # Single group   → one wedge per y_data value (label from x_data)
        if len(data_groups) > 1:
            pie_vals   = [sum(float(v) for v in (g.get("y_data") or []) if v is not None)
                          for g in data_groups]
            pie_labels = [g.get("label") or f"Slice {i+1}" for i, g in enumerate(data_groups)]
        else:
            grp      = data_groups[0]
            x_items  = grp.get("x_data") or []
            y_items  = [float(v) for v in (grp.get("y_data") or []) if v is not None]
            if x_items and len(x_items) == len(y_items):
                pie_vals   = y_items
                pie_labels = [str(x) for x in x_items]
            else:
                pie_vals   = y_items
                pie_labels = ([str(x) for x in x_items[:len(y_items)]] if x_items
                              else [grp.get("label") or f"Slice {i+1}"
                                    for i in range(len(y_items))])
        pie_colors = PALETTE[:len(pie_vals)]
        if any(v > 0 for v in pie_vals):
            result = ax.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%",
                            colors=pie_colors,
                            textprops={"color": "#ccc", "fontsize": 9})
            chart_items = [result[:3] if len(result) >= 3 else result]

    elif chart_type == "scatter":
        for i, grp in enumerate(data_groups):
            sc = ax.scatter(grp["x_data"], grp["y_data"],
                            color=PALETTE[i % len(PALETTE)],
                            s=80, alpha=0.8, label=grp["label"], zorder=3)
            chart_items.append(sc)
        ax.grid(color="#2a2a3a", linewidth=0.8, zorder=0)
        if len(data_groups) > 1:
            ax.legend(facecolor="#1a1a24", edgecolor="#2a2a3a", labelcolor="#ccc", fontsize=8)

    # ── Apply y-axis config from VisualizationConfig ──────────────────────────
    if chart_type != "pie":
        y_lim   = cfg.get("y_lim")
        y_ticks = cfg.get("y_ticks")
        y_lbls  = cfg.get("y_tick_label")
        if y_lim and len(y_lim) == 2 and all(v is not None for v in y_lim):
            try:
                ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
            except (ValueError, TypeError):
                pass
        if y_ticks and len(y_ticks) > 0:
            try:
                ticks = [float(t) for t in y_ticks]
                ax.set_yticks(ticks)
                if y_lbls and len(y_lbls) == len(ticks):
                    ax.set_yticklabels([str(l) for l in y_lbls], color="#aaa", fontsize=9)
            except (ValueError, TypeError):
                pass

    # ── Annotations: numbered circles on chart + text legend below ────────────
    anno_log = []
    for ann in (annotations or []):
        txt = (ann.get("text") or "").strip()
        if not txt:
            continue
        pos = _resolve_chart_position(ann, chart_type, chart_items, data_groups)
        if pos is None:
            continue
        n = len(anno_log) + 1
        try:
            ax.annotate(
                f"({n})",
                xy=pos,
                xytext=(0, 18),
                textcoords="offset points",
                ha="center", va="bottom",
                color="#f0e080", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="#1a1a24",
                          edgecolor="#f0e080", alpha=0.85, linewidth=1.2),
                arrowprops=dict(arrowstyle="->", color="#f0e080", lw=1.0),
                zorder=11,
            )
        except Exception:
            pass
        anno_log.append((n, txt))

    # ── Formatters / labels ───────────────────────────────────────────────────
    if chart_type != "pie":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if abs(v) >= 1e6
                         else f"{v/1e3:.0f}K" if abs(v) >= 1e3 else f"{v:.0f}"))

    ax.set_xlabel(xlabel, color="#888", fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color="#888", fontsize=10, labelpad=8)
    ax.set_title(title, color="#e8e8f0", fontsize=13, fontweight="bold", pad=16)
    n_x = len(data_groups[0]["x_data"]) if data_groups and chart_type != "pie" else 0
    plt.xticks(rotation=30 if n_x > 6 else 0, ha="right" if n_x > 6 else "center")
    plt.tight_layout(pad=1.5)

    # ── Annotation legend below chart ─────────────────────────────────────────
    if anno_log:
        legend_lines = []
        for num, txt in anno_log:
            words, cur, char_count = txt.split(), [], 0
            parts = []
            for w in words:
                if cur and char_count + len(w) + 1 > 65:
                    parts.append(" ".join(cur))
                    cur, char_count = [w], len(w)
                else:
                    cur.append(w); char_count += len(w) + 1
            if cur:
                parts.append(" ".join(cur))
            legend_lines.append(f"({num}) " + "\n     ".join(parts))
        legend_str = "\n".join(legend_lines)
        n_lines    = legend_str.count("\n") + 1
        bot        = min(0.28, 0.06 + 0.032 * n_lines)
        plt.subplots_adjust(bottom=bot)
        fig.text(0.04, bot * 0.35, legend_str,
                 ha="left", va="bottom",
                 color="#aaa", fontsize=7.5, fontfamily="monospace")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# QWEN JUDGE
# ═════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = """You are an independent expert evaluator for LLM-generated chart configurations.

You will receive: (1) the original user question, (2) the source data table, (3) a generated JSON chart configuration.

Step through these checks before writing your verdict:
1. Chart type suitability (trend→line, comparison→bar, proportion→pie)
2. Correct metric / column being plotted
3. Time range correctness
4. Axis labels and title meaningfulness
5. Data value accuracy vs. source table
6. Business usefulness for the user's question

Return VALID JSON ONLY — no markdown, no preamble:
{
  "verdict": "PASS" or "FAIL",
  "score": 0.0-1.0,
  "reasoning_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "evaluation_statement": "one concise summary sentence",
  "main_strengths": ["strength 1"],
  "main_weaknesses": ["weakness 1"],
  "suggested_improvement": "one concrete fix, or null if PASS"
}"""

def _fix_invalid_escapes(s: str) -> str:
    """Replace backslashes not part of a valid JSON escape with a double backslash."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

def _repair_truncated_json(raw: str) -> str:
    """Close any open arrays/objects left by a truncated response."""
    depth_brace   = 0
    depth_bracket = 0
    in_string     = False
    escape_next   = False
    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":  depth_brace   += 1
        elif ch == "}": depth_brace   -= 1
        elif ch == "[": depth_bracket += 1
        elif ch == "]": depth_bracket -= 1
    # close any open string, then close brackets/braces
    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "]" * max(depth_bracket, 0)
    suffix += "}" * max(depth_brace, 0)
    return raw + suffix

def run_qwen_judge(question, md_table, cfg):
    try:
        model   = get_qwen_model_name()
        client  = get_qwen_client()
        cfg_str = json.dumps(cfg, indent=2, default=str)[:3000]
        resp    = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content":
                    f"QUESTION:\n{question}\n\nSOURCE DATA:\n{md_table}\n\nCONFIG:\n{cfg_str}"},
            ],
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        if "{" in raw: raw = raw[raw.index("{"):]
        if "}" in raw:
            raw = raw[:raw.rindex("}")+1]
        else:
            raw = _repair_truncated_json(raw)
        raw = _fix_invalid_escapes(raw)
        return json.loads(raw)
    except Exception as e:
        return {
            "verdict": "ERROR", "score": 0.0,
            "evaluation_statement": str(e),
            "reasoning_steps": [], "main_strengths": [],
            "main_weaknesses": [], "suggested_improvement": "",
        }


# ═════════════════════════════════════════════════════════════════════════════
# CORE EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def execute_scenario(query, md_table, sid, use_cache, threshold,
                     use_parallel, encoder, enable_judge, api_key):
    result = {
        "sid":           sid,
        "label":         scenario_label(sid),
        "cache_hit":     False,
        "cache_sim":     0.0,
        "timings":       {},
        "cfg":           {},
        "fig":           None,
        "chart_bytes":   None,
        "chart_path":    None,
        "errors":        [],
        "verdict":          "—",
        "score":            "—",
        "reasoning":        "",
        "reasoning_steps":  [],
        "strengths":        [],
        "weaknesses":       [],
        "improvement":      "",
        "total_latency":    0.0,
        "token_counts":     {},
        "total_tokens":     0,
    }
    t_global = time.perf_counter()

    # ── Cache lookup ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cached, sim = (semantic_cache_lookup(query, encoder, threshold, scenario_id=sid)
                   if use_cache else (None, 0.0))
    result["timings"]["cache_lookup"] = time.perf_counter() - t0
    result["cache_sim"] = sim

    if cached:
        result["cache_hit"]        = True
        result["verdict"]          = cached.get("verdict", "—")
        result["score"]            = str(cached.get("score", "—"))
        result["reasoning"]        = cached.get("reasoning", "")
        result["reasoning_steps"]  = cached.get("reasoning_steps", [])
        result["strengths"]        = cached.get("strengths", [])
        result["weaknesses"]       = cached.get("weaknesses", [])
        result["improvement"]      = cached.get("improvement", "")
        cfg_path = Path(cached["config_path"])
        if cfg_path.exists():
            result["cfg"] = json.loads(cfg_path.read_text(encoding="utf-8"))
        chart_path = Path(cached["chart_path"])
        if chart_path.exists():
            result["chart_path"]  = chart_path
            result["chart_bytes"] = chart_path.read_bytes()
        elif result["cfg"]:
            t0 = time.perf_counter()
            fig = render_chart(result["cfg"])
            result["timings"]["render"] = time.perf_counter() - t0
            result["chart_bytes"] = _fig_to_bytes(fig)
            fig.savefig(chart_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            result["chart_path"] = chart_path
            plt.close(fig)
        result["total_latency"] = time.perf_counter() - t_global
        return result

    # ── Run pipeline ──────────────────────────────────────────────────────────
    inst_client, raw_client = get_clients(api_key)

    if sid.startswith("CB-"):
        # Custom baseline: full pipeline (analyze → extract → justify) on a single model
        model_id = sid[3:]
        try:
            _inst  = inst_client
            _model = model_id

            token_counts = {}
            timings      = {}

            analysis, t_a, tok_a = step_analyze(_inst, _model, md_table, query)
            timings["analyze"] = t_a
            _merge_tokens(token_counts, tok_a)

            if use_parallel:
                t0 = time.perf_counter()
                _par_errors = []
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        f_ext = ex.submit(step_extract, _inst, _model, md_table, analysis)
                        f_jus = ex.submit(step_justify, _inst, _model, md_table, analysis)
                        cfg, _, tok_e = f_ext.result()
                        charttype, _, tok_j = f_jus.result()
                    cfg["charttype"] = charttype
                    timings["extract∥justify"] = time.perf_counter() - t0
                    _merge_tokens(token_counts, tok_e)
                    _merge_tokens(token_counts, tok_j)
                except Exception as _par_err:
                    _par_errors.append(f"Parallel failed ({_par_err}) — fell back to sequential")
                    cfg, t_e, tok_e = step_extract(_inst, _model, md_table, analysis)
                    timings["extract"] = t_e
                    _merge_tokens(token_counts, tok_e)
                    charttype, t_j, tok_j = step_justify(_inst, _model, md_table, analysis)
                    cfg["charttype"] = charttype
                    timings["justify"] = t_j
                    _merge_tokens(token_counts, tok_j)
                errors = _par_errors
            else:
                cfg, t_e, tok_e = step_extract(_inst, _model, md_table, analysis)
                timings["extract"] = t_e
                _merge_tokens(token_counts, tok_e)

                charttype, t_j, tok_j = step_justify(_inst, _model, md_table, analysis)
                cfg["charttype"] = charttype
                timings["justify"] = t_j
                _merge_tokens(token_counts, tok_j)

            forced_type = detect_requested_chart_type(query)
            if forced_type:
                cfg["charttype"] = forced_type
            errors = []
        except Exception as e:
            result["errors"] = [str(e)]
            result["total_latency"] = time.perf_counter() - t_global
            return result
    else:
        try:
            cfg, errors, timings, token_counts = run_pipeline(
                sid, query, md_table, use_parallel, inst_client, raw_client)
        except Exception as e:
            result["errors"] = [str(e)]
            result["total_latency"] = time.perf_counter() - t_global
            return result

    result["cfg"]          = cfg or {}
    result["errors"]       = errors
    result["timings"].update(timings)
    result["token_counts"] = token_counts
    result["total_tokens"] = token_counts.get("total_tokens", 0)

    # ── Qwen judge ────────────────────────────────────────────────────────────
    jr = None
    if enable_judge and cfg and not errors:
        t0 = time.perf_counter()
        jr = run_qwen_judge(query, md_table, cfg)
        result["timings"]["judge"]  = time.perf_counter() - t0
        result["verdict"]           = jr.get("verdict", "—")
        result["score"]             = str(jr.get("score", "—"))
        result["reasoning"]         = jr.get("evaluation_statement", "")
        result["reasoning_steps"]   = jr.get("reasoning_steps", [])
        result["strengths"]         = jr.get("main_strengths", [])
        result["weaknesses"]        = jr.get("main_weaknesses", [])
        result["improvement"]       = jr.get("suggested_improvement", "")

    # ── Render ────────────────────────────────────────────────────────────────
    if cfg:
        try:
            t0 = time.perf_counter()
            fig = render_chart(cfg)
            result["timings"]["render"] = time.perf_counter() - t0

            result["chart_bytes"] = _fig_to_bytes(fig)

            chart_id   = hashlib.md5(f"{query}::{sid}".encode()).hexdigest()[:10]
            chart_path = CHARTS_DIR / f"{chart_id}.png"
            chart_path.write_bytes(result["chart_bytes"])
            result["chart_path"] = chart_path
            plt.close(fig)

            if use_cache and not errors:
                save_to_cache(query, cfg, chart_path,
                              result["verdict"], result["score"],
                              time.perf_counter() - t_global, sid, encoder,
                              judge_result=jr if enable_judge and cfg and not errors else None)
        except Exception as _render_err:
            result["errors"].append(f"Render error: {_render_err}")

    result["total_latency"] = time.perf_counter() - t_global
    return result


def run_combined_methods(query, md_table, sid, encoder, api_key,
                         use_parallel=False, progress_cb=None):
    """
    Runs one scenario in 4 optimization combinations.
    Purges the existing cache entry first so miss/hit are always genuine.
    Calls progress_cb(i, label) before each step if provided.
    Returns list of 4 result dicts, each with a 'combo_label' key.
    """
    _purge_cache_for_combo(query, sid)

    combos = [
        ("Baseline",       False, False),
        ("+Parallel",      False, use_parallel),
        ("+Cache (miss)",  True,  False),
        ("+Cache (hit)",   True,  False),   # guaranteed hit from step above
    ]
    results = []
    for i, (label, use_cache, par) in enumerate(combos):
        if progress_cb:
            progress_cb(i, label)
        r = execute_scenario(
            query, md_table, sid,
            use_cache, SEM_THRESHOLD, par,
            encoder, False, api_key,
        )
        r["combo_label"] = label
        results.append(r)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def save_run_results(query, selected_scenarios, all_results, use_cache, use_parallel, label="run"):
    """Save every artifact from a run to results/{timestamp}/ and return the Path."""
    import zipfile as _zf
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{ts}_{label}"
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)

    # ── run_info.json ──────────────────────────────────────────────────────────
    (run_dir / "run_info.json").write_text(json.dumps({
        "timestamp":  ts,
        "query":      query,
        "scenarios":  selected_scenarios,
        "settings":   {"cache_enabled": use_cache, "parallel": use_parallel},
        "n_scenarios": len(all_results),
    }, indent=2), encoding="utf-8")

    # ── summary.csv ───────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        llm_t = sum(v for k, v in r["timings"].items()
                    if k not in ("cache_lookup", "render", "judge"))
        row = {
            "scenario_id":       r["sid"],
            "label":             r.get("label", r["sid"]),
            "total_latency_s":   round(r["total_latency"], 3),
            "llm_latency_s":     round(llm_t, 3) if llm_t > 0.001 else 0,
            "cache_hit":         r["cache_hit"],
            "cache_similarity":  round(r["cache_sim"], 3),
            "verdict":           r["verdict"],
            "quality_score":     r["score"],
            "prompt_tokens":     r.get("token_counts", {}).get("prompt_tokens", 0),
            "completion_tokens": r.get("token_counts", {}).get("completion_tokens", 0),
            "total_tokens":      r.get("total_tokens", 0),
            "errors":            "; ".join(r["errors"]) if r["errors"] else "",
        }
        if r.get("agent_iterations") is not None:
            row["agent_iterations"]  = r["agent_iterations"]
            row["agent_eval_scores"] = str(r.get("agent_eval_scores", []))
        if r.get("agent_tool_calls") is not None:
            row["agent_tool_calls"] = " → ".join(r["agent_tool_calls"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8")

    # ── latency_breakdown.csv ─────────────────────────────────────────────────
    lat_rows = []
    for r in all_results:
        for stage, t in r["timings"].items():
            lat_rows.append({"scenario_id": r["sid"], "stage": stage,
                             "latency_s": round(t, 4)})
    pd.DataFrame(lat_rows).to_csv(run_dir / "latency_breakdown.csv",
                                  index=False, encoding="utf-8")

    # ── chart images ──────────────────────────────────────────────────────────
    for r in all_results:
        img = r.get("chart_bytes")
        if not img and r.get("chart_path") and Path(r["chart_path"]).exists():
            img = Path(r["chart_path"]).read_bytes()
        if img:
            safe = r["sid"].replace("/", "_")
            (run_dir / "charts" / f"{safe}.png").write_bytes(img)

    # ── latency comparison chart ───────────────────────────────────────────────
    try:
        _lf = make_latency_chart(all_results)
        _lf.savefig(run_dir / "latency_comparison.png", dpi=120,
                    bbox_inches="tight", facecolor=_lf.get_facecolor())
        plt.close(_lf)
    except Exception:
        pass

    # ── per-scenario stage latency charts ─────────────────────────────────────
    for r in all_results:
        try:
            _sf = make_stage_latency_chart(r)
            safe = r["sid"].replace("/", "_")
            _sf.savefig(run_dir / "charts" / f"{safe}_stage_latency.png", dpi=120,
                        bbox_inches="tight", facecolor=_sf.get_facecolor())
            plt.close(_sf)
        except Exception:
            pass

    # ── LLM output configs ────────────────────────────────────────────────────
    for r in all_results:
        if r.get("cfg"):
            safe = r["sid"].replace("/", "_")
            (run_dir / "configs" / f"{safe}.json").write_text(
                json.dumps(r["cfg"], indent=2, default=str), encoding="utf-8")

    return run_dir


def _zip_run(run_dir: Path) -> bytes:
    """Return in-memory ZIP bytes of a results run folder."""
    import zipfile as _zf
    buf = io.BytesIO()
    with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(run_dir.parent))
    buf.seek(0)
    return buf.getvalue()


def make_stage_latency_chart(r):
    """Compact horizontal bar chart of stage latencies for a single result."""
    stages = [k for k in r["timings"] if r["timings"][k] > 0]
    vals   = [r["timings"][k] for k in stages]
    colors = [STAGE_COLORS.get(s, "#667788") for s in stages]

    fig, ax = plt.subplots(figsize=(5, max(1.8, len(stages) * 0.6 + 0.5)))
    _style_axes(fig, ax)

    bars = ax.barh(stages, vals, color=colors, alpha=0.88, height=0.52)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", color="#ccc",
                fontsize=8, fontfamily="monospace")

    ax.set_xlabel("Time (s)", color="#888", fontsize=8, labelpad=4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}s"))
    ax.set_xlim(right=max(vals) * 1.28)
    ax.grid(axis="x", color="#2a2a3a", linewidth=0.6, zorder=0)
    ax.set_title(f"Stage Latency  ·  {r['total_latency']:.2f}s total",
                 color="#e8e8f0", fontsize=9, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


def make_latency_chart(results):
    all_stages = list(dict.fromkeys(s for r in results for s in r["timings"]))
    labels     = [r["sid"] for r in results]
    n          = len(results)

    fig, ax = plt.subplots(figsize=(11, max(2.5, n * 1.3 + 1.2)))
    _style_axes(fig, ax)

    lefts = [0.0] * n
    for stage in all_stages:
        vals  = [r["timings"].get(stage, 0.0) for r in results]
        color = STAGE_COLORS.get(stage, "#667788")
        ax.barh(labels, vals, left=lefts, color=color, alpha=0.85,
                label=stage, height=0.55)
        lefts = [l + v for l, v in zip(lefts, vals)]

    for i, r in enumerate(results):
        ax.text(lefts[i] + 0.02, i, f"  {r['total_latency']:.2f}s",
                va="center", color="#ccc", fontsize=8.5, fontfamily="monospace")

    ax.legend(facecolor="#1a1a24", edgecolor="#2a2a3a", labelcolor="#ccc",
              fontsize=8, loc="lower right")
    ax.set_xlabel("Time (s)", color="#888", fontsize=9, labelpad=6)
    ax.set_title("Latency Breakdown by Stage",
                 color="#e8e8f0", fontsize=11, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}s"))
    ax.grid(axis="x", color="#2a2a3a", linewidth=0.6, zorder=0)
    plt.tight_layout()
    return fig

def _card(val, label):
    return (f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>")

def display_result(r):
    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(_card(f"{r['total_latency']:.2f}s", "Total Latency"),
                    unsafe_allow_html=True)
    with m2:
        b = ("<span class='badge-hit'>CACHE HIT</span>" if r["cache_hit"]
             else "<span class='badge-miss'>CACHE MISS</span>")
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:0.8rem;padding-top:12px'>{b}</div><div class='metric-label'>Cache</div></div>",
                    unsafe_allow_html=True)
    with m3:
        st.markdown(_card(f"{r['cache_sim']:.3f}", "Similarity"), unsafe_allow_html=True)
    with m4:
        v = r["verdict"]
        vb = (f"<span class='badge-pass'>{v}</span>" if v == "PASS"
              else f"<span class='badge-fail'>{v}</span>" if v == "FAIL" else v)
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:0.8rem;padding-top:12px'>{vb}</div><div class='metric-label'>Verdict</div></div>",
                    unsafe_allow_html=True)
    with m5:
        st.markdown(_card(r["score"], "Quality Score"), unsafe_allow_html=True)
    with m6:
        tok = r.get("total_tokens", 0)
        tok_str = f"{tok:,}" if tok else "—"
        st.markdown(_card(tok_str, "Tokens Used"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        st.markdown("<div class='section-title'>Visualization</div>", unsafe_allow_html=True)
        if r.get("chart_bytes"):
            st.image(r["chart_bytes"], use_container_width=True)
        elif r.get("chart_path") and Path(r["chart_path"]).exists():
            st.image(Path(r["chart_path"]).read_bytes(), use_container_width=True)
        elif r.get("cfg"):
            try:
                _fig = render_chart(r["cfg"])
                st.image(_fig_to_bytes(_fig), use_container_width=True)
                plt.close(_fig)
            except Exception:
                st.warning("No chart rendered.")
        else:
            st.warning("No chart rendered.")

    if r["errors"]:
        st.error("⚠️ " + " | ".join(r["errors"]))

    with col_info:
        st.markdown("<div class='section-title'>Stage Latency</div>", unsafe_allow_html=True)
        if r["timings"]:
            lat_fig = make_stage_latency_chart(r)
            st.pyplot(lat_fig, use_container_width=True)
            plt.close(lat_fig)

            # Per-step numbers as compact metric cards
            llm_stages = {k: v for k, v in r["timings"].items()
                          if k not in ("cache_lookup", "render", "judge")}
            if llm_stages:
                llm_total = sum(llm_stages.values())
                st.markdown(
                    f"<div class='metric-card' style='margin-top:6px'>"
                    f"<div class='metric-value' style='font-size:1rem'>{llm_total:.2f}s</div>"
                    f"<div class='metric-label'>LLM Time (excl. cache/render)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Token breakdown
        tok = r.get("token_counts", {})
        if tok:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Token Usage</div>", unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            tc1.markdown(_card(f"{tok.get('prompt_tokens', 0):,}", "Prompt"),
                         unsafe_allow_html=True)
            tc2.markdown(_card(f"{tok.get('completion_tokens', 0):,}", "Completion"),
                         unsafe_allow_html=True)
            tc3.markdown(_card(f"{tok.get('total_tokens', 0):,}", "Total"),
                         unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>JSON Config</div>", unsafe_allow_html=True)
        if r["cfg"]:
            with st.expander("View config"):
                st.json(r["cfg"])

        if r["reasoning"]:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Qwen Judge</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='reasoning-box'>{r['reasoning']}</div>",
                        unsafe_allow_html=True)
            if r.get("reasoning_steps"):
                with st.expander("Reasoning steps"):
                    for step in r["reasoning_steps"]:
                        st.markdown(f"- {step}")
            sc1, sc2 = st.columns(2)
            with sc1:
                if r.get("strengths"):
                    st.markdown("**✓ Strengths**")
                    for s in r["strengths"]:
                        st.markdown(f"<span style='color:#4caf50;font-size:0.82rem'>+ {s}</span>",
                                    unsafe_allow_html=True)
            with sc2:
                if r.get("weaknesses"):
                    st.markdown("**✗ Weaknesses**")
                    for w in r["weaknesses"]:
                        st.markdown(f"<span style='color:#ef5350;font-size:0.82rem'>− {w}</span>",
                                    unsafe_allow_html=True)
            imp = r.get("improvement", "")
            if imp and imp not in ("null", "None", None, ""):
                st.markdown(f"<span style='color:#f0a855;font-size:0.82rem'>💡 {imp}</span>",
                            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("---")
    st.markdown("### 🔑 API Key")
    api_key_in = st.text_input("OpenAI API Key", type="password",
                                value=os.getenv("OPENAI_API_KEY", ""))
    if api_key_in:
        os.environ["OPENAI_API_KEY"] = api_key_in

    st.markdown("---")
    st.markdown("### 🏗 Scenarios (18 total)")
    st.caption("Select one or more to run and compare")

    # Group them for display
    groups = {}
    for sid, (_, cfg) in SCENARIOS.items():
        groups.setdefault(cfg.group, []).append(sid)

    selected_scenarios = []
    for group_name, sids in groups.items():
        st.markdown(f"**{group_name}**")
        for sid in sids:
            _, cfg = SCENARIOS[sid]
            steps = []
            if cfg.router_model: steps.append(f"Router[{cfg.router_model}]")
            if cfg.analyze_model: steps.append(f"A[{cfg.analyze_model}]")
            steps.append(f"E[{cfg.extract_model}]")
            if cfg.justify_model: steps.append(f"J[{cfg.justify_model}]")
            label = f"{sid}: " + "→".join(steps)
            if st.checkbox(label, key=f"chk_{sid}"):
                selected_scenarios.append(sid)

    st.markdown("---")
    st.markdown("### 🔬 Custom Baselines")
    st.caption("Full pipeline (Analyze → Extract → Justify) on a single model — pick any model to benchmark")
    with st.expander("Select models"):
        for model_id, display_name in BASELINE_MODELS:
            lbl = f"CB-{model_id}: Analyze→Extract→Justify[{display_name}]"
            if st.checkbox(lbl, key=f"cb_{model_id}"):
                selected_scenarios.append(f"CB-{model_id}")

    st.markdown("---")
    st.markdown("### ⚡ Execution")
    use_parallel = st.toggle("Parallel Extract∥Justify", value=False,
                             help="Runs Extract and Justify concurrently for scenarios that have both steps")

    st.markdown("---")
    st.markdown("### 💾 Semantic Cache")
    use_cache = st.toggle("Enable Cache", value=False)
    threshold = st.slider("Similarity Threshold", 0.5, 1.0, SEM_THRESHOLD, 0.05,
                          disabled=not use_cache)
    cache_idx = load_cache_index()
    c1, c2 = st.columns(2)
    with c1: st.metric("Cached", len(cache_idx))
    with c2:
        if st.button("🗑 Clear"):
            save_cache_index([])
            st.success("Cleared.")

    st.markdown("---")
    st.markdown("### 🔍 Qwen Judge")
    enable_judge = st.toggle("Enable Qwen Judge", value=False)
    st.caption("hal9000.skim.th-owl.de\n(model auto-discovered on first use)")

    st.markdown("---")
    st.markdown("### 🔭 Phoenix Tracing")
    if _PHOENIX_AVAILABLE:
        enable_phoenix = st.toggle("Enable Phoenix", value=False, key="phoenix_toggle")
        if enable_phoenix:
            _ph_url, _ph_err = start_phoenix()
            if _ph_url:
                st.success("Connected to Phoenix")
                st.markdown(f"[Open Phoenix UI ↗]({_ph_url})")
                st.caption("All OpenAI calls are now traced.")
            else:
                st.warning(_ph_err or "Could not connect.")
                st.caption("Start Phoenix first:\n```\npython -m phoenix.server.main serve\n```")
        else:
            st.caption("Requires Phoenix running separately.\nStart it with:\npython -m phoenix.server.main serve")
    else:
        st.caption("Install to enable:\npip install arize-phoenix\npip install openinference-instrumentation-openai")

    st.markdown("---")
    st.markdown("### 📋 History")
    if LOG_FILE.exists():
        log_data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        if log_data:
            df_log = pd.DataFrame(log_data[-15:])
            df_log["cache_hit"] = df_log["cache_hit"].map({True: "✅", False: "❌"})
            wanted = ["scenario", "cache_hit", "latency_s", "verdict"]
            cols   = [c for c in wanted if c in df_log.columns]
            st.dataframe(df_log[cols].tail(10),
                         use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📁 Saved Runs")
    if RESULTS_DIR.exists():
        _past_runs = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)[:15]
        if _past_runs:
            for _rd in _past_runs:
                _info_f = _rd / "run_info.json"
                if _info_f.exists():
                    try:
                        _info = json.loads(_info_f.read_text(encoding="utf-8"))
                        _qlabel = (_info.get("query","")[:35] + "…") if len(_info.get("query","")) > 35 else _info.get("query","—")
                        _slabel = f"{_rd.name[:15]}  |  {_qlabel}"
                    except Exception:
                        _slabel = _rd.name
                else:
                    _slabel = _rd.name
                st.download_button(
                    label=f"⬇️ {_slabel}",
                    data=_zip_run(_rd),
                    file_name=f"{_rd.name}.zip",
                    mime="application/zip",
                    key=f"dl_past_{_rd.name}",
                    use_container_width=True,
                )
        else:
            st.caption("No saved runs yet. Run a scenario to save results.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:1.2rem 0 0.6rem 0'>
  <span style='font-family:DM Mono,monospace;font-size:0.7rem;color:#555;letter-spacing:0.12em'>
    TH OWL · THESIS
  </span>
  <h1 style='margin:0;font-size:1.85rem;font-weight:600;color:#e8e8f0'>
    LLM Visualization Benchmark
  </h1>
  <p style='color:#666;font-size:0.83rem;margin-top:4px'>
    18 architectural scenarios · Semantic caching · Parallel execution · Qwen quality judge
  </p>
</div>
""", unsafe_allow_html=True)

EXAMPLE_PROMPTS = [
    "Wieviel Umsatz hatte Teckentrup in den Jahren 2021 bis 2024?",
    "Show the monthly revenue trend for 2022 and highlight the peak month.",
    "Compare total annual revenue across all four years with a bar chart.",
    "Which year had the highest total revenue and by how much did it differ from the lowest?",
    "Visualize the revenue distribution across months for 2023.",
    "Show year-over-year revenue growth from 2021 to 2024.",
]

st.markdown("<div class='section-title'>Example questions</div>", unsafe_allow_html=True)
ex_cols = st.columns(3)
for i, ex in enumerate(EXAMPLE_PROMPTS):
    if ex_cols[i % 3].button(ex[:55] + ("…" if len(ex) > 55 else ""),
                              key=f"ex_{i}", use_container_width=True):
        st.session_state["query_box"] = ex
        st.rerun()

query = st.text_area(
    "Business Question",
    placeholder='e.g. "Wieviel Umsatz hatte Teckentrup in den Jahren 2021 bis 2024?"',
    height=85,
    key="query_box",
)

col_run, _ = st.columns([1, 6])
with col_run:
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

if run_btn and query.strip():
    if not selected_scenarios:
        st.warning("Tick at least one scenario in the sidebar.")
        st.stop()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("No OpenAI API key found. Enter it in the sidebar.")
        st.stop()

    encoder  = load_embedding_model(api_key) if use_cache else None
    MD_TABLE = retrieve_data(None, type="test")

    # ── Warm up OpenAI server before timed runs ───────────────────────────────
    with st.spinner("Warming up models on OpenAI server…"):
        warmup_models(selected_scenarios, api_key)

    # ── Run all selected scenarios ────────────────────────────────────────────
    all_results = []
    bar = st.progress(0, text="Starting…")

    for i, sid in enumerate(selected_scenarios):
        bar.progress(i / len(selected_scenarios), text=f"Running {sid}…")
        with st.spinner(f"⏳ {scenario_label(sid)}"):
            r = execute_scenario(
                query, MD_TABLE, sid,
                use_cache, threshold, use_parallel,
                encoder, enable_judge, api_key,
            )
        all_results.append(r)
        log_query(query, sid, r["cache_hit"], r["cache_sim"],
                  r["total_latency"], r["verdict"])

    bar.progress(1.0, text="✓ Done")

    # ── Auto-save + download ──────────────────────────────────────────────────
    _run_dir = save_run_results(query, selected_scenarios, all_results,
                                use_cache, use_parallel, label="scenarios")
    st.session_state["_last_run_dir"] = str(_run_dir)
    _dl_col, _ = st.columns([1, 4])
    with _dl_col:
        st.download_button(
            label="⬇️ Download Results (ZIP)",
            data=_zip_run(_run_dir),
            file_name=f"viz_run_{_run_dir.name}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    multi      = len(all_results) > 1
    tab_labels = (["📊 Comparison"] if multi else []) + [r["sid"] for r in all_results]
    tabs       = st.tabs(tab_labels)

    if multi:
        with tabs[0]:
            st.markdown("### Scenario Comparison")

            # Summary table
            # Gather all stage keys that appear across results
            all_stage_keys = list(dict.fromkeys(
                s for r in all_results for s in r["timings"]
            ))
            rows = []
            for r in all_results:
                llm_t = sum(v for k, v in r["timings"].items()
                            if k not in ("cache_lookup", "render", "judge"))
                row = {
                    "Scenario":   r["sid"],
                    "Total (s)":  f"{r['total_latency']:.3f}",
                    "LLM (s)":    f"{llm_t:.3f}",
                }
                for s in all_stage_keys:
                    v = r["timings"].get(s)
                    row[s] = f"{v:.3f}" if v else "—"
                row["Cache"]      = "HIT ✅" if r["cache_hit"] else "MISS ❌"
                row["Similarity"] = f"{r['cache_sim']:.3f}"
                row["Tokens"]     = f"{r.get('total_tokens', 0):,}" if r.get("total_tokens") else "—"
                row["Verdict"]    = r["verdict"]
                row["Score"]      = r["score"]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            lat_fig = make_latency_chart(all_results)
            st.pyplot(lat_fig, use_container_width=True)
            plt.close(lat_fig)

            # ── Visualizations grid ───────────────────────────────────────────
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Generated Visualizations</div>",
                        unsafe_allow_html=True)
            viz_cols = st.columns(len(all_results))
            for col, r in zip(viz_cols, all_results):
                with col:
                    st.caption(r["sid"])
                    img = r.get("chart_bytes")
                    if not img and r.get("chart_path") and Path(r["chart_path"]).exists():
                        img = Path(r["chart_path"]).read_bytes()
                    if not img and r.get("cfg"):
                        try:
                            _f = render_chart(r["cfg"])
                            img = _fig_to_bytes(_f)
                            plt.close(_f)
                        except Exception:
                            pass
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("No chart")

        for i, r in enumerate(all_results):
            with tabs[i + 1]:
                hit_tag = " 🔵 CACHE HIT" if r["cache_hit"] else ""
                st.markdown(f"<div class='scenario-header'>{r['label']}{hit_tag}</div>",
                            unsafe_allow_html=True)
                display_result(r)
    else:
        r = all_results[0]
        with tabs[0]:
            hit_tag = " 🔵 CACHE HIT" if r["cache_hit"] else ""
            st.markdown(f"<div class='scenario-header'>{r['label']}{hit_tag}</div>",
                        unsafe_allow_html=True)
            display_result(r)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Latency Breakdown</div>",
                        unsafe_allow_html=True)
            lat_fig = make_latency_chart(all_results)
            st.pyplot(lat_fig, use_container_width=True)
            plt.close(lat_fig)

elif run_btn:
    st.warning("Please enter a business question.")

# ── Combined Methods Benchmark ─────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### 🔀 Combined Methods Benchmark")
st.markdown(
    "<p style='color:#666;font-size:0.83rem;margin-top:-8px'>"
    "Runs one scenario in 4 optimization variants — Baseline · +Parallel · "
    "+Cache(miss) · +Cache(hit) — and shows the visualization and latency for each.</p>",
    unsafe_allow_html=True,
)

_cb1, _cb2, _cb3 = st.columns([3, 1, 1])
with _cb1:
    combo_sid = st.selectbox("Scenario", list(SCENARIOS.keys()), key="combo_sid")
with _cb2:
    combo_parallel = st.toggle("Parallel", value=False, key="combo_par",
                               help="Parallel Extract∥Justify for applicable scenarios")
with _cb3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    combo_btn = st.button("▶ Run Combined", key="combo_btn", type="secondary",
                          use_container_width=True)

# Cache key — invalidates stored results when query or scenario changes
_combo_cache_key = f"{query.strip()}::{combo_sid}"

if combo_btn:
    if not query.strip():
        st.warning("Enter a business question in the query box above first.")
    elif not os.getenv("OPENAI_API_KEY", ""):
        st.error("No OpenAI API key found.")
    else:
        _encoder  = load_embedding_model(os.getenv("OPENAI_API_KEY", ""))
        _MD_TABLE = retrieve_data(None, type="test")
        _prog     = st.progress(0, text="Preparing…")

        def _cb_progress(i, label):
            _prog.progress(i / 4, text=f"Step {i+1}/4: {label}…")

        _combo_results = run_combined_methods(
            query, _MD_TABLE, combo_sid, _encoder,
            os.getenv("OPENAI_API_KEY", ""),
            use_parallel=combo_parallel,
            progress_cb=_cb_progress,
        )
        _prog.progress(1.0, text="✓ All 4 combinations done")

        _combo_run_dir = save_run_results(
            query, [combo_sid], _combo_results,
            use_cache=True, use_parallel=combo_parallel, label="combined")
        st.session_state["_combo_run_dir"]  = str(_combo_run_dir)
        st.session_state["_combo_results"]  = _combo_results
        st.session_state["_combo_key"]      = _combo_cache_key

# ── Render combined results (persisted in session state) ─────────────────────
if (st.session_state.get("_combo_key") == _combo_cache_key
        and st.session_state.get("_combo_results")):

    _cr       = st.session_state["_combo_results"]
    _base_t   = _cr[0]["total_latency"] or 1e-9

    # Summary comparison table
    _sum_rows = []
    for _r in _cr:
        _llm = sum(v for k, v in _r["timings"].items()
                   if k not in ("cache_lookup", "render", "judge"))
        _sum_rows.append({
            "Method":              _r["combo_label"],
            "Total (s)":           f"{_r['total_latency']:.3f}",
            "LLM (s)":             f"{_llm:.3f}" if _llm > 0.001 else "—",
            "Cache":               "HIT ✅" if _r["cache_hit"] else "MISS ❌",
            "Speedup vs Baseline": f"{_base_t / max(_r['total_latency'], 0.001):.1f}×",
            "Errors":              "; ".join(_r["errors"]) if _r["errors"] else "—",
        })
    st.dataframe(pd.DataFrame(_sum_rows), hide_index=True, use_container_width=True)

    # Download button for combined results
    if st.session_state.get("_combo_run_dir"):
        _cdl_col, _ = st.columns([1, 4])
        with _cdl_col:
            st.download_button(
                label="⬇️ Download Combined Results (ZIP)",
                data=_zip_run(Path(st.session_state["_combo_run_dir"])),
                file_name=f"viz_combined_{Path(st.session_state['_combo_run_dir']).name}.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_combo",
            )

    # Latency breakdown chart (relabel sid for axis)
    _cr_labeled = [{**_r, "sid": _r["combo_label"]} for _r in _cr]
    _lat_fig    = make_latency_chart(_cr_labeled)
    st.pyplot(_lat_fig, use_container_width=True)
    plt.close(_lat_fig)

    # Per-combination tabs — shows the actual visualization + stage chart
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _ctabs = st.tabs([_r["combo_label"] for _r in _cr])
    for _ctab, _r in zip(_ctabs, _cr):
        with _ctab:
            _hit = " 🔵 CACHE HIT" if _r["cache_hit"] else ""
            st.markdown(
                f"<div class='scenario-header'>{combo_sid} · {_r['combo_label']}{_hit}</div>",
                unsafe_allow_html=True,
            )
            display_result(_r)

# ── Cache Browser ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
with st.expander("📦 Cache Browser", expanded=False):
    idx = load_cache_index()
    if not idx:
        st.info("No cached entries yet.")
    else:
        for entry in reversed(idx[-20:]):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                q = entry["query"]
                st.markdown(f"**{q[:72]}{'…' if len(q) > 72 else ''}**")
                st.caption(f"{entry.get('scenario','—')} · {entry['latency_s']}s · {entry['timestamp'][:16]}")
            with c2:
                v = entry.get("verdict", "—")
                color = "#4caf50" if v == "PASS" else "#ef5350" if v == "FAIL" else "#888"
                st.markdown(f"<span style='color:{color};font-weight:600'>{v}</span>",
                            unsafe_allow_html=True)
            with c3:
                st.caption(f"sim {entry.get('similarity', '—')}")
            st.markdown("---")
