"""
RQ5 — Parallel Execution Benchmark WITH ROUTER
================================================
Mirrors rq5_parallel_benchmark.ipynb exactly, but prepends a real router
LLM call (tool_calling_client) before the three pipeline stages.

This lets you compare:
  - Serial   (router + analysis + config + type, sequential)
  - Parallel (router + analysis + [config ∥ type])
  vs the router-free baselines from the original RQ5 notebook.

Run from the project root:
    python rq5_with_router_benchmark.py

Outputs:
    rq5_router_latency_comparison.png
    rq5_router_gantt.png
    rq5_router_throughput.png
    rq5_router_throughput_results.csv
"""

import sys
import os
sys.path.insert(0, "/Users/srujanakadambari/Desktop/ffresh thesis/data-to-visual/data-to-visual-nicos-branch")

import time
import json
import statistics
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from retrieve_data import retrieve_data
from init_phoenix import init_phoenix
from prompts.default import (
    DATA_ANALYSIS_PROMPT,
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_TYPE_JUSTIFICATION_PROMPT,
    SYSTEM_PROMPT,
)
from response_models.default import (
    DataAnalysis, VisualizationConfig, ChartTypeJustification, ChartType
)
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Init ─────────────────────────────────────────────────────────────────────
client, tool_calling_client, tracer = init_phoenix("rq5-router-benchmark")
MODEL = "o4-mini"

# ── Data & queries (same as original RQ5 notebook) ───────────────────────────
MD_TABLE = retrieve_data(None, type="test")

QUERIES = [
    "Summarize Umsatz for 2021–2024 and generate a grouped bar chart by month and year.",
    "Show the monthly revenue trend as a line chart and highlight the top 3 peaks.",
    "Compare total annual revenue across 2021, 2022, 2023, 2024 with a bar chart.",
    "Identify the months with the lowest revenue and visualize with annotations.",
    "Plot the full 2021–2024 time series and annotate any months with zero or near-zero revenue.",
]

# Tool definition — same as Agent.tools in generate_visualization.py
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Analyze the data and generate a visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "data":       {"type": "string", "description": "The data as a markdown table"},
                    "user_query": {"type": "string", "description": "The users query"},
                },
                "required": ["data", "user_query"],
            },
        },
    }
]

N_RUNS = 3
CONCURRENCY_LEVELS = [1, 2, 4, 8]

# ── Helpers ───────────────────────────────────────────────────────────────────
def call_instructor(model, prompt, response_model):
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
    )
    if isinstance(result, tuple):
        return result[0]
    return result

def to_dict(x):
    if hasattr(x, "model_dump"): return x.model_dump()
    if hasattr(x, "dict"):       return x.dict()
    return x

# ── Router call (the stage that was missing from original RQ5) ────────────────
def run_router(data: str, query: str) -> float:
    """
    Fires the router LLM call and returns its wall time in seconds.
    Uses tool_calling_client (same as Agent.run_agent).
    We don't need the tool_call result — we just need the timing.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    s = time.perf_counter()
    tool_calling_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    return time.perf_counter() - s

# ── Stage helpers ─────────────────────────────────────────────────────────────
def _extract_config(data, analysis):
    cfg = to_dict(call_instructor(
        MODEL,
        CHART_CONFIGURATION_PROMPT.format(data=data, analysis=analysis),
        VisualizationConfig
    ))
    if isinstance(cfg.get("charttype"), ChartType):
        cfg["charttype"] = cfg["charttype"].value
    return cfg

def _justify_type(data, analysis):
    charttypes = {ct.name for ct in ChartType}
    parsed = call_instructor(
        MODEL,
        CREATE_CHART_TYPE_JUSTIFICATION_PROMPT.format(
            charttypes=charttypes, analysis=analysis, data=data
        ),
        ChartTypeJustification
    )
    return parsed.chart_type.value if hasattr(parsed, "chart_type") else to_dict(parsed).get("chart_type")

# ── Pipelines WITH router ──────────────────────────────────────────────────────
def run_serial_with_router(data: str, query: str) -> dict:
    """Router → Analysis → Config → Type  (all sequential)"""
    t0 = time.perf_counter()

    # Stage 0: Router
    s = time.perf_counter()
    run_router(data, query)
    t_router = time.perf_counter() - s

    # Stage 1: Data analysis
    s = time.perf_counter()
    analysis = to_dict(call_instructor(
        MODEL, DATA_ANALYSIS_PROMPT.format(data=data, query=query), DataAnalysis
    ))
    t_analysis = time.perf_counter() - s

    # Stage 2: Chart config
    s = time.perf_counter()
    cfg = _extract_config(data, analysis)
    t_config = time.perf_counter() - s

    # Stage 3: Chart type
    s = time.perf_counter()
    _justify_type(data, analysis)
    t_type = time.perf_counter() - s

    t_total = time.perf_counter() - t0
    return dict(t_router=t_router, t_analysis=t_analysis,
                t_config=t_config, t_type=t_type, t_total=t_total)


def run_parallel_with_router(data: str, query: str) -> dict:
    """Router → Analysis → [Config ∥ Type]"""
    t0 = time.perf_counter()

    # Stage 0: Router (sequential — must decide before anything else)
    s = time.perf_counter()
    run_router(data, query)
    t_router = time.perf_counter() - s

    # Stage 1: Analysis (sequential — config+type both depend on it)
    s = time.perf_counter()
    analysis = to_dict(call_instructor(
        MODEL, DATA_ANALYSIS_PROMPT.format(data=data, query=query), DataAnalysis
    ))
    t_analysis = time.perf_counter() - s

    # Stages 2+3: run concurrently
    s_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_cfg  = executor.submit(_extract_config, data, analysis)
        fut_type = executor.submit(_justify_type, data, analysis)

        s = time.perf_counter()
        cfg = fut_cfg.result()
        t_config = time.perf_counter() - s

        s = time.perf_counter()
        _justify_type_result = fut_type.result()
        t_type = time.perf_counter() - s

    t_parallel_wall = time.perf_counter() - s_parallel
    t_total = time.perf_counter() - t0

    return dict(t_router=t_router, t_analysis=t_analysis,
                t_config=t_config, t_type=t_type,
                t_parallel_wall=t_parallel_wall, t_total=t_total)


# ── Throughput experiment (same structure as original notebook) ───────────────
def run_throughput_experiment(run_fn, concurrency: int) -> dict:
    n_requests = concurrency
    jobs = [(MD_TABLE, QUERIES[i % len(QUERIES)]) for i in range(n_requests)]
    individual_latencies = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_fn, d, q): i for i, (d, q) in enumerate(jobs)}
        for fut in as_completed(futures):
            result = fut.result()
            individual_latencies.append(result["t_total"])

    wall_time = time.perf_counter() - wall_start
    return dict(
        concurrency=concurrency,
        wall_time=wall_time,
        requests_per_sec=n_requests / wall_time,
        mean_latency=statistics.mean(individual_latencies),
        individual_latencies=individual_latencies,
    )


# ── Single-request serial vs parallel latency (like Experiment 1 in notebook) ─
def run_single_request_benchmark():
    print("\n=== Single-request latency (averaged over N_RUNS × QUERIES) ===")
    serial_results   = []
    parallel_results = []

    for run_idx in range(N_RUNS):
        for q_idx, query in enumerate(QUERIES):
            print(f"  [Serial+Router]   run={run_idx+1}  q={q_idx+1} ...", end=" ", flush=True)
            r = run_serial_with_router(MD_TABLE, query)
            r.update(run=run_idx, query_idx=q_idx)
            serial_results.append(r)
            print(f"{r['t_total']:.2f}s")

    for run_idx in range(N_RUNS):
        for q_idx, query in enumerate(QUERIES):
            print(f"  [Parallel+Router] run={run_idx+1}  q={q_idx+1} ...", end=" ", flush=True)
            r = run_parallel_with_router(MD_TABLE, query)
            r.update(run=run_idx, query_idx=q_idx)
            parallel_results.append(r)
            print(f"{r['t_total']:.2f}s")

    return serial_results, parallel_results


# ── Throughput benchmark ───────────────────────────────────────────────────────
def run_throughput_benchmark():
    print("\n=== Throughput under concurrent load ===")
    throughput_serial   = []
    throughput_parallel = []

    for c in CONCURRENCY_LEVELS:
        print(f"\n── Concurrency = {c} ──")

        print(f"  Serial+Router   ...", end=" ", flush=True)
        r_s = run_throughput_experiment(run_serial_with_router, concurrency=c)
        throughput_serial.append(r_s)
        print(f"wall={r_s['wall_time']:.2f}s  RPS={r_s['requests_per_sec']:.3f}")

        print(f"  Parallel+Router ...", end=" ", flush=True)
        r_p = run_throughput_experiment(run_parallel_with_router, concurrency=c)
        throughput_parallel.append(r_p)
        print(f"wall={r_p['wall_time']:.2f}s  RPS={r_p['requests_per_sec']:.3f}")

    return throughput_serial, throughput_parallel


# ── Gantt (single representative request) ────────────────────────────────────
def run_gantt_benchmark():
    print("\n=== Running single timed requests for Gantt ===")
    query = QUERIES[0]

    # Serial timed
    t0 = time.perf_counter()
    stages_serial = []
    s = time.perf_counter() - t0
    run_router(MD_TABLE, query)
    stages_serial.append(("Router", s, time.perf_counter() - t0))

    s = time.perf_counter() - t0
    analysis = to_dict(call_instructor(
        MODEL, DATA_ANALYSIS_PROMPT.format(data=MD_TABLE, query=query), DataAnalysis
    ))
    stages_serial.append(("Data analysis", s, time.perf_counter() - t0))

    s = time.perf_counter() - t0
    _extract_config(MD_TABLE, analysis)
    stages_serial.append(("Chart config", s, time.perf_counter() - t0))

    s = time.perf_counter() - t0
    _justify_type(MD_TABLE, analysis)
    stages_serial.append(("Chart type", s, time.perf_counter() - t0))

    # Parallel timed
    t0 = time.perf_counter()
    stages_parallel = []

    s = time.perf_counter() - t0
    run_router(MD_TABLE, query)
    stages_parallel.append(("Router", s, time.perf_counter() - t0))

    s = time.perf_counter() - t0
    analysis = to_dict(call_instructor(
        MODEL, DATA_ANALYSIS_PROMPT.format(data=MD_TABLE, query=query), DataAnalysis
    ))
    stages_parallel.append(("Data analysis", s, time.perf_counter() - t0))

    def timed_config():
        s = time.perf_counter(); r = _extract_config(MD_TABLE, analysis); return r, time.perf_counter()-s
    def timed_type():
        s = time.perf_counter(); r = _justify_type(MD_TABLE, analysis); return r, time.perf_counter()-s

    par_start = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_cfg  = ex.submit(timed_config)
        f_type = ex.submit(timed_type)
        (cfg, dur_cfg)    = f_cfg.result()
        (ctype, dur_type) = f_type.result()

    stages_parallel.append(("Chart config (∥)", par_start, par_start + dur_cfg))
    stages_parallel.append(("Chart type (∥)",   par_start, par_start + dur_type))

    print("Gantt data collected.")
    return stages_serial, stages_parallel


# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = {
    "Router":           "#8B2252",
    "Data analysis":    "#263753",
    "Chart config":     "#3C547B",
    "Chart type":       "#5E86C5",
    "Chart config (∥)": "#3C547B",
    "Chart type (∥)":   "#5E86C5",
}

def plot_gantt(stages_serial, stages_parallel):
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=False)
    fig.suptitle("RQ5 (with Router) — Stage Timeline: Serial vs Parallel",
                 fontsize=14, fontweight="bold", x=0.05, ha="left")

    def draw_gantt(ax, stages, title):
        for name, start, end in stages:
            ax.barh(name, end-start, left=start, color=COLORS.get(name, "#888"),
                    edgecolor="white", linewidth=0.5, height=0.5)
            ax.text(end+0.2, name, f"{end-start:.1f}s", va="center", fontsize=9, color="#444")
        total = max(end for _, _, end in stages)
        ax.axvline(total, color="#D85A30", linewidth=1.2, linestyle="--")
        ax.text(total+0.3, len(stages)-0.5, f"Total: {total:.1f}s",
                color="#D85A30", fontsize=9, va="top")
        ax.set_xlabel("Time (seconds)")
        ax.set_title(title)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    draw_gantt(axes[0], stages_serial,   "Serial (Router + Analysis + Config + Type)")
    draw_gantt(axes[1], stages_parallel, "Parallel (Router + Analysis + [Config ∥ Type])")
    plt.tight_layout()
    plt.savefig("rq5_router_gantt.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rq5_router_gantt.png")


def plot_latency_comparison(serial_results, parallel_results):
    serial_totals   = [r["t_total"] for r in serial_results]
    parallel_totals = [r["t_total"] for r in parallel_results]

    # Stage means
    def mean(lst, key): return statistics.mean(r[key] for r in lst)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RQ5 (with Router) — Serial vs Parallel: End-to-End Latency",
                 fontsize=14, fontweight="bold", x=0.05, ha="left")

    # Box plot
    ax = axes[0]
    bp = ax.boxplot([serial_totals, parallel_totals],
                    labels=["Serial\n(+Router)", "Parallel\n(+Router)"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#3C547B")
    bp["boxes"][1].set_facecolor("#1D9E75")
    for median in bp["medians"]:
        median.set_color("white")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Distribution across all queries × runs")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # Stacked bar
    ax2 = axes[1]
    stages = ["t_router", "t_analysis", "t_config", "t_type"]
    colors = [COLORS["Router"], COLORS["Data analysis"], COLORS["Chart config"], COLORS["Chart type"]]
    labels = ["Router", "Data analysis", "Chart config", "Chart type"]

    for col_idx, (results, label) in enumerate([
        (serial_results,   "Serial\n(+Router)"),
        (parallel_results, "Parallel\n(+Router)")
    ]):
        bottom = 0
        for stage, color, slabel in zip(stages, colors, labels):
            val = mean(results, stage)
            ax2.bar(col_idx, val, bottom=bottom, color=color,
                    label=slabel if col_idx == 0 else "")
            bottom += val

    diff = statistics.mean(serial_totals) - statistics.mean(parallel_totals)
    pct  = diff / statistics.mean(serial_totals) * 100
    ax2.annotate(f"−{diff:.1f}s ({pct:.0f}%)",
                 xy=(1, statistics.mean(parallel_totals)),
                 xytext=(1.15, statistics.mean(parallel_totals)),
                 color="#D85A30", fontsize=9, va="center")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Serial\n(+Router)", "Parallel\n(+Router)"])
    ax2.set_ylabel("Mean latency (seconds)")
    ax2.set_title("Stage breakdown (mean)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("rq5_router_latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rq5_router_latency_comparison.png")


def plot_throughput(throughput_serial, throughput_parallel):
    levels        = CONCURRENCY_LEVELS
    rps_serial    = [r["requests_per_sec"] for r in throughput_serial]
    rps_parallel  = [r["requests_per_sec"] for r in throughput_parallel]
    lat_serial    = [r["mean_latency"]     for r in throughput_serial]
    lat_parallel  = [r["mean_latency"]     for r in throughput_parallel]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RQ5 (with Router) — Throughput Under Concurrent Load",
                 fontsize=14, fontweight="bold", x=0.05, ha="left")

    ax = axes[0]
    ax.plot(levels, rps_serial,   "o-", color="#3C547B", linewidth=2, label="Serial (+Router)")
    ax.plot(levels, rps_parallel, "o-", color="#1D9E75", linewidth=2, label="Parallel (+Router)")
    ax.set_xlabel("Concurrent requests"); ax.set_ylabel("Throughput (req / second)")
    ax.set_title("Throughput vs concurrency")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False); ax.set_xticks(levels)

    ax2 = axes[1]
    ax2.plot(levels, lat_serial,   "o-", color="#3C547B", linewidth=2, label="Serial (+Router)")
    ax2.plot(levels, lat_parallel, "o-", color="#1D9E75", linewidth=2, label="Parallel (+Router)")
    ax2.set_xlabel("Concurrent requests"); ax2.set_ylabel("Mean per-request latency (s)")
    ax2.set_title("Latency under load")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.spines[["top","right"]].set_visible(False); ax2.set_xticks(levels)

    plt.tight_layout()
    plt.savefig("rq5_router_throughput.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rq5_router_throughput.png")


def save_csv(throughput_serial, throughput_parallel):
    rows = []
    for s, p in zip(throughput_serial, throughput_parallel):
        gain = (p["requests_per_sec"] - s["requests_per_sec"]) / s["requests_per_sec"] * 100
        rows.append({
            "Concurrency": s["concurrency"],
            "Serial wall time (s)":    round(s["wall_time"], 2),
            "Parallel wall time (s)":  round(p["wall_time"], 2),
            "Serial RPS":              round(s["requests_per_sec"], 3),
            "Parallel RPS":            round(p["requests_per_sec"], 3),
            "Throughput gain":         f"{gain:+.1f}%",
        })
    with open("rq5_router_throughput_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Saved: rq5_router_throughput_results.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Queries: {len(QUERIES)}  Runs: {N_RUNS}  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Estimated total LLM calls (single-request): {len(QUERIES)*N_RUNS*2*4}")  # 4 stages × 2 modes

    # 1. Single-request latency benchmark
    serial_results, parallel_results = run_single_request_benchmark()
    plot_latency_comparison(serial_results, parallel_results)

    # 2. Throughput under concurrent load
    throughput_serial, throughput_parallel = run_throughput_benchmark()
    plot_throughput(throughput_serial, throughput_parallel)
    save_csv(throughput_serial, throughput_parallel)

    # 3. Gantt chart for a single representative request
    stages_serial, stages_parallel = run_gantt_benchmark()
    plot_gantt(stages_serial, stages_parallel)

    print("\n✅ All done. Outputs: rq5_router_*.png + rq5_router_throughput_results.csv")
