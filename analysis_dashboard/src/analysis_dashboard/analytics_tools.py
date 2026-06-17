"""
analysis_dashboard/analytics_tools.py

LangChain @tools for operational metrics, trend analysis, and
chart generation using matplotlib / seaborn.

Tools

  calculate_latency_trends   - moving-average latency over time
  calculate_token_trends     - token consumption pattern analysis
  calculate_error_frequency  - error/failure count over time windows
  generate_performance_chart - produce and optionally save a PNG chart
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain.tools import tool

from analysis_dashboard.settings import settings

matplotlib.use("Agg")  # non-interactive backend

logger = logging.getLogger("analysis_agent")
sns.set_theme(style="darkgrid", palette="muted")


# In-process chart image registry
# Keeps base64 blobs out of the LLM message history entirely.

_chart_registry: dict[str, str] = {}  # chart_id -> base64 PNG string


def get_chart_image(chart_id: str) -> str | None:
    """
    Called by the Streamlit UI to retrieve a base64 PNG by chart_id.
    Returns None if the id is not found.
    """
    return _chart_registry.get(chart_id)


def get_all_chart_ids() -> list[str]:
    """Return all chart ids produced so far this session."""
    return list(_chart_registry.keys())


# Helpers


def _parse_log_list(logs_json: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(logs_json)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return []
    except Exception:
        return []


def _moving_average(values: list[float], window: int = 3) -> list[float]:
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return result


def _ensure_charts_dir() -> str:
    charts_dir = settings.charts_output_dir
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir


# Tools


@tool
def calculate_latency_trends(logs_json: str, window: int = 3) -> str:
    """
    Calculate moving-average latency trends from MCP tool call log entries.

    Extracts latency_ms values from tool_invocation and resource_read
    entries, computes a rolling average, and returns the trend data as JSON.

    Args:
        logs_json: JSON string from search_logs_semantic or list_recent_logs.
        window:    Moving-average window size (default 3).

    Returns:
        JSON string with raw latencies, moving averages, mean, p95, p99.
    """
    logger.info("calculate_latency_trends called | window=%d", window)
    entries = _parse_log_list(logs_json)

    latency_points: list[dict[str, Any]] = []
    for entry in entries:
        metadata = entry.get("metadata", {})
        latency_ms = metadata.get("latency_ms")
        if latency_ms is not None:
            latency_points.append(
                {
                    "timestamp": entry.get("timestamp", 0.0),
                    "latency_ms": float(latency_ms),
                    "tool_name": metadata.get(
                        "tool_name", entry.get("component", "unknown")
                    ),
                    "session_id": entry.get("session_id", ""),
                }
            )

    if not latency_points:
        return json.dumps(
            {
                "status": "no_data",
                "message": "No latency_ms values found. This is a final result — "
                "do not retry. Proceed to summarise.",
                "conclusion": "No latency data is available in the current logs.",
            }
        )

    latency_points.sort(key=lambda x: x["timestamp"])
    raw = [p["latency_ms"] for p in latency_points]
    ma = _moving_average(raw, window)

    result = {
        "status": "ok",
        "count": len(raw),
        "mean_ms": float(np.mean(raw)),
        "p50_ms": float(np.percentile(raw, 50)),
        "p95_ms": float(np.percentile(raw, 95)),
        "p99_ms": float(np.percentile(raw, 99)),
        "moving_average_window": window,
        "moving_averages": ma,
        "raw_latencies": raw,
        "points": latency_points,
    }
    logger.info(
        "calculate_latency_trends: count=%d mean=%.1f p95=%.1f",
        len(raw),
        result["mean_ms"],
        result["p95_ms"],
    )
    return json.dumps(result, indent=2, default=str)


@tool
def calculate_token_trends(logs_json: str) -> str:
    """
    Analyse token consumption patterns from sampling_request log entries.

    Extracts token_count metadata from sampling entries and computes
    aggregated statistics per session and overall.

    Args:
        logs_json: JSON string from search_logs_semantic or list_recent_logs.

    Returns:
        JSON with per-session token sums and overall statistics.
    """
    logger.info("calculate_token_trends called")
    entries = _parse_log_list(logs_json)

    token_points: list[dict[str, Any]] = []
    for entry in entries:
        # Accept any entry that has token_count or response_length in metadata
        metadata = entry.get("metadata", {})
        token_count = metadata.get("token_count") or metadata.get("response_length")

        # Also accept entries whose interaction type is sampling_request
        if (
            token_count is None
            and entry.get("mcp_interaction_type") != "sampling_request"
        ):
            continue
        if token_count is None:
            continue

        token_points.append(
            {
                "timestamp": entry.get("timestamp", 0.0),
                "token_count": int(token_count),
                "session_id": entry.get("session_id", "unknown"),
            }
        )

    if not token_points:
        return json.dumps(
            {
                "status": "no_data",
                "message": "No token_count values found in any log entries."
                "This is a final result — do not retry with different queries.",
                "conclusion": "Token consumption data is not available "
                "in the current logs. The sampling_request entries do not contain "
                "token_count or response_length metadata.",
            }
        )

    token_points.sort(key=lambda x: x["timestamp"])
    counts = [p["token_count"] for p in token_points]

    session_totals: dict[str, int] = {}
    for p in token_points:
        sid = p["session_id"]
        session_totals[sid] = session_totals.get(sid, 0) + p["token_count"]

    result = {
        "status": "ok",
        "total_sampling_requests": len(counts),
        "total_tokens": int(np.sum(counts)),
        "mean_tokens_per_request": float(np.mean(counts)),
        "p95_tokens": float(np.percentile(counts, 95)),
        "per_session_totals": session_totals,
        "points": token_points,
    }
    logger.info(
        "calculate_token_trends: total=%d mean=%.1f",
        result["total_tokens"],
        result["mean_tokens_per_request"],
    )
    return json.dumps(result, indent=2, default=str)


@tool
def calculate_error_frequency(logs_json: str) -> str:
    """
    Count error and failure events over time from log entries.

    Identifies entries with component paths containing "error" or
    metadata keys indicating failures, then groups them by session.

    Args:
        logs_json: JSON string from search_logs_semantic or list_recent_logs.

    Returns:
        JSON with error counts, time series, and session breakdown.
    """
    logger.info("calculate_error_frequency called")
    entries = _parse_log_list(logs_json)

    error_entries: list[dict[str, Any]] = []
    for entry in entries:
        component = entry.get("component", "").lower()
        content = entry.get("content", "").lower()
        metadata = entry.get("metadata", {})

        is_error = (
            "error" in component
            or "error" in content[:100]
            or "failed" in content[:100]
            or metadata.get("error")
            or "error" in str(metadata.get("level", "")).lower()
        )
        if is_error:
            error_entries.append(
                {
                    "timestamp": entry.get("timestamp", 0.0),
                    "component": entry.get("component", ""),
                    "content_snippet": entry.get("content", "")[:200],
                    "session_id": entry.get("session_id", ""),
                }
            )

    if not error_entries:
        return json.dumps(
            {
                "status": "ok",
                "total_errors": 0,
                "message": "No error entries found. This is a "
                "final result — do not retry.",
                "conclusion": "System appears healthy with no errors "
                "in the current logs.",
            }
        )

    error_entries.sort(key=lambda x: x["timestamp"])
    per_session: dict[str, int] = {}
    for e in error_entries:
        sid = e["session_id"]
        per_session[sid] = per_session.get(sid, 0) + 1

    result = {
        "status": "ok",
        "total_errors": len(error_entries),
        "per_session_counts": per_session,
        "error_entries": error_entries,
    }
    logger.info("calculate_error_frequency: total_errors=%d", result["total_errors"])
    return json.dumps(result, indent=2, default=str)


@tool
def generate_performance_chart(
    metric_json: str,
    chart_type: str = "latency",
    save_to_disk: bool = False,
    filename: str = "",
) -> str:
    """
    Generate a performance trend chart using matplotlib / seaborn.

    Accepts the output of calculate_latency_trends, calculate_token_trends,
    or calculate_error_frequency and produces a visual chart.

    The chart image is stored internally and displayed in the Streamlit UI
    automatically. Only a lightweight summary is returned to you.

    Args:
        metric_json:  JSON string from one of the metric calculation tools.
        chart_type:   One of "latency", "tokens", "errors", "combined".
        save_to_disk: If True, save the chart as a PNG file.
        filename:     Optional filename (without extension).

    Returns:
        JSON with "chart_id" (for UI retrieval), "chart_type", "description",
        and optionally "chart_path". Does NOT return raw image bytes.
    """
    logger.info(
        "generate_performance_chart | chart_type=%s | save=%s", chart_type, save_to_disk
    )
    import base64
    from io import BytesIO

    try:
        data = json.loads(metric_json)

        fig, axes = plt.subplots(
            1,
            2 if chart_type == "combined" else 1,
            figsize=(12 if chart_type == "combined" else 8, 5),
            dpi=100,
        )
        ax = axes[0] if chart_type == "combined" else axes
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        chart_path = ""
        description = ""

        # Latency chart
        if chart_type in ("latency", "combined"):
            raw = data.get("raw_latencies", [])
            ma = data.get("moving_averages", [])
            indices = list(range(len(raw)))

            ax.plot(
                indices,
                raw,
                color="#89b4fa",
                alpha=0.5,
                linewidth=1.2,
                label="Raw latency (ms)",
                marker="o",
                markersize=4,
            )
            if ma:
                ax.plot(
                    indices,
                    ma,
                    color="#f38ba8",
                    linewidth=2.5,
                    label=f"MA({data.get('moving_average_window', 3)})",
                )

            mean_val = data.get("mean_ms", 0)
            ax.axhline(
                mean_val,
                color="#a6e3a1",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_val:.0f}ms",
            )
            p95 = data.get("p95_ms", 0)
            ax.axhline(
                p95,
                color="#fab387",
                linestyle=":",
                linewidth=1.5,
                label=f"p95: {p95:.0f}ms",
            )

            ax.set_title("Tool Call Latency Trend", color="white", fontsize=13)
            ax.set_xlabel("Call Index", color="#cdd6f4")
            ax.set_ylabel("Latency (ms)", color="#cdd6f4")
            ax.tick_params(colors="#cdd6f4")
            ax.legend(facecolor="#313244", labelcolor="white", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#45475a")
            description = (
                f"Latency chart: {len(raw)} calls, "
                f"mean={mean_val:.0f}ms, p95={p95:.0f}ms"
            )

        # Token chart
        elif chart_type == "tokens":
            per_session = data.get("per_session_totals", {})
            if per_session:
                sessions = [s[:8] + "…" for s in per_session.keys()]
                totals = list(per_session.values())
                bars = ax.bar(
                    sessions, totals, color="#cba6f7", alpha=0.85, edgecolor="#313244"
                )
                for bar, val in zip(bars, totals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(totals) * 0.02,
                        str(val),
                        ha="center",
                        va="bottom",
                        color="white",
                        fontsize=8,
                    )
                ax.set_title(
                    "Token Consumption per Session", color="white", fontsize=13
                )
                ax.set_xlabel("Session ID (truncated)", color="#cdd6f4")
                ax.set_ylabel("Total Tokens", color="#cdd6f4")
                ax.tick_params(colors="#cdd6f4", axis="x", rotation=30)
                ax.tick_params(colors="#cdd6f4", axis="y")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#45475a")
                description = f"Token chart: {len(sessions)} sessions plotted"
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No token data available",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                description = "No token data to plot"

        # Error frequency chart
        elif chart_type == "errors":
            per_session = data.get("per_session_counts", {})
            if per_session:
                sessions = [s[:8] + "…" for s in per_session.keys()]
                counts = list(per_session.values())
                ax.barh(
                    sessions, counts, color="#f38ba8", alpha=0.85, edgecolor="#313244"
                )
                ax.set_title("Error Frequency per Session", color="white", fontsize=13)
                ax.set_xlabel("Error Count", color="#cdd6f4")
                ax.set_ylabel("Session ID", color="#cdd6f4")
                ax.tick_params(colors="#cdd6f4")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#45475a")
                description = f"Error chart: {data.get('total_errors', 0)} total errors"
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No errors found — system healthy ✓",
                    ha="center",
                    va="center",
                    color="#a6e3a1",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                description = "No errors detected"

        # Combined second panel placeholder
        if chart_type == "combined":
            ax2 = axes[1]
            ax2.set_facecolor("#1e1e2e")
            ax2.text(
                0.5,
                0.5,
                "Combined view — pass error JSON here",
                ha="center",
                va="center",
                color="white",
                transform=ax2.transAxes,
            )

        plt.tight_layout()

        # Encode to base64
        buf = BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        # Store in registry — NOT returned to LLM
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_id = f"{chart_type}_{ts}"
        _chart_registry[chart_id] = img_b64
        logger.info("Chart stored in registry with id: %s", chart_id)

        # Optionally save to disk
        if save_to_disk:
            charts_dir = _ensure_charts_dir()
            fname = (filename or chart_id) + ".png"
            chart_path = os.path.join(charts_dir, fname)
            fig.savefig(
                chart_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor()
            )
            logger.info("Chart saved to %s", chart_path)

        plt.close(fig)

        # Return only the lightweight summary to the LLM
        return json.dumps(
            {
                "status": "ok",
                "chart_type": chart_type,
                "chart_id": chart_id,  # UI uses this to fetch the image
                "description": description,
                "chart_path": chart_path,
                # NO image_base64 here — keeps token count tiny
            },
            indent=2,
        )

    except Exception as exc:
        logger.error("generate_performance_chart error: %s", exc)
        plt.close("all")
        return json.dumps({"status": "error", "message": str(exc)})
