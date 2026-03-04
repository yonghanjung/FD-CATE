#!/usr/bin/env python3
"""Render a lightweight SVG bar chart from the quick benchmark golden JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = ROOT / "tests" / "benchmark_quick_reference.json"
OUTPUT_SVG = ROOT / "benchmark_quick_rmse.svg"


def main() -> None:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = [
        ("clean", payload["results"]["clean"]),
        ("weak-overlap", payload["results"]["weak-overlap"]),
        (
            "aggregate mean",
            {
                "fd-pi_rmse": payload["results"]["aggregate_mean_rmse"]["fd-pi"],
                "fd-dr_rmse": payload["results"]["aggregate_mean_rmse"]["fd-dr"],
                "fd-r_rmse": payload["results"]["aggregate_mean_rmse"]["fd-r"],
            },
        ),
    ]

    series = [
        ("FD-PI", "fd-pi_rmse", "#2563eb"),
        ("FD-DR", "fd-dr_rmse", "#dc2626"),
        ("FD-R", "fd-r_rmse", "#16a34a"),
    ]

    width = 860
    height = 460
    left = 80
    right = 30
    top = 50
    bottom = 100
    chart_w = width - left - right
    chart_h = height - top - bottom

    y_max = 1.0
    group_count = len(rows)
    group_w = chart_w / group_count
    bar_w = group_w * 0.18
    bar_gap = group_w * 0.05

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="FD-CATE quick benchmark RMSE plot">'
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(
        '<text x="430" y="26" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#111827">FD-CATE Quick Benchmark RMSE (XGB)</text>'
    )

    # Axes
    x0 = left
    y0 = top + chart_h
    lines.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#111827" stroke-width="2"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{left + chart_w}" y2="{y0}" stroke="#111827" stroke-width="2"/>')

    # Y ticks
    for k in range(6):
        value = y_max * (k / 5)
        y = y0 - (chart_h * value / y_max)
        lines.append(f'<line x1="{x0 - 6}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#111827" stroke-width="1"/>')
        lines.append(
            f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{value:.1f}</text>'
        )
        if k > 0:
            lines.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )

    lines.append(
        f'<text x="18" y="{top + chart_h / 2:.1f}" transform="rotate(-90 18 {top + chart_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#111827">RMSE (lower is better)</text>'
    )

    # Bars
    for gi, (label, metrics) in enumerate(rows):
        gx = left + gi * group_w + group_w * 0.18
        for si, (_, key, color) in enumerate(series):
            v = float(metrics[key])
            bar_h = chart_h * min(v, y_max) / y_max
            x = gx + si * (bar_w + bar_gap)
            y = y0 - bar_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.90"/>'
            )
            lines.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 6, top + 12):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{v:.3f}</text>'
            )
        lines.append(
            f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{y0 + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{label}</text>'
        )

    # Legend
    legend_x = left + 12
    legend_y = height - 42
    for i, (name, _, color) in enumerate(series):
        lx = legend_x + i * 150
        lines.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        lines.append(
            f'<text x="{lx + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{name}</text>'
        )

    lines.append("</svg>")
    OUTPUT_SVG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
