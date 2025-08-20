#!/usr/bin/env python
"""Generate a simple coverage badge SVG from coverage.xml.

Usage: python scripts/make_coverage_badge.py coverage.xml coverage-badge.svg

Color thresholds (line coverage %):
  >= 90: brightgreen
  >= 80: green
  >= 70: yellowgreen
  >= 60: yellow
  >= 50: orange
  else: red
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Threshold constants (line coverage percentage cutoffs)
THRESHOLD_BRIGHTGREEN: float = 90.0
THRESHOLD_GREEN: float = 80.0
THRESHOLD_YELLOWGREEN: float = 70.0
THRESHOLD_YELLOW: float = 60.0
THRESHOLD_ORANGE: float = 50.0

# Layout constants
CHAR_WIDTH: int = 6
H_PADDING: int = 20
HEIGHT: int = 20

# Expected CLI argument count (program name + 2 args)
EXPECTED_ARGS: int = 3


def pick_color(pct: float) -> str:
    if pct >= THRESHOLD_BRIGHTGREEN:
        return "#4c1"  # brightgreen
    if pct >= THRESHOLD_GREEN:
        return "#97CA00"  # green
    if pct >= THRESHOLD_YELLOWGREEN:
        return "#a4a61d"  # yellowgreen
    if pct >= THRESHOLD_YELLOW:
        return "#dfb317"  # yellow
    if pct >= THRESHOLD_ORANGE:
        return "#fe7d37"  # orange
    return "#e05d44"  # red


def main(in_path: str, out_path: str) -> None:
    xml_path = Path(in_path)
    if not xml_path.exists():
        raise SystemExit(f"Input coverage file not found: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:  # pragma: no cover - defensive
        raise SystemExit(f"Failed to parse coverage xml: {e}") from e
    line_rate_attr = root.attrib.get("line-rate")
    if line_rate_attr is None:
        raise SystemExit("coverage.xml missing line-rate attribute")
    try:
        pct = float(line_rate_attr) * 100.0
    except ValueError as e:
        raise SystemExit(f"Invalid line-rate value: {line_rate_attr}") from e
    pct_text = f"{pct:.2f}%"
    color = pick_color(pct)
    label = "coverage"
    value = pct_text
    label_w = CHAR_WIDTH * len(label) + H_PADDING
    value_w = CHAR_WIDTH * len(value) + H_PADDING
    total_w = label_w + value_w
    height = HEIGHT
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{total_w}' height='{height}'>
  <linearGradient id='s' x2='0' y2='100%'>
    <stop offset='0' stop-color='#bbb' stop-opacity='.1'/>
    <stop offset='1' stop-opacity='.1'/>
  </linearGradient>
  <rect rx='3' width='{total_w}' height='{height}' fill='#555'/>
  <rect rx='3' x='{label_w}' width='{value_w}' height='{height}' fill='{color}'/>
  <rect rx='3' width='{total_w}' height='{height}' fill='url(#s)'/>
  <g fill='#fff' text-anchor='middle' font-family='Verdana,Geneva,DejaVu Sans,sans-serif' font-size='11'>
    <text x='{label_w / 2:.1f}' y='14'>{label}</text>
    <text x='{label_w + value_w / 2:.1f}' y='14'>{value}</text>
  </g>
</svg>"""
    Path(out_path).write_text(svg)


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != EXPECTED_ARGS:
        print("Usage: make_coverage_badge.py coverage.xml coverage-badge.svg", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
