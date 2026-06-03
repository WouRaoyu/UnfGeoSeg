"""Small reporting helper: write a list-of-dict table to CSV + Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence


def write_table(
    rows: List[Dict],
    out_stem: str | Path,
    columns: Sequence[str] | None = None,
    float_fmt: str = "{:.4f}",
) -> None:
    """Write ``rows`` to ``<out_stem>.csv`` and ``<out_stem>.md``."""
    import csv

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        cols = list(columns or [])
    else:
        cols = list(columns or rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return "" if v is None else str(v)

    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    out_stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
