from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate post-eval JSON metrics into paper-ready mean/std CSV tables."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/results/local_runs"),
    )
    parser.add_argument(
        "--glob",
        default="**/post_eval/test/test_metrics.json",
        help="Pattern relative to runs-root. Use '**/post_eval/test/test_metrics.json' by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/tables/post_eval_summary.csv"),
    )
    return parser.parse_args()


def flatten(prefix: str, value, output: dict[str, float | str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}_{key}" if prefix else key, child, output)
    elif isinstance(value, (int, float, str)):
        output[prefix] = value


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.runs_root.glob(args.glob))
    rows = []
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": str(path)}
        flatten("", payload, row)
        row["model"] = str(payload.get("model", path.parts[-5] if len(path.parts) > 5 else "unknown"))
        row["fold"] = next((part for part in path.parts if part.startswith("test_D")), "standard")
        rows.append(row)

    numeric_by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = str(row.get("model", "unknown"))
        split = str(row.get("split", "test"))
        for key, value in row.items():
            if key in {"path", "model", "fold", "split"}:
                continue
            if isinstance(value, (int, float)):
                numeric_by_group[(model, split)][key].append(float(value))

    summary_rows = []
    for (model, split), metrics in sorted(numeric_by_group.items()):
        summary = {"model": model, "split": split, "folds": len(next(iter(metrics.values()), []))}
        for key, values in sorted(metrics.items()):
            arr = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(arr.mean())
            summary[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        summary_rows.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row})
        preferred = ["model", "split", "folds"]
        fieldnames = preferred + [key for key in fieldnames if key not in preferred]
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        args.output.write_text("", encoding="utf-8")

    raw_output = args.output.with_name(args.output.stem + "_raw.csv")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with raw_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raw_output.write_text("", encoding="utf-8")

    print(f"Found metric files: {len(metric_paths)}")
    print(f"Saved summary: {args.output}")
    print(f"Saved raw table: {raw_output}")


if __name__ == "__main__":
    main()
