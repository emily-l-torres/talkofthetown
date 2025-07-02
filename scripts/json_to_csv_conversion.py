#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_to_csv_conversion.py
---------------------

Convert one or many Yelp-style JSONL files to CSV (or Parquet) while retaining:
  • parent mapping columns (e.g. 'hours')
  • every flattened child column (e.g. 'hours.Monday')

Options
-------
python json_to_relational.py                           # all *.json in ./data → CSV
python json_to_relational.py data/*.json --parquet     # selected files → Parquet
python json_to_relational.py business.json --out out.csv
python json_to_relational.py business.json --batch 25000 --expand-dicts

Flags
-----
--parquet        Write .parquet instead of .csv
--out <file>     Only when a single JSON file is passed
--batch <n>      Flush Parquet every n rows (default 100_000)
--expand-dicts   Parse stringified dict values into extra columns
"""

import ast
import csv
import glob
import re
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    import orjson as _json
    _loads = _json.loads
except ImportError:
    import json as _json  # noqa: F401
    _loads = _json.loads


# ─────────────────────────── helpers ────────────────────────────
_DICT_RE = re.compile(r"\s*\{.*\}\s*$")


def _maybe_parse_dict_str(v):
    """Return dict if v looks like a JSON/Python dict string, else unchanged."""
    if isinstance(v, str) and _DICT_RE.match(v):
        try:
            return ast.literal_eval(v)
        except Exception:  # malformed → leave as-is
            return v
    return v


def _flatten_keys(d: Mapping, parent: str = "", expand=False):
    """
    Yield dotted keys for *all* entries in d.
    If expand=True, also flatten dict-like strings.
    """
    for k, v in d.items():
        full = f"{parent}.{k}" if parent else k
        yield full                      # parent mapping key always emitted

        if expand:
            v = _maybe_parse_dict_str(v)

        if isinstance(v, Mapping):
            yield from _flatten_keys(v, full, expand)


def _nested_get(d, dotted):
    """Return value at dotted path; None if any step missing."""
    for part in dotted.split("."):
        if not isinstance(d, Mapping):
            return None
        d = d.get(part)
        if d is None:
            return None
    return d


def build_header(paths, expand=False):
    cols = set()
    for p in paths:
        with p.open("rb") as fh:
            for line in fh:
                cols.update(_flatten_keys(_loads(line), expand=expand))
    return sorted(cols)


class Writer:
    """CSV or Parquet writer with buffering."""
    def __init__(self, path: Path, header, parquet, batch_rows):
        self.parquet = parquet
        self.header = header
        self.batch_rows = batch_rows
        if parquet:
            import pandas as pd
            self._pd, self._buf, self._path = pd, [], path
        else:
            self._fh = path.open("w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._fh)
            self._csv.writerow(header)

    def write(self, row):
        if self.parquet:
            self._buf.append(row)
            if len(self._buf) >= self.batch_rows:
                self.flush()
        else:
            self._csv.writerow(row)

    def flush(self):
        if self.parquet and self._buf:
            df = self._pd.DataFrame(self._buf, columns=self.header)
            df.to_parquet(
                self._path,
                index=False,
                append=self._path.exists(),
            )
            self._buf.clear()

    def close(self):
        self.flush()
        if not self.parquet:
            self._fh.close()


# ────────────────────── CLI parsing (minimal) ─────────────────────────
parquet_flag = "--parquet" in sys.argv
expand_flag = "--expand-dicts" in sys.argv
batch_size = 100_000
if "--batch" in sys.argv:
    batch_size = int(sys.argv[sys.argv.index("--batch") + 1])

fixed_out = None
if "--out" in sys.argv:
    idx = sys.argv.index("--out") + 1
    fixed_out = Path(sys.argv[idx]).expanduser().resolve()

positional = [a for a in sys.argv[1:] if not a.startswith("-")]
if positional:
    json_paths = [
        Path(pat).expanduser().resolve()
        for pat in positional
        for p in glob.glob(pat)
    ]
else:
    json_paths = list(Path("data").glob("*.json"))

if not json_paths:
    sys.exit("No JSON files found.")

if fixed_out and len(json_paths) > 1:
    sys.exit("--out can be used only with a single JSON input")

# ────────────────────── header discovery (pass 1) ─────────────────────
header = build_header(json_paths, expand=expand_flag)
print(f"Discovered {len(header)} columns")

# ────────────────────── conversion loop (pass 2) ──────────────────────
for src in json_paths:
    dst = fixed_out or src.with_suffix(".parquet" if parquet_flag else ".csv")
    print(f"Converting {src.name} -> {dst.name}")

    writer = Writer(dst, header, parquet_flag, batch_size)
    with src.open("rb") as fh:
        for line in fh:
            rec = _loads(line)
            row = []
            for col in header:
                val = _nested_get(rec, col)
                if expand_flag:
                    val = _maybe_parse_dict_str(val)
                if val is None:
                    row.append("")
                elif isinstance(val, (Mapping, list)):
                    row.append(_json.dumps(val).decode("utf-8") if hasattr(_json, "dumps") else
                               __import__("json").dumps(val, ensure_ascii=False))
                else:
                    row.append(str(val))
            writer.write(row)
    writer.close()
    print(f"{dst.name} written")

print("Conversion finished")
