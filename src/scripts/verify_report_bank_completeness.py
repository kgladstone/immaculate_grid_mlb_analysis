#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
REPORT_BANK_JSON = ROOT / "bin" / "json" / "report_bank_data.json"
REPORT_BANK_PY = ROOT / "src" / "app" / "services" / "report_bank.py"


def _load_report_bank_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in raw if isinstance(row, dict)]


def _extract_functions_keys(path: Path) -> set[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FUNCTIONS":
                    if isinstance(node.value, ast.Dict):
                        keys = set()
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                keys.add(key.value)
                        return keys
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "FUNCTIONS":
                if isinstance(node.value, ast.Dict):
                    keys = set()
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
                    return keys
    raise ValueError(f"Could not find FUNCTIONS dict in {path}")


def _normalize_title(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _report(msg: str) -> None:
    print(f"[verify-report-bank] {msg}")


def verify(expect_titles: Iterable[str]) -> int:
    rows = _load_report_bank_json(REPORT_BANK_JSON)
    function_keys = _extract_functions_keys(REPORT_BANK_PY)

    json_funcs = [str(r.get("func", "")).strip() for r in rows]
    json_titles = [str(r.get("title", "")).strip() for r in rows]

    missing_function_mapping = sorted({f for f in json_funcs if f and f not in function_keys})
    unused_function_mapping = sorted({f for f in function_keys if f not in set(json_funcs)})
    duplicate_titles = sorted({t for t in json_titles if t and json_titles.count(t) > 1})
    empty_title_rows = [idx for idx, t in enumerate(json_titles) if not t]
    empty_func_rows = [idx for idx, f in enumerate(json_funcs) if not f]

    expected_missing = []
    titles_norm = {_normalize_title(t): t for t in json_titles if t}
    for expected in expect_titles:
        if _normalize_title(expected) not in titles_norm:
            expected_missing.append(expected)

    _report(f"Rows in report bank JSON: {len(rows)}")
    _report(f"FUNCTIONS keys in report_bank.py: {len(function_keys)}")

    if missing_function_mapping:
        _report("ERROR: report_bank_data.json contains funcs not registered in FUNCTIONS:")
        for f in missing_function_mapping:
            _report(f"  - {f}")
    else:
        _report("OK: all JSON funcs are registered in FUNCTIONS.")

    if empty_title_rows:
        _report(f"ERROR: rows with empty title: {empty_title_rows}")
    else:
        _report("OK: all rows have non-empty title.")

    if empty_func_rows:
        _report(f"ERROR: rows with empty func: {empty_func_rows}")
    else:
        _report("OK: all rows have non-empty func.")

    if duplicate_titles:
        _report("WARN: duplicate titles found:")
        for t in duplicate_titles:
            _report(f"  - {t}")
    else:
        _report("OK: no duplicate titles.")

    if unused_function_mapping:
        _report("WARN: FUNCTIONS entries not referenced in report_bank_data.json:")
        for f in unused_function_mapping:
            _report(f"  - {f}")
    else:
        _report("OK: all FUNCTIONS entries are used in report_bank_data.json.")

    if expected_missing:
        _report("ERROR: expected report title(s) missing from report_bank_data.json:")
        for t in expected_missing:
            _report(f"  - {t}")
    elif expect_titles:
        _report("OK: all expected report titles are present.")

    has_errors = bool(
        missing_function_mapping
        or empty_title_rows
        or empty_func_rows
        or expected_missing
    )
    return 1 if has_errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify report_bank_data.json completeness against report_bank.py")
    parser.add_argument(
        "--expect-title",
        action="append",
        default=[],
        help="Require this report title to exist in report_bank_data.json (can be repeated).",
    )
    args = parser.parse_args()
    raise SystemExit(verify(args.expect_title))


if __name__ == "__main__":
    main()
