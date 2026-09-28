from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.services.monthly_snapshot import generate_monthly_snapshot_pdf, rolling_report_bounds, report_filename
from config.constants import IMAGES_METADATA_CSV_PATH, MESSAGES_CSV_PATH


def _load_derived_images() -> pd.DataFrame:
    """Rebuild screenshot-level responses from the fuzzy-normalized derived CSV."""
    flat = pd.read_csv(IMAGES_METADATA_CSV_PATH)
    required = {"submitter", "date", "grid_number", "position", "response"}
    missing = required - set(flat.columns)
    if missing:
        raise ValueError(f"Derived image metadata is missing columns: {sorted(missing)}")
    index_columns = ["submitter", "date", "grid_number"]
    if "image_filename" in flat.columns:
        index_columns.append("image_filename")
    flat = flat.copy()
    flat["response"] = flat["response"].fillna("").astype(str)
    rows = []
    for keys, group in flat.groupby(index_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(index_columns, keys))
        record["responses"] = dict(zip(group["position"], group["response"]))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a rolling 90-day quarterly Immaculate Grid report.")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--month", help="Calendar-month report in YYYY-MM format.")
    period.add_argument("--start-date", help="Custom report start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Report ending date (YYYY-MM-DD); defaults to latest available data. Used with a 90-day lookback unless --start-date is given.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination PDF path.",
    )
    args = parser.parse_args()
    if args.start_date and not args.end_date:
        parser.error("--end-date is required with --start-date")
    if args.end_date and args.month:
        parser.error("--end-date cannot be used with --month")

    texts_df = pd.read_csv(MESSAGES_CSV_PATH)
    images_df = _load_derived_images()
    if not args.month and not args.start_date:
        end = args.end_date or min(pd.to_datetime(texts_df["date"]).max(), pd.Timestamp.now()).normalize()
        args.start_date, args.end_date = rolling_report_bounds(end)
    if args.output is None:
        if args.month:
            start = pd.Period(args.month, freq="M").start_time.normalize()
            end = pd.Period(args.month, freq="M").end_time.normalize()
        else:
            start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
        args.output = ROOT / "output" / "pdf" / report_filename(start, end)
    if args.month:
        output = generate_monthly_snapshot_pdf(texts_df, images_df, args.output, args.month)
    else:
        output = generate_monthly_snapshot_pdf(
            texts_df,
            images_df,
            args.output,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
