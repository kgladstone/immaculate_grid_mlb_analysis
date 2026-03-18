from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from config.constants import (
    APPLE_TEXTS_DB_PATH,
    IMAGES_METADATA_CSV_PATH,
    IMAGES_METADATA_FUZZY_LOG_PATH,
    IMAGES_METADATA_PATH,
    IMAGES_PATH,
    MESSAGES_CSV_PATH,
    PROMPTS_CSV_PATH,
)
from data.io.mlb_reference import correct_typos_with_fuzzy_matching
from data.io.prompts_loader import PromptsLoader
from data.transforms.data_prep import create_disaggregated_results_df
from data.vision.image_processor import ImageProcessor


def _norm_grid(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _default_messages_db_path() -> Path:
    cwd = Path.cwd()
    for candidate in [
        cwd / "bin" / "chat_snapshot" / "chat_backup.db",
        cwd / "bin" / "chat_snapshot" / "chat.db",
        cwd / "chat.db",
        Path(APPLE_TEXTS_DB_PATH).expanduser(),
    ]:
        if candidate.exists():
            return candidate
    return Path(APPLE_TEXTS_DB_PATH).expanduser()


def _load_metadata_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON in {path}")
    rows = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        row["submitter"] = str(row.get("submitter") or "").strip()
        row["grid_number_norm"] = _norm_grid(row.get("grid_number"))
        row["image_filename"] = str(row.get("image_filename") or "").strip()
        row["date"] = str(row.get("date") or row.get("image_date") or "").strip()
        rows.append(row)
    return rows


def _is_bad_responses(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return False
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def _count_bad_responses(path: Path) -> tuple[int, int]:
    rows = _load_metadata_rows(path)
    bad = sum(1 for r in rows if _is_bad_responses(r.get("responses")))
    return bad, len(rows)


def _load_messages_lookup(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, usecols=["name", "grid_number", "matrix"], dtype=str)
    except Exception:
        return {}
    df["name"] = df["name"].astype(str).str.strip()
    df["grid_number_norm"] = df["grid_number"].map(_norm_grid)
    lookup: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        key = (row["name"], row["grid_number_norm"])
        if key not in lookup:
            lookup[key] = str(row.get("matrix") or "")
    return lookup


def _resolve_image_path(row: dict) -> Path | None:
    fn = str(row.get("image_filename") or "").strip()
    if not fn:
        return None
    p = Path(IMAGES_PATH) / fn
    return p if p.exists() else None


def _rebuild_derivatives() -> tuple[bool, str]:
    try:
        prompts_loader = PromptsLoader(str(PROMPTS_CSV_PATH))
        prompts_loader.load().validate()
        prompts_df = prompts_loader.get_data()

        ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
        image_metadata_df = ip.load_image_metadata()
        if image_metadata_df.empty:
            pd.DataFrame().to_csv(Path(IMAGES_METADATA_CSV_PATH), index=False)
            pd.DataFrame().to_csv(Path(IMAGES_METADATA_FUZZY_LOG_PATH), index=False)
            return True, "Rebuilt derivatives (empty metadata)."

        disagg_df = create_disaggregated_results_df(image_metadata_df, prompts_df)
        disagg_df, typo_log = correct_typos_with_fuzzy_matching(disagg_df, "response")
        disagg_df.to_csv(Path(IMAGES_METADATA_CSV_PATH), index=False)
        typo_log.to_csv(Path(IMAGES_METADATA_FUZZY_LOG_PATH), index=False)
        return True, f"Rebuilt derivatives: rows={len(disagg_df)}, fuzzy_changes={len(typo_log)}."
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reprocess images listed in images_metadata.json and replace rows on success."
    )
    parser.add_argument("--submitter", help="Optional submitter filter")
    parser.add_argument("--grid", help="Optional grid number filter")
    parser.add_argument("--filename", help="Optional image filename filter")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show planned rows only")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip matrix validation when parsing each image.",
    )
    parser.add_argument(
        "--rebuild-derivatives",
        action="store_true",
        help="Rebuild images_metadata.csv and fuzzy log after reprocessing.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a timestamped backup copy of images_metadata.json before processing.",
    )
    args = parser.parse_args()

    meta_path = Path(IMAGES_METADATA_PATH)
    rows = _load_metadata_rows(meta_path)

    if args.submitter:
        rows = [r for r in rows if r.get("submitter", "").lower() == args.submitter.strip().lower()]
    if args.grid:
        grid_norm = _norm_grid(args.grid)
        rows = [r for r in rows if r.get("grid_number_norm") == grid_norm]
    if args.filename:
        rows = [r for r in rows if r.get("image_filename") == args.filename.strip()]

    rows = [r for r in rows if r.get("submitter") and r.get("grid_number_norm") and r.get("image_filename")]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(f"Selected metadata rows: {len(rows)}")
    if not rows:
        return 0

    preview = pd.DataFrame(
        [
            {
                "submitter": r.get("submitter"),
                "grid_number": r.get("grid_number_norm"),
                "date": r.get("date"),
                "image_filename": r.get("image_filename"),
            }
            for r in rows
        ]
    )
    print(preview.to_string(index=False))

    if args.dry_run:
        return 0

    backups_dir = meta_path.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    guard_backup_path = backups_dir / f"{meta_path.stem}.guard_backup.{timestamp}.json"
    shutil.copy2(meta_path, guard_backup_path)
    print(f"Guard backup written: {guard_backup_path}")

    if args.backup:
        backup_path = backups_dir / f"{meta_path.stem}.backup.{timestamp}.json"
        shutil.copy2(meta_path, backup_path)
        print(f"Backup written: {backup_path}")

    bad_before, total_before = _count_bad_responses(meta_path)
    print(f"Response quality before run: bad={bad_before} / total={total_before}")

    messages_lookup = _load_messages_lookup(Path(MESSAGES_CSV_PATH))
    ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))

    success = 0
    failed = 0
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        submitter = row["submitter"]
        grid_norm = row["grid_number_norm"]
        date = row["date"]
        filename = row["image_filename"]

        src_path = _resolve_image_path(row)
        if src_path is None:
            print(f"[{idx}/{len(rows)}] SKIP missing file: {filename}")
            skipped += 1
            continue

        matrix = messages_lookup.get((submitter, grid_norm), "")
        if not matrix:
            print(f"[{idx}/{len(rows)}] SKIP missing matrix for {submitter} grid {grid_norm}")
            skipped += 1
            continue

        try:
            grid_int = int(grid_norm)
        except Exception:
            print(f"[{idx}/{len(rows)}] SKIP invalid grid value: {grid_norm}")
            skipped += 1
            continue

        temp_path = None
        try:
            # process_image_with_dynamic_grid copies source image to IMAGES_PATH;
            # use a temp source path to avoid SameFileError when source is already in IMAGES_PATH.
            with tempfile.NamedTemporaryFile(
                prefix="reparse_",
                suffix=src_path.suffix or ".jpg",
                delete=False,
                dir=str(Path(IMAGES_PATH)),
            ) as tmp:
                temp_path = Path(tmp.name)
            shutil.copy2(src_path, temp_path)

            source_hash = ip._file_sha256(src_path)
            msg = ip.process_image_with_dynamic_grid(
                str(temp_path),
                submitter,
                date,
                grid_int,
                matrix,
                skip_validation=bool(args.skip_validation),
                source_hash=source_hash,
            )
            if str(msg).startswith("Success"):
                print(f"[{idx}/{len(rows)}] OK {submitter} grid {grid_norm} ({filename})")
                success += 1
            else:
                print(f"[{idx}/{len(rows)}] FAIL {submitter} grid {grid_norm} ({filename}): {msg}")
                failed += 1
        except Exception as exc:
            print(f"[{idx}/{len(rows)}] ERROR {submitter} grid {grid_norm} ({filename}): {exc}")
            failed += 1
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    print(f"Done. success={success} failed={failed} skipped={skipped}")

    bad_after, total_after = _count_bad_responses(meta_path)
    print(f"Response quality after run: bad={bad_after} / total={total_after}")
    if bad_after > bad_before:
        shutil.copy2(guard_backup_path, meta_path)
        print(
            "Guard triggered: bad response count increased. "
            f"Restored metadata from {guard_backup_path}."
        )
        return 3

    if args.rebuild_derivatives:
        ok, msg = _rebuild_derivatives()
        if ok:
            print(msg)
        else:
            print(f"Derivative rebuild failed: {msg}")
            return 2
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
