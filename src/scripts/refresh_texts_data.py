from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.constants import APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH
from data.io.messages_loader import MessagesLoader
from scripts.copy_chat_db import main as copy_chat_snapshot


def _resolve_default_db_path() -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh bin/csv/text_message_responses.csv from Apple Messages."
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip snapshot copy step and read directly from --db-path/default DB path.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional DB path override (chat.db or chat_backup.db).",
    )
    args = parser.parse_args()

    live_db_path = Path(APPLE_TEXTS_DB_PATH).expanduser()

    if not args.skip_copy:
        print("[1/2] Creating chat snapshot (db + wal + attachments)...")
        try:
            copy_chat_snapshot()
        except PermissionError as exc:
            fallback_db = _resolve_default_db_path()
            if fallback_db.exists() and fallback_db != live_db_path:
                print(
                    "[warn] Snapshot copy blocked by macOS permissions; "
                    f"continuing with existing local snapshot: {fallback_db}"
                )
                print(f"[warn] Original error: {exc}")
            else:
                raise PermissionError(
                    "Cannot read live Messages database due to macOS permissions.\n"
                    "Grant Full Disk Access to the terminal/app running Python "
                    "(e.g., Terminal, iTerm, VSCode, Cursor) and retry,\n"
                    "or run with --skip-copy --db-path <existing_snapshot_db>."
                ) from exc
    else:
        print("[1/2] Skipping snapshot copy (--skip-copy set).")

    db_path = Path(args.db_path).expanduser() if args.db_path else _resolve_default_db_path()
    print(f"[2/2] Refreshing texts cache from: {db_path}")

    loader = MessagesLoader(str(db_path), str(MESSAGES_CSV_PATH))
    loader.load().validate()
    print(f"Done. Wrote/updated: {MESSAGES_CSV_PATH}")


if __name__ == "__main__":
    main()
