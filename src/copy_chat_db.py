from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from utils.constants import APPLE_TEXTS_DB_PATH


def copy_chat_db(src: Path, dest: Path) -> Path:
    """
    Simple copy of chat.db (no WAL). Useful when WAL isn't present or for quick copies.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source DB not found at {src}")
    if not src.is_file():
        raise ValueError(f"Source path is not a file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.resolve()


def copy_chat_db_with_wal(
    src_dir: Path | None = None,
    dest_dir: Path = Path(__file__).resolve().parent.parent / "chat_snapshot",
) -> Path:
    """
    Copy chat.db, chat.db-wal, and chat.db-shm to a snapshot folder.
    If WAL exists, this captures the latest writes; if not, chat.db alone is copied.
    """
    if src_dir is None:
        src_dir = Path(APPLE_TEXTS_DB_PATH).expanduser().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    db_path = src_dir / "chat.db"
    wal_path = src_dir / "chat.db-wal"
    shm_path = src_dir / "chat.db-shm"

    if not db_path.exists():
        raise FileNotFoundError(f"Source DB not found at {db_path}")

    shutil.copy2(db_path, dest_dir / "chat.db")
    if wal_path.exists():
        shutil.copy2(wal_path, dest_dir / "chat.db-wal")
    if shm_path.exists():
        shutil.copy2(shm_path, dest_dir / "chat.db-shm")

    return (dest_dir / "chat.db").resolve()


def sqlite_backup(src: Path, dest: Path) -> None:
    """
    Use SQLite backup API to create a consistent snapshot of chat.db.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_con = sqlite3.connect(src)
    dst_con = sqlite3.connect(dest)
    src_con.backup(dst_con)
    dst_con.close()
    src_con.close()


def validate_db(path: Path) -> tuple:
    con = sqlite3.connect(path)
    journal = con.execute("PRAGMA journal_mode;").fetchone()
    integrity = con.execute("PRAGMA integrity_check;").fetchone()
    con.close()
    return journal, integrity


def main():
    src_dir = Path(APPLE_TEXTS_DB_PATH).expanduser().parent
    dest_dir = Path(__file__).resolve().parent.parent / "chat_snapshot"

    # Step 1: copy raw files (chat.db + WAL/SHM)
    dest_path = copy_chat_db_with_wal(src_dir=src_dir, dest_dir=dest_dir)

    # Step 2: create a stable backup from the copied chat.db to chat_snapshot/chat_backup.db
    backup_path = dest_dir / "chat_backup.db"
    sqlite_backup(dest_path, backup_path)

    # Step 3: validate both files
    summaries = []
    for label, path in [("snapshot chat.db", dest_path), ("backup chat_backup.db", backup_path)]:
        try:
            journal, integrity = validate_db(path)
            summaries.append(f"{label}: journal_mode={journal}, integrity_check={integrity}")
        except Exception as exc:
            summaries.append(f"{label}: validation failed: {exc}")

    print(f"Snapshot folder: {dest_dir}")
    print("\n".join(summaries))
    print("Use chat_snapshot/chat_backup.db for Streamlit 'Messages DB path' (stable snapshot).")


if __name__ == "__main__":
    main()
