from __future__ import annotations

import shutil
import sqlite3
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.constants import APPLE_TEXTS_DB_PATH

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".gif", ".heif"}


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
    dest_dir: Path = Path(__file__).resolve().parents[2] / "chat_snapshot",
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


def copy_image_attachments(src_root: Path, dest_root: Path, progress_cb=None) -> dict:
    """
    Copy image attachments into a workspace-friendly cache. Returns counts and bytes.
    """
    if not src_root.exists():
        raise FileNotFoundError(f"Attachments folder not found at {src_root}")
    if not os.access(src_root, os.R_OK):
        raise PermissionError(
            f"Attachments folder not readable at {src_root}. "
            "Grant Full Disk Access to your terminal/IDE or copy from an FDA-enabled shell."
        )

    candidates = []
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(src_root)
        dest_path = dest_root / rel
        if dest_path.exists():
            continue
        size = path.stat().st_size
        candidates.append((path, dest_path, size))

    total_bytes = sum(size for _, _, size in candidates)
    if total_bytes == 0:
        print("Attachments already cached; nothing to copy.")
        return {"copied": 0, "skipped": 0, "bytes": 0}

    mb_total = total_bytes / (1024 * 1024)
    print(f"Copying {len(candidates)} attachment(s) (~{mb_total:.1f} MB) to {dest_root} ...")

    copied = 0
    copied_bytes = 0
    next_threshold = 0.1

    for src_path, dest_path, size in candidates:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        copied += 1
        copied_bytes += size
        if total_bytes > 0:
            pct = copied_bytes / total_bytes
            if pct >= next_threshold:
                mb_done = copied_bytes / (1024 * 1024)
                print(f"  {pct*100:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                if progress_cb:
                    progress_cb(pct, f"Copying attachments ({pct*100:.1f}%)")
                next_threshold += 0.1

    skipped = 0  # skipped counted implicitly by existing files above
    return {"copied": copied, "skipped": skipped, "bytes": total_bytes}


def main(progress_cb=None, src_dir: Path | None = None):
    if src_dir is None:
        src_dir = Path(APPLE_TEXTS_DB_PATH).expanduser().parent
    dest_dir = Path(__file__).resolve().parents[2] / "chat_snapshot"

    # Step 1: copy raw files (chat.db + WAL/SHM)
    if progress_cb:
        progress_cb(0.05, "Copying chat.db and WAL files")
    dest_path = copy_chat_db_with_wal(src_dir=src_dir, dest_dir=dest_dir)

    # Step 2: create a stable backup from the copied chat.db to chat_snapshot/chat_backup.db
    if progress_cb:
        progress_cb(0.35, "Creating backup snapshot")
    backup_path = dest_dir / "chat_backup.db"
    sqlite_backup(dest_path, backup_path)

    # Step 3: validate both files
    if progress_cb:
        progress_cb(0.55, "Validating snapshot integrity")
    summaries = []
    for label, path in [("snapshot chat.db", dest_path), ("backup chat_backup.db", backup_path)]:
        try:
            journal, integrity = validate_db(path)
            summaries.append(f"{label}: journal_mode={journal}, integrity_check={integrity}")
        except Exception as exc:
            summaries.append(f"{label}: validation failed: {exc}")

    # Step 4: copy image attachments into chat_snapshot/Attachments
    attachments_src = src_dir / "Attachments"
    attachments_dest = dest_dir / "Attachments"
    try:
        if progress_cb:
            progress_cb(0.7, "Copying attachments")
        attachment_counts = copy_image_attachments(attachments_src, attachments_dest, progress_cb=progress_cb)
        summaries.append(
            f"Attachments: copied {attachment_counts['copied']}, skipped {attachment_counts['skipped']} (dest: {attachments_dest})"
        )
    except Exception as exc:
        summaries.append(f"Attachments: copy failed: {exc}")

    if progress_cb:
        progress_cb(1.0, "Cache reset complete")

    print(f"Snapshot folder: {dest_dir}")
    print("\n".join(summaries))
    print("Use chat_snapshot/chat_backup.db for Streamlit 'Messages DB path' (stable snapshot).")


if __name__ == "__main__":
    main()
