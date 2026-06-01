#!/usr/bin/env bash
set -euo pipefail

MESSAGES_DIR="${MESSAGES_DIR:-$HOME/Library/Messages}"
CHAT_DB="$MESSAGES_DIR/chat.db"
BACKUP_ROOT="${CHAT_DB_RESET_BACKUP_DIR:-$PWD/bin/chat_snapshot/reset_backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$BACKUP_ROOT/$STAMP"

usage() {
  cat <<'EOF'
Usage:
  src/scripts/reset_live_chat_db_for_cloud_rebuild.sh
  src/scripts/reset_live_chat_db_for_cloud_rebuild.sh --yes

Deletes the live Apple Messages chat.db files and creates an empty chat.db
with touch. This is intended to trigger Messages/iCloud to rebuild the local
database.

Run from Terminal with Full Disk Access.

Environment overrides:
  MESSAGES_DIR=/path/to/Messages
  CHAT_DB_RESET_BACKUP_DIR=/path/to/backups
EOF
}

ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --yes) ASSUME_YES=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$MESSAGES_DIR" ]]; then
  echo "Messages directory not found: $MESSAGES_DIR" >&2
  exit 1
fi

cat <<EOF
This will reset the live Apple Messages database files:
  $CHAT_DB
  $CHAT_DB-wal
  $CHAT_DB-shm

A backup will be written first to:
  $BACKUP_DIR

Quit Messages before continuing. After the reset, reopen Messages and allow
iCloud sync to rebuild the local database.
EOF

if [[ "$ASSUME_YES" != "1" ]]; then
  printf "Type RESET to continue: "
  read -r confirm
  if [[ "$confirm" != "RESET" ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

mkdir -p "$BACKUP_DIR"

for path in "$CHAT_DB" "$CHAT_DB-wal" "$CHAT_DB-shm"; do
  if [[ -e "$path" ]]; then
    cp -p "$path" "$BACKUP_DIR/"
    echo "Backed up: $path"
  else
    echo "Not present, skipping backup: $path"
  fi
done

for path in "$CHAT_DB" "$CHAT_DB-wal" "$CHAT_DB-shm"; do
  if [[ -e "$path" ]]; then
    rm "$path"
    echo "Deleted: $path"
  fi
done

touch "$CHAT_DB"
echo "Created empty file: $CHAT_DB"
echo "Done. Reopen Messages and wait for iCloud to rebuild the local DB."
