#!/usr/bin/env bash
set -euo pipefail

MESSAGES_DIR="${MESSAGES_DIR:-$HOME/Library/Messages}"
CHAT_DB="${CHAT_DB_PATH:-$MESSAGES_DIR/chat.db}"
INTERVAL_SECONDS="${CHAT_DB_WATCH_INTERVAL:-10}"

while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -e "$CHAT_DB" ]]; then
    size_bytes="$(stat -f '%z' "$CHAT_DB")"
    size_mb="$(awk "BEGIN { printf \"%.2f\", $size_bytes / 1024 / 1024 }")"
    echo "$timestamp | $CHAT_DB | $size_bytes bytes | ${size_mb} MB"
  else
    echo "$timestamp | $CHAT_DB | missing"
  fi
  sleep "$INTERVAL_SECONDS"
done
