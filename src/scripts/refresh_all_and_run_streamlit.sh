#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${STREAMLIT_PORT:-8501}"
ADDRESS="${STREAMLIT_ADDRESS:-127.0.0.1}"
DB_PATH="${CHAT_DB_PATH:-bin/chat_snapshot/chat_backup.db}"

echo "[1/4] Copying Messages database and attachments..."
python3 src/scripts/copy_chat_db.py

echo "[2/4] Refreshing text message responses..."
python3 src/scripts/refresh_texts_data.py --skip-copy --db-path "$DB_PATH"

echo "[3/4] Processing image attachments and rebuilding derived image CSVs..."
PYTHONPATH=src python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

from config.constants import (
    IMAGES_METADATA_CSV_PATH,
    IMAGES_METADATA_FUZZY_LOG_PATH,
    IMAGES_METADATA_PATH,
    IMAGES_PATH,
    PROMPTS_CSV_PATH,
)
from data.io.mlb_reference import correct_typos_with_fuzzy_matching
from data.io.prompts_loader import PromptsLoader
from data.transforms.data_prep import create_disaggregated_results_df
from data.vision.image_processor import ImageProcessor

db_path = Path(os.environ.get("CHAT_DB_PATH", "bin/chat_snapshot/chat_backup.db"))

ip = ImageProcessor(str(db_path), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
ip.process_images()

prompts_loader = PromptsLoader(str(PROMPTS_CSV_PATH))
prompts_loader.load().validate()
prompts_df = prompts_loader.get_data()

image_metadata_df = ip.load_image_metadata()
if image_metadata_df.empty:
    pd.DataFrame().to_csv(IMAGES_METADATA_CSV_PATH, index=False)
    pd.DataFrame().to_csv(IMAGES_METADATA_FUZZY_LOG_PATH, index=False)
else:
    disagg_df = create_disaggregated_results_df(image_metadata_df, prompts_df)
    disagg_df, typo_log = correct_typos_with_fuzzy_matching(disagg_df, "response")
    disagg_df.to_csv(IMAGES_METADATA_CSV_PATH, index=False)
    typo_log.to_csv(IMAGES_METADATA_FUZZY_LOG_PATH, index=False)

print(f"Image metadata entries: {len(image_metadata_df)}")
print(f"Wrote {IMAGES_METADATA_CSV_PATH}")
print(f"Wrote {IMAGES_METADATA_FUZZY_LOG_PATH}")
PY

echo "[4/4] Starting Streamlit at http://${ADDRESS}:${PORT} ..."
python3 -m streamlit run src/streamlit_app.py --server.port "$PORT" --server.address "$ADDRESS"
