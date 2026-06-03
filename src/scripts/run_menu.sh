#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

run_cmd() {
  local label="$1"
  shift
  echo
  echo "==> $label"
  echo "+ $*"
  echo
  "$@"
}

run_shell_cmd() {
  local label="$1"
  local cmd="$2"
  echo
  echo "==> $label"
  echo "+ $cmd"
  echo
  bash -lc "$cmd"
}

while true; do
  cat <<'MENU'

Immaculate Grid MLB Analysis - Run Menu

  1) Start Streamlit app
  2) Refresh all data and start Streamlit
  3) Verify report bank completeness
  4) Build baseball cache
  5) Build career WAR cache
  6) Reprocess images from metadata
  7) Copy local Messages DB snapshot
  8) Watch live chat DB size
  9) Reset live chat DB for cloud rebuild
 10) Run tests
 11) Compile Python files
 12) Git status
 13) Open Rule 5 ban CSV preview
  q) Quit

MENU

  read -r -p "Choose an option: " choice

  case "$choice" in
    1)
      run_cmd "Start Streamlit app" python3 -m streamlit run src/streamlit_app.py --server.port 8501
      ;;
    2)
      run_cmd "Refresh all data and start Streamlit" bash src/scripts/refresh_all_and_run_streamlit.sh
      ;;
    3)
      run_cmd "Verify report bank completeness" python3 src/scripts/verify_report_bank_completeness.py
      ;;
    4)
      run_cmd "Build baseball cache" python3 src/scripts/build_baseball_cache.py
      ;;
    5)
      run_cmd "Build career WAR cache" python3 src/scripts/build_career_war_cache.py --auto-download
      ;;
    6)
      run_cmd "Reprocess images from metadata" python3 src/scripts/reprocess_images_from_metadata.py --rebuild-derived
      ;;
    7)
      run_cmd "Copy local Messages DB snapshot" python3 src/scripts/copy_chat_db.py
      ;;
    8)
      run_cmd "Watch live chat DB size" bash src/scripts/watch_live_chat_db_size.sh
      ;;
    9)
      run_cmd "Reset live chat DB for cloud rebuild" bash src/scripts/reset_live_chat_db_for_cloud_rebuild.sh
      ;;
    10)
      run_cmd "Run tests" pytest
      ;;
    11)
      run_shell_cmd "Compile Python files" "python3 -m py_compile \$(find src tests -name '*.py' -print)"
      ;;
    12)
      run_cmd "Git status" git status --short
      ;;
    13)
      run_cmd "Rule 5 ban CSV preview" python3 -c "import pandas as pd; df=pd.read_csv('bin/csv/rule5_full_bans.csv', dtype=str); print(df.to_string(index=False))"
      ;;
    q|Q)
      exit 0
      ;;
    *)
      echo "Unknown option: $choice"
      ;;
  esac
done
