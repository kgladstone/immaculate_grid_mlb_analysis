# Immaculate Grid Analysis
This repository is designed to automate the analysis of [Immaculate Grid](https://www.immaculategrid.com/) performance from a group of players who submit their results via Apple Messages.

## Getting Started
To generate the complete analysis, follow these steps in order:

## Cache Setup
- **SABR Lahman (via the Cache Builder or `python src/scripts/build_baseball_cache.py`):** downloads `Teams.csv`, `People.csv`, and `Appearances.csv`, writes `bin/baseball_cache/teams.csv`, `bin/baseball_cache/People.csv`, and `bin/baseball_cache/appearances.csv`, and guarantees each file includes the raw Lahman box-score inputs needed by the simulator and analytics modules.
- **Player dataset:** `People.csv` is the canonical local player table used for name and `playerID` lookups; `appearances.csv` is the raw archive of team-year-player rows used to map median years.
- **Baseball-Reference WAR exports:** the Cache Builder now downloads the latest WAR ZIP from https://www.baseball-reference.com/data/ (first alphabetically, Z→A), extracts the WAR files, and writes `bin/baseball_cache/war.csv`; the `src/scripts/build_career_war_cache.py` helper exposes `--auto-download` for the same behavior or still accepts `--bat-file`/`--pitch-file` when you already have the exports on disk.
- **Images metadata:** `bin/csv/images_metadata.csv` comes from the refresh tab (or the analytics cache) and feeds the UI with parsed player responses.

### 1. User Setup
- **File:** `bin/json/config.json`
- **Template:** `bin/json/config_SAMPLE.json`
- **Purpose:** Ensure all users are set up in `config.json` for accurate analysis and reporting.

### 2. Database Refresh and Report Generation
- **Script:** `src/streamlit_app.py`
- **Purpose:** Processes the Immaculate Grid results that are sent through text messages, storing them locally in a chat database. Compiles a dataset of all daily grid prompts by scraping and updating the existing prompt records. Analyzes the latest grid results and prompts, generating a detailed weekly report for performance insights.
- **Requirement:** You need a Mac that stores iMessages locally in the `chat.db` file.

### 3. Image Processor
- **Module:** `src/data/vision/image_processor.py`
- **Purpose:** Extracts metadata from result screenshots and writes `bin/json/images_metadata.json`.

## Mini Games
Open `🎮 Mini Games` in Streamlit.

- **Intersection Checker:** Fuzzy name match + yes/no check for a team-team intersection.
- **Study Guide:** Rank players by distinct franchises with optional league/division and franchID filters.
- **Random Immaculate Grid (3x3):** Generate solvable team-team grids, fill answers, and check puzzle.
- **Random Immaculate Cube (3x3x3):** Generate solvable 3D team puzzles, fill answers, and check puzzle.
- **Reverse Immaculate Grid:** Pre-filled 3x3 player grid; guess hidden row/column team axes.
  - Supports submitter filtering and defaults to non-restricted `GRID_PLAYERS`.

## Analytics: Player Data Subtab
- In Streamlit, open `📊 Analytics` and then the `Player Data` subtab.
- Includes:
  - Player search view: usage by submitter for any searched player.
  - Mosaic-style chart: per-user composition of top players with percentages.
- Median-year histograms (per-user and submitter-unique) depend on the local Lahman cache in `bin/baseball_cache` (build with `python src/scripts/build_baseball_cache.py`) and the derived response log at `bin/csv/images_metadata.csv`.
- Career WAR usage distribution is available in the same tab and maps every usage to Baseball-Reference WAR totals that are cached underneath `bin/baseball_cache/war.csv`. The analytics cache rebuild now downloads the latest WAR ZIP (first alphabetical link, Z→A) by default if a local cache is missing.
- **Script:** `src/scripts/build_career_war_cache.py`
- **Purpose:** Download `war_daily_bat.txt` and/or `war_daily_pitch.txt` from Baseball-Reference's `/data/` directory (or automatically pull the newest ZIP via `--auto-download`), aggregate career WAR per `bbrefID`, and persist the lookup so the analytics tab can map images to WAR.
- **Command:** `python src/scripts/build_career_war_cache.py --auto-download` (or provide `--bat-file`/`--pitch-file` when you already have the exports).
- **Output file:** `bin/baseball_cache/war.csv`

## Code Organization
- `src/streamlit_app.py`: App entrypoint.
- `src/app/tabs/`: Streamlit UI tabs.
- `src/app/services/`: Report registry and shared UI data loaders.
- `src/app/tabs/mini_games/`: Mini-game subtab renderers.
- `src/data/io/`: I/O loaders and external data mapping helpers.
- `src/data/transforms/`: Data prep/transform logic.
- `src/data/vision/`: Image parsing and OCR pipeline.
- `src/analytics/`: Heavier cross-user analytics/report computations.
- `src/config/`: Python config module (`constants.py`).
- `src/utils/`: Shared utility helpers.
- `src/scripts/`: Runnable terminal scripts (cache builders, DB snapshot helpers).
- `bin/json/`: JSON runtime data (`config.json`, `images_metadata.json`, `images_parser.json`, etc.).
- `bin/csv/`: CSV runtime data (`text_message_responses.csv`, `prompts.csv`, derived exports).
- `bin/chat_snapshot/`: Local copied iMessage database and attachments snapshot.
- `bin/logos/`: Static logo assets.

This sequence ensures a smooth workflow, from data refresh to analysis and reporting. Make sure to run the scripts in the specified order for accurate results.

## Troubleshooting
We have found that sometimes Apple Messages do not store properly in the `chat.db` file. To troubleshoot, clear the file, disable syncing, and then let the background process on the Mac run until the file replenishes. Then try again. Be sure to make a backup copy of the file to avoid data loss.

## Collaborators
- [Keith Gladstone](https://github.com/kgladstone)
- [Samuel Arnesen](https://github.com/samuelarnesen)
