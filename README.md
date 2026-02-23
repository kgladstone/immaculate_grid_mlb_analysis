# Immaculate Grid Analysis
This repository is designed to automate the analysis of [Immaculate Grid](https://www.immaculategrid.com/) performance from a group of players who submit their results via Apple Messages.

## Getting Started
To generate the complete analysis, follow these steps in order:

### 1. User Setup
- **File:** `config.json`
- **Purpose:** Ensure that all users are properly set up in the `config.json` file for accurate analysis and reporting. Note that the repo includes a file called `config_SAMPLE.json` to help you get started with yours. This file needs to be set up correctly for the script to work.

### 2. Database Refresh and Report Generation
- **Script:** `streamlit_app.py`
- **Purpose:** Processes the Immaculate Grid results that are sent through text messages, storing them locally in a chat database. Compiles a dataset of all daily grid prompts by scraping and updating the existing prompt records. Analyzes the latest grid results and prompts, generating a detailed weekly report for performance insights.
- **Requirement:** You need a Mac that stores iMessages locally in the `chat.db` file.

### 3. Image Processor
- **Script:** `image_processor.py`
- **Purpose:** Extracts metadata from images (e.g., screenshots of Immaculate Grid results) and compiles it into a file called `images_metadata.json`

## Additional Analysis Tools
## Skill Practice
- **Simulator:** An interactive game that lets users test their Immaculate Grid skills.

## Player History Database (Terminal Script)
- **Script:** `src/scripts/build_player_history_database.py`
- **Purpose:** Downloads and compiles a historical MLB player database with player ID, names, teams, years played, career mid-year, and (when available) positions and awards.
- **Command:**
  - `python src/scripts/build_player_history_database.py`
- **Output file:**
  - `csv/mlb_player_history.csv`

## Analytics: Player Data Subtab
- In Streamlit, open `📊 Analytics` and then the `Player Data` subtab.
- Includes:
  - Player search view: usage by submitter for any searched player.
  - Mosaic-style chart: per-user composition of top players with percentages.
  - Career mid-year histogram by submitter (requires `csv/mlb_player_history.csv`).

## Code Organization
- `src/streamlit_app.py`: App entrypoint.
- `src/app/tabs/`: Streamlit UI tabs.
- `src/app/operations/`: Report registry and shared UI data loaders.
- `src/data/`: Ingestion and parsing pipelines (messages, prompts, images).
- `src/analytics/`: Heavier cross-user analytics/report computations.
- `src/utils/`: Constants and reusable grid parsing utilities.
- `src/scripts/`: Runnable terminal scripts (cache builders, DB snapshot helpers).

This sequence ensures a smooth workflow, from data refresh to analysis and reporting. Make sure to run the scripts in the specified order for accurate results.

## Troubleshooting
We have found that sometimes Apple Messages do not store properly in the `chat.db` file. To troubleshoot, clear the file, disable syncing, and then let the background process on the Mac run until the file replenishes. Then try again. Be sure to make a backup copy of the file to avoid data loss.

## Collaborators
- [Keith Gladstone](https://github.com/kgladstone)
- [Samuel Arnesen](https://github.com/samuelarnesen)
