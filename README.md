# Immaculate Grid Analysis
This repository is designed to automate the analysis of [Immaculate Grid](https://www.immaculategrid.com/) performance from a group of players who submit their results via Apple Messages.

## Getting Started
To generate the complete analysis, follow these steps in order:

### 1. Database Refresh
- **Script:** `refresh_db.py`
- **Purpose:** Processes the Immaculate Grid results that are sent through text messages, storing them locally in a chat database. Compiles a dataset of all daily grid prompts by scraping and updating the existing prompt records.
- **Requirement:** You need a Mac that stores iMessages locally in the `chat.db` file.

### 2. Generate Weekly Report
- **Script:** `make_report.py`
- **Purpose:** Analyzes the latest grid results and prompts, generating a detailed weekly report for performance insights.

### 3. User Setup
- **File:** `config.json`
- **Purpose:** Ensure that all users are properly set up in the `config.json` file for accurate analysis and reporting. Note that the repo includes a file called `config_SAMPLE.json` to help you get started with yours.

## Additional Analysis Tools
## Skill Practice
- **Simulator:** An interactive game that lets users test their Immaculate Grid skills.

This sequence ensures a smooth workflow, from data refresh to analysis and reporting. Make sure to run the scripts in the specified order for accurate results.

## Troubleshooting
We have found that sometimes Apple Messages do not store properly in the `chat.db` file. To troubleshoot, clear the file, disable syncing, and then let the background process on the Mac run until the file replenishes. Then try again. Be sure to make a backup copy of the file to avoid data loss.

## Collaborators
- [Keith Gladstone](https://github.com/kgladstone)
- [Samuel Arnesen](https://github.com/samuelarnesen)
