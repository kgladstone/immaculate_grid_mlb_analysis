# Immaculate Grid Analysis

This repository is designed to automate the analysis of Immaculate Grid performance from a group of players who submit their results via Apple Messages.

## Getting Started
To generate the complete analysis, follow these steps in order:

### 1. Database Refresh
- **Script:** `refresh_db.py`
- **Purpose:** Processes the Immaculate Grid results that are sent through text messages, storing them locally in a chat database.

### 2. Prompt Update
- **Script:** `refresh_prompts.py`
- **Purpose:** Compiles a dataset of all daily grid prompts by scraping and updating the existing prompt records.

### 3. Generate Weekly Report
- **Script:** `weekly_report.py`
- **Purpose:** Analyzes the latest grid results and prompts, generating a detailed weekly report for performance insights.

## Additional Analysis Tools
- **Data Visualization:** `immaculate-grid-scraper.ipynb`  
  Use this Jupyter notebook to analyze the parsed message data and generate graphs and insights.

## Skill Practice
- **Simulator:** An interactive game that lets users test their Immaculate Grid skills.

This sequence ensures a smooth workflow, from data refresh to analysis and reporting. Make sure to run the scripts in the specified order for accurate results.
