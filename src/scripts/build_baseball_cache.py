from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import unidecode

try:
    from scripts.lahman_box_crawler import (
        APPEARANCES_REQUIRED_COLUMNS,
        PEOPLE_REQUIRED_COLUMNS,
        TEAMS_REQUIRED_COLUMNS,
        LahmanBoxCrawler,
    )
except ImportError:  # pragma: no cover - allows running as direct script
    from lahman_box_crawler import (
        APPEARANCES_REQUIRED_COLUMNS,
        PEOPLE_REQUIRED_COLUMNS,
        TEAMS_REQUIRED_COLUMNS,
        LahmanBoxCrawler,
    )
from utils.constants import canonicalize_franchid


ProgressCb = Callable[[str], None]


def _emit(message: str, progress_cb: ProgressCb | None = None) -> None:
    print(message, flush=True)
    if progress_cb is not None:
        progress_cb(message)


def _load_required_tables(
    cache_dir: Path,
    lahman_zip_path: Path | None,
    lahman_url: str | None,
    progress_cb: ProgressCb | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _emit("[1/4] Downloading required Lahman files via crawler (Teams, People, Appearances)...", progress_cb)

    crawler = LahmanBoxCrawler(progress_cb=progress_cb)
    candidate_urls = [lahman_url] if lahman_url else []
    local_zip_paths = [lahman_zip_path] if lahman_zip_path else []
    downloaded = crawler.download_required_csvs(
        output_dir=cache_dir,
        required_files={
            "Teams.csv": TEAMS_REQUIRED_COLUMNS,
            "People.csv": PEOPLE_REQUIRED_COLUMNS,
            "Appearances.csv": APPEARANCES_REQUIRED_COLUMNS,
        },
        candidate_urls=candidate_urls,
        local_zip_paths=local_zip_paths,
    )
    teams_path = downloaded.get("Teams.csv")
    people_path = downloaded.get("People.csv")
    appearances_path = downloaded.get("Appearances.csv")
    if teams_path is None or people_path is None or appearances_path is None:
        raise RuntimeError("Lahman crawler did not return all required files (Teams, People, Appearances).")

    teams_raw = pd.read_csv(teams_path)
    people_raw = pd.read_csv(people_path)
    appearances = pd.read_csv(appearances_path)
    _emit(
        f"[2/4] Downloaded raw rows: Teams={len(teams_raw):,}, People={len(people_raw):,}, Appearances={len(appearances):,}",
        progress_cb,
    )
    return teams_raw, people_raw, appearances


def build_cache(
    cache_dir: Path,
    max_year: int,
    lahman_zip_path: Path | None = None,
    lahman_url: str | None = None,
    progress_cb: ProgressCb | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    teams_raw, people_raw, appearances = _load_required_tables(cache_dir, lahman_zip_path, lahman_url, progress_cb)

    _emit("[3/4] Building local simulator cache files...", progress_cb)
    teams = teams_raw.loc[:, ["yearID", "teamID", "franchID"]].copy()
    teams = teams[pd.to_numeric(teams["yearID"], errors="coerce").fillna(0).astype(int) <= int(max_year)]
    teams["franchID"] = teams["franchID"].apply(canonicalize_franchid)
    teams = teams.drop_duplicates()
    teams.to_csv(cache_dir / "teams.csv", index=False)

    players = people_raw.loc[:, ["nameLast", "nameFirst", "playerID"]].copy()
    players = players.rename(columns={"nameLast": "name_last", "nameFirst": "name_first", "playerID": "key_bbref"})
    players["name_last"] = players["name_last"].apply(lambda x: unidecode.unidecode(str(x)))
    players["name_first"] = players["name_first"].apply(lambda x: unidecode.unidecode(str(x)))
    players = players.drop_duplicates()
    players.to_csv(cache_dir / "players.csv", index=False)

    appearances.to_csv(cache_dir / "appearances.csv", index=False)
    _emit(
        f"[3/4] Wrote: teams.csv={len(teams):,}, players.csv={len(players):,}, appearances.csv={len(appearances):,}",
        progress_cb,
    )

    _emit("[4/4] Writing cache metadata...", progress_cb)
    meta = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "max_year": max_year,
                "teams_rows": len(teams),
                "players_rows": len(players),
                "appearances_rows": len(appearances),
            }
        ]
    )
    meta.to_csv(cache_dir / "cache_meta.csv", index=False)

    _emit(f"Cache written to: {cache_dir}", progress_cb)
    _emit(meta.to_string(index=False), progress_cb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local baseball cache from pybaseball + Lahman crawler fallback.")
    parser.add_argument("--cache-dir", default="bin/baseball_cache", help="Output directory for cached CSVs.")
    parser.add_argument("--max-year", type=int, default=2023, help="Max year for team IDs.")
    parser.add_argument("--lahman-zip", default=None, help="Optional path to local Lahman ZIP.")
    parser.add_argument("--lahman-url", default=None, help="Optional explicit Lahman URL.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    zip_path = Path(args.lahman_zip).expanduser().resolve() if args.lahman_zip else None
    build_cache(cache_dir, max_year=args.max_year, lahman_zip_path=zip_path, lahman_url=args.lahman_url)


if __name__ == "__main__":
    main()
