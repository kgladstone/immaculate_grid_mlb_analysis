from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

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
from config.constants import canonicalize_franchid


ProgressCb = Callable[[str], None]
PLAYER_FIRST_TEAM_CACHE_FILENAME = "player_first_team.csv"
TEAM_YEAR_OLDEST_CACHE_FILENAME = "team_year_oldest_players.csv"


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
    _emit("[1/5] Downloading required Lahman files via crawler (Teams, People, Appearances)...", progress_cb)

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
        f"[2/5] Downloaded raw rows: Teams={len(teams_raw):,}, People={len(people_raw):,}, Appearances={len(appearances):,}",
        progress_cb,
    )
    return teams_raw, people_raw, appearances


def _to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _build_oldest_player_tables(
    teams_raw: pd.DataFrame,
    people_raw: pd.DataFrame,
    appearances: pd.DataFrame,
    max_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = teams_raw.loc[:, ["yearID", "teamID", "franchID"]].copy()
    teams["yearID"] = _to_int_series(teams["yearID"])
    teams = teams[teams["yearID"].notna()]
    teams = teams[teams["yearID"].astype(int) <= int(max_year)].copy()
    teams["yearID"] = teams["yearID"].astype(int)
    teams["franchID"] = teams["franchID"].apply(canonicalize_franchid)
    teams = teams.drop_duplicates()

    people_cols = [c for c in ["playerID", "nameFirst", "nameLast", "birthYear", "birthMonth", "birthDay"] if c in people_raw.columns]
    people = people_raw[people_cols].drop_duplicates(subset=["playerID"]).copy()
    for col in ("birthYear", "birthMonth", "birthDay"):
        if col in people.columns:
            people[col] = _to_int_series(people[col])
        else:
            people[col] = pd.Series(pd.array([pd.NA] * len(people), dtype="Int64"))

    base = appearances.copy()
    base["yearID"] = _to_int_series(base["yearID"])
    base = base[base["yearID"].notna()].copy()
    base["yearID"] = base["yearID"].astype(int)
    base = base[base["yearID"] <= int(max_year)].copy()
    base["G_all_num"] = pd.to_numeric(base.get("G_all", 0), errors="coerce").fillna(0.0)

    merged = base.merge(teams, on=["yearID", "teamID"], how="left")
    merged = merged.merge(people, on="playerID", how="left")
    merged = merged[merged["franchID"].notna()].copy()

    # Earliest year/team in appearances data. Ties in first year are broken by more games, then teamID.
    first_team = merged.sort_values(
        ["playerID", "yearID", "G_all_num", "teamID"],
        ascending=[True, True, False, True],
    ).drop_duplicates(subset=["playerID"], keep="first").copy()
    first_team = first_team.rename(
        columns={
            "yearID": "first_yearID",
            "teamID": "first_teamID",
            "franchID": "first_franchID",
        }
    )
    first_team = first_team[
        [
            "playerID",
            "nameFirst",
            "nameLast",
            "birthYear",
            "birthMonth",
            "birthDay",
            "first_yearID",
            "first_teamID",
            "first_franchID",
        ]
    ].copy()

    oldest_pool = merged[merged["birthYear"].notna()].copy()
    if oldest_pool.empty:
        return first_team, pd.DataFrame(
            columns=[
                "yearID",
                "teamID",
                "franchID",
                "playerID",
                "nameFirst",
                "nameLast",
                "birthYear",
                "birthMonth",
                "birthDay",
                "age_on_july_1",
                "first_yearID",
                "first_teamID",
                "first_franchID",
            ]
        )

    oldest_pool["birthMonthSort"] = oldest_pool["birthMonth"].fillna(12).astype(int)
    oldest_pool["birthDaySort"] = oldest_pool["birthDay"].fillna(31).astype(int)
    oldest_pool["age_on_july_1"] = (
        oldest_pool["yearID"]
        - oldest_pool["birthYear"].astype(int)
        - (
            (oldest_pool["birthMonthSort"] > 7)
            | ((oldest_pool["birthMonthSort"] == 7) & (oldest_pool["birthDaySort"] > 1))
        ).astype(int)
    )

    oldest = oldest_pool.sort_values(
        ["yearID", "teamID", "birthYear", "birthMonthSort", "birthDaySort", "G_all_num", "playerID"],
        ascending=[True, True, True, True, True, False, True],
    ).drop_duplicates(subset=["yearID", "teamID"], keep="first")

    oldest = oldest.merge(
        first_team[["playerID", "first_yearID", "first_teamID", "first_franchID"]],
        on="playerID",
        how="left",
    )
    oldest = oldest[
        [
            "yearID",
            "teamID",
            "franchID",
            "playerID",
            "nameFirst",
            "nameLast",
            "birthYear",
            "birthMonth",
            "birthDay",
            "age_on_july_1",
            "first_yearID",
            "first_teamID",
            "first_franchID",
        ]
    ].copy()
    return first_team, oldest


def build_cache(
    cache_dir: Path,
    max_year: int,
    lahman_zip_path: Path | None = None,
    lahman_url: str | None = None,
    progress_cb: ProgressCb | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    teams_raw, people_raw, appearances = _load_required_tables(cache_dir, lahman_zip_path, lahman_url, progress_cb)

    _emit("[3/5] Building local simulator cache files...", progress_cb)
    teams = teams_raw.loc[:, ["yearID", "teamID", "franchID"]].copy()
    teams = teams[pd.to_numeric(teams["yearID"], errors="coerce").fillna(0).astype(int) <= int(max_year)]
    teams["franchID"] = teams["franchID"].apply(canonicalize_franchid)
    teams = teams.drop_duplicates()
    teams.to_csv(cache_dir / "teams.csv", index=False)

    people = people_raw.drop_duplicates()
    people.to_csv(cache_dir / "People.csv", index=False)

    appearances.to_csv(cache_dir / "appearances.csv", index=False)
    _emit(
        f"[3/5] Wrote: teams.csv={len(teams):,}, People.csv={len(people):,}, appearances.csv={len(appearances):,}",
        progress_cb,
    )

    _emit("[4/5] Building oldest-player relational cache tables...", progress_cb)
    player_first_team, team_year_oldest = _build_oldest_player_tables(teams_raw, people_raw, appearances, max_year=max_year)
    player_first_team.to_csv(cache_dir / PLAYER_FIRST_TEAM_CACHE_FILENAME, index=False)
    team_year_oldest.to_csv(cache_dir / TEAM_YEAR_OLDEST_CACHE_FILENAME, index=False)
    _emit(
        f"[4/5] Wrote: {PLAYER_FIRST_TEAM_CACHE_FILENAME}={len(player_first_team):,}, "
        f"{TEAM_YEAR_OLDEST_CACHE_FILENAME}={len(team_year_oldest):,}",
        progress_cb,
    )

    _emit("[5/5] Writing cache metadata...", progress_cb)
    meta = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "max_year": max_year,
                "teams_rows": len(teams),
                "people_rows": len(people),
                "appearances_rows": len(appearances),
                "player_first_team_rows": len(player_first_team),
                "team_year_oldest_rows": len(team_year_oldest),
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
