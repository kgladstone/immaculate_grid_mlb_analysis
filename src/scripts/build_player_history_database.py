from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import unidecode


def _pipe_join(values) -> str:
    unique = sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip()})
    return "|".join(unique)


def _safe_lahman_table(table_name: str) -> pd.DataFrame:
    import pybaseball.lahman as lahman

    loader = getattr(lahman, table_name, None)
    if loader is None:
        return pd.DataFrame()
    try:
        df = loader()
    except Exception:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def build_player_history_database(output_path: Path) -> pd.DataFrame:
    import pybaseball as pb

    output_path.parent.mkdir(parents=True, exist_ok=True)

    players = pb.chadwick_register().copy()
    players["name_first"] = players["name_first"].fillna("").astype(str).map(unidecode.unidecode)
    players["name_last"] = players["name_last"].fillna("").astype(str).map(unidecode.unidecode)
    players["player_id"] = players["key_bbref"].fillna("").astype(str)

    keep_cols = [
        "player_id",
        "name_first",
        "name_last",
        "key_retro",
        "key_mlbam",
        "mlb_played_first",
        "mlb_played_last",
    ]
    players = players[[c for c in keep_cols if c in players.columns]].drop_duplicates("player_id")
    players["full_name"] = (players["name_first"] + " " + players["name_last"]).str.strip()

    appearances = _safe_lahman_table("appearances")
    fielding = _safe_lahman_table("fielding")
    awards = _safe_lahman_table("awards_players")

    if appearances.empty:
        raise RuntimeError("Could not load Lahman appearances table from pybaseball.")

    appearances = appearances.loc[:, [c for c in ["playerID", "yearID", "teamID"] if c in appearances.columns]].copy()
    appearances["playerID"] = appearances["playerID"].astype(str)
    appearances["yearID"] = pd.to_numeric(appearances["yearID"], errors="coerce")
    appearances = appearances.dropna(subset=["playerID", "yearID"])
    appearances["yearID"] = appearances["yearID"].astype(int)

    base = (
        appearances.groupby("playerID", as_index=False)
        .agg(
            teams=("teamID", _pipe_join),
            years_played=("yearID", _pipe_join),
            first_year=("yearID", "min"),
            last_year=("yearID", "max"),
            career_mid_year=("yearID", "median"),
        )
        .rename(columns={"playerID": "player_id"})
    )
    base["career_mid_year"] = base["career_mid_year"].round().astype("Int64")

    if not fielding.empty and {"playerID", "POS"}.issubset(fielding.columns):
        positions = (
            fielding.groupby("playerID", as_index=False)
            .agg(positions=("POS", _pipe_join))
            .rename(columns={"playerID": "player_id"})
        )
    else:
        positions = pd.DataFrame(columns=["player_id", "positions"])

    if not awards.empty and {"playerID", "awardID"}.issubset(awards.columns):
        awards_df = (
            awards.groupby("playerID", as_index=False)
            .agg(awards=("awardID", _pipe_join))
            .rename(columns={"playerID": "player_id"})
        )
    else:
        awards_df = pd.DataFrame(columns=["player_id", "awards"])

    out = (
        base.merge(players, on="player_id", how="left")
        .merge(positions, on="player_id", how="left")
        .merge(awards_df, on="player_id", how="left")
    )
    out["positions"] = out["positions"].fillna("")
    out["awards"] = out["awards"].fillna("")
    out["full_name"] = out["full_name"].fillna("").astype(str).str.strip()

    out = out[
        [
            "player_id",
            "name_first",
            "name_last",
            "full_name",
            "teams",
            "years_played",
            "first_year",
            "last_year",
            "career_mid_year",
            "positions",
            "awards",
            "key_retro",
            "key_mlbam",
            "mlb_played_first",
            "mlb_played_last",
        ]
    ].sort_values(["name_last", "name_first", "player_id"])

    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + compile a historical MLB player database for local analytics use."
    )
    parser.add_argument(
        "--output",
        default="csv/mlb_player_history.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    out = build_player_history_database(output_path)
    print(f"Wrote {len(out):,} rows -> {output_path}")
    print("Columns:")
    print(", ".join(out.columns))


if __name__ == "__main__":
    main()
