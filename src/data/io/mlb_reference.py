import pandas as pd
import re
from rapidfuzz import process
import unicodedata
import unidecode
from functools import lru_cache
from pathlib import Path
import ast

from config.constants import TEAM_LIST, canonicalize_franchid

NEGRO_LEAGUE_IDS = {"NNL", "NN2", "NSL"}

# Known aliases for common Grid-style issues
MANUAL_ALIASES = {
    "Henry Aaron": "Hank Aaron",
    "Larry Berra": "Yogi Berra",
    "Robert Gibson": "Bob Gibson",
    "Lawrence Jackson": "Bo Jackson",
    "Fred Mcgriff": "Fred McGriff",
    "Robert Dickey": "R.A. Dickey",
    "Allan Burnett": "A.J. Burnett",
    "David Parker": "Dave Parker",
    "William Madlock": "Bill Madlock",
    "Melvin Cabrera": "Miguel Cabrera",
    "George Ruth": "Babe Ruth",
    "John Pedro Gonzalez": "Pedro Gonzalez",
    "Happ": "Ian Happ",
    "Jarrod": "Jarrod Saltalamacchia",
    "Grover Alexander": "Grover Cleveland Alexander",
    "Bobby Witt Jr.": "Bobby Witt Jr.",
    "Fernando Tatis Jr.": "Fernando Tatis Jr.",
    "Vladimir Guerrero Jr.": "Vladimir Guerrero Jr.",
    # Add more as needed...
    "Kike Hernandez": "Enrique Hernandez",
    "Kiké Hernandez": "Enrique Hernandez",
    "Joe Jackson": "Shoeless Joe Jackson",
}

def load_mlb_player_names():
    # Offline-only source: local Lahman cache names.
    local_names, _, _ = load_local_player_name_team_map()
    return local_names


@lru_cache(maxsize=1)
def load_local_player_name_team_map(cache_dir: str = "bin/baseball_cache"):
    """
    Load local Lahman-derived player -> franchID context from cache CSVs.
    Returns:
      - local_names: set[str]
      - name_to_franchids: dict[str, set[str]]
      - name_to_lgids: dict[str, set[str]]
    """
    cache_path = Path(cache_dir)
    people_path = cache_path / "People.csv"
    teams_path = cache_path / "teams.csv"
    apps_path = cache_path / "appearances.csv"
    if not people_path.exists():
        return set(), {}, {}

    try:
        players = pd.read_csv(people_path)
        pid_col = "playerID"
        first_col = "nameFirst"
        last_col = "nameLast"
        players = players[[pid_col, first_col, last_col]].copy()
        players.columns = ["player_id", "name_first", "name_last"]
    except Exception:
        return set(), {}, {}

    player_to_franch = {}
    player_to_lgid = {}
    if teams_path.exists() and apps_path.exists():
        try:
            teams = pd.read_csv(teams_path, usecols=["yearID", "teamID", "franchID", "lgID"])
            apps = pd.read_csv(apps_path, usecols=["yearID", "teamID", "playerID"])
            teams["franchID"] = teams["franchID"].apply(canonicalize_franchid)
            teams["lgID"] = teams["lgID"].astype(str).str.upper()
            merged = apps.merge(teams, on=["yearID", "teamID"], how="left")
            player_to_franch = (
                merged.groupby("playerID", dropna=False)["franchID"]
                .agg(lambda s: {str(v) for v in s.dropna().tolist()})
                .to_dict()
            )
            player_to_lgid = (
                merged.groupby("playerID", dropna=False)["lgID"]
                .agg(lambda s: {str(v) for v in s.dropna().tolist()})
                .to_dict()
            )
        except Exception:
            player_to_franch = {}
            player_to_lgid = {}

    if player_to_franch is None:
        player_to_franch = {}
    if player_to_lgid is None:
        player_to_lgid = {}

    name_to_franchids = {}
    name_to_lgids = {}
    for _, row in players.iterrows():
        pid = str(row.get("player_id", "")).strip()
        first = str(row.get("name_first", "")).strip()
        last = str(row.get("name_last", "")).strip()
        if not first or not last:
            continue
        full_name = clean_name(f"{first} {last}")
        if not full_name:
            continue
        name_to_franchids.setdefault(full_name, set()).update(player_to_franch.get(pid, set()))
        name_to_lgids.setdefault(full_name, set()).update(player_to_lgid.get(pid, set()))

    return set(name_to_franchids.keys()), name_to_franchids, name_to_lgids


def _parse_prompt_parts(prompt) -> list[str]:
    if prompt is None:
        return []

    parts = []
    if isinstance(prompt, (tuple, list)):
        parts = [str(p) for p in prompt]
    elif isinstance(prompt, str):
        p = prompt.strip()
        if " + " in p:
            parts = [x.strip() for x in p.split(" + ")]
        else:
            try:
                parsed = ast.literal_eval(p)
                if isinstance(parsed, (tuple, list)):
                    parts = [str(x) for x in parsed]
                else:
                    parts = [p]
            except Exception:
                parts = [p]
    else:
        parts = [str(prompt)]
    return parts


def _extract_team_codes_from_prompt(prompt) -> set[str]:
    parts = _parse_prompt_parts(prompt)
    if not parts:
        return set()

    team_codes = set()
    for part in parts:
        part_low = str(part).lower()
        part_up = str(part).upper()
        for team_name, code in TEAM_LIST.items():
            if str(team_name).lower() in part_low or part_up == str(code).upper():
                team_codes.add(canonicalize_franchid(code))
    return team_codes


def _extract_league_codes_from_prompt(prompt) -> set[str]:
    parts = _parse_prompt_parts(prompt)
    if not parts:
        return set()

    league_codes = set()
    for part in parts:
        part_str = str(part)
        part_low = part_str.lower()
        tokens = re.findall(r"[A-Za-z0-9]+", part_str.upper())
        for tok in tokens:
            if tok in NEGRO_LEAGUE_IDS:
                league_codes.add(tok)
        if "negro league" in part_low or "negro leagues" in part_low:
            league_codes.update(NEGRO_LEAGUE_IDS)
    return league_codes


def _pick_best_candidate(
    cleaned: str,
    team_codes: set[str],
    league_codes: set[str],
    all_names: set[str],
    local_name_to_franchids: dict[str, set[str]],
    local_name_to_lgids: dict[str, set[str]],
) -> tuple[str, float]:
    if not cleaned:
        return cleaned, 0.0

    top = process.extract(cleaned, all_names, limit=20)
    if not top:
        return cleaned, 0.0

    best_name = cleaned
    best_score = -10_000.0
    best_raw_fuzzy = 0.0
    for candidate, fuzzy_score, _ in top:
        score = float(fuzzy_score)
        candidate_franch = local_name_to_franchids.get(candidate, set())
        candidate_lgids = local_name_to_lgids.get(candidate, set())
        if team_codes and candidate_franch:
            # Strongly prefer candidates that satisfy full team context for the cell.
            if team_codes.issubset(candidate_franch):
                score += 25.0
            elif candidate_franch.intersection(team_codes):
                score += 8.0
            else:
                score -= 8.0
        elif team_codes and not candidate_franch:
            score -= 4.0

        if league_codes and candidate_lgids:
            if league_codes.issubset(candidate_lgids):
                score += 15.0
            elif candidate_lgids.intersection(league_codes):
                score += 6.0
            else:
                score -= 6.0
        elif league_codes and not candidate_lgids:
            score -= 3.0

        if score > best_score:
            best_score = score
            best_name = candidate
            best_raw_fuzzy = float(fuzzy_score)

    return best_name, best_raw_fuzzy

def clean_name(name):
    if not isinstance(name, str):
        return name

    # Normalize accents (e.g., José → Jose)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')

    # Replace problematic characters
    name = name.replace('!', 'l')
    name = name.replace('|', ' ')  # << this is the fix: treat pipe as space
    name = name.strip()

    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)

    # Fix suffixes like Ji/J./Sr./etc
    suffix = ''
    if re.search(r'\b(Jr|Ji|J\.|Sr|S\.?)\b$', name, re.IGNORECASE):
        suffix = 'Jr' if re.search(r'\bJ', name, re.IGNORECASE) else 'Sr'
        name = re.sub(r'\b(Jr|Ji|J\.|Sr|S\.?)\b$', '', name, flags=re.IGNORECASE).strip()

    # Remove anything not part of a legit name
    name = re.sub(r'[^A-Za-z.\-\' ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    if suffix:
        name = f"{name} {suffix}"

    return name.title()



def correct_typos_with_fuzzy_matching(df, response_column, similarity_threshold=0.85, progress_callback=None, verbose=True):
    if verbose:
        print(f"Correcting typos in column '{response_column}'...")
    mlb_names = load_mlb_player_names()
    local_names, local_name_to_franchids, local_name_to_lgids = load_local_player_name_team_map()
    all_names = set(mlb_names).union(local_names)
    canonical_mapping = {}

    def get_corrected(name, prompt=None):
        cleaned = clean_name(name)
        team_codes = _extract_team_codes_from_prompt(prompt)
        league_codes = _extract_league_codes_from_prompt(prompt)
        cache_key = (cleaned, tuple(sorted(team_codes)), tuple(sorted(league_codes)))
        if cache_key in canonical_mapping:
            return canonical_mapping[cache_key]

        # Apply manual override if available
        if cleaned in MANUAL_ALIASES:
            out = (
                MANUAL_ALIASES[cleaned],
                "manual_alias",
                f"manual alias: '{cleaned}' -> '{MANUAL_ALIASES[cleaned]}'",
            )
            canonical_mapping[cache_key] = out
            return out

        # Team-aware fuzzy match when prompt context exists.
        match, best_raw = _pick_best_candidate(
            cleaned,
            team_codes,
            league_codes,
            all_names,
            local_name_to_franchids,
            local_name_to_lgids,
        )
        # Only autocorrect when confidence is strong, or when team context strongly supports the candidate.
        threshold_pct = float(similarity_threshold) * 100.0
        candidate_franch = local_name_to_franchids.get(match, set())
        candidate_lgids = local_name_to_lgids.get(match, set())
        has_team_support = bool(team_codes) and bool(candidate_franch.intersection(team_codes))
        has_league_support = bool(league_codes) and bool(candidate_lgids.intersection(league_codes))
        reason_code = "fuzzy_high_confidence"
        reason_text = f"high-confidence fuzzy match (score={best_raw:.1f})"
        if has_team_support or has_league_support:
            reason_code = "fuzzy_context"
            reason_text = (
                f"context-aware fuzzy match (score={best_raw:.1f}); "
                f"prompt_teams={','.join(sorted(team_codes))}; "
                f"candidate_teams={','.join(sorted(candidate_franch))}; "
                f"prompt_leagues={','.join(sorted(league_codes))}; "
                f"candidate_leagues={','.join(sorted(candidate_lgids))}"
            )
        if best_raw < threshold_pct and not has_team_support and not has_league_support:
            out = (
                cleaned,
                "unchanged_low_confidence",
                f"left unchanged: low fuzzy score ({best_raw:.1f} < {threshold_pct:.1f}) and no context support",
            )
            canonical_mapping[cache_key] = out
            return out
        out = (match, reason_code, reason_text)
        canonical_mapping[cache_key] = out
        return out

    corrected_rows, changes_log = [], []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        original = row[response_column]
        corrected = original

        if isinstance(original, dict):
            corrected = {}
            for k, v in original.items():
                if isinstance(v, str) and len(v.strip()) > 0:
                    new, reason_code, reason_text = get_corrected(v, row.get("prompt"))
                    if new != v:
                        changes_log.append({
                            "grid_number": row['grid_number'],
                            "submitter": row['submitter'],
                            "original_name": v,
                            "corrected_name": new,
                            "response_location": k,
                            "prompt": row.get("prompt"),
                            "team_context": ", ".join(sorted(_extract_team_codes_from_prompt(row.get("prompt")))),
                            "league_context": ", ".join(sorted(_extract_league_codes_from_prompt(row.get("prompt")))),
                            "audit_reason": reason_code,
                            "audit_reason_detail": reason_text,
                        })
                    corrected[k] = new
                else:
                    corrected[k] = v
        elif isinstance(original, str) and len(original.strip()) > 0:
            new, reason_code, reason_text = get_corrected(original, row.get("prompt"))
            if new != original:
                changes_log.append({
                    "grid_number": row['grid_number'],
                    "submitter": row['submitter'],
                    "original_name": original,
                    "corrected_name": new,
                    "response_location": None,
                    "prompt": row.get("prompt"),
                    "team_context": ", ".join(sorted(_extract_team_codes_from_prompt(row.get("prompt")))),
                    "league_context": ", ".join(sorted(_extract_league_codes_from_prompt(row.get("prompt")))),
                    "audit_reason": reason_code,
                    "audit_reason_detail": reason_text,
                })
            corrected = new

        row_copy = row.copy()
        row_copy[response_column] = corrected
        corrected_rows.append(row_copy)

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    return pd.DataFrame(corrected_rows), pd.DataFrame(changes_log)
