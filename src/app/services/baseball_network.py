from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

RosterKey = tuple[str, int]


@dataclass(frozen=True)
class TeammateHop:
    from_player: str
    to_player: str
    team_id: str
    year: int
    franchise_id: str | None = None
    team_name: str | None = None


class BaseballNetwork:
    """Memory-efficient player <-> team-season network built from Lahman data."""

    def __init__(self, appearances: pd.DataFrame, teams: pd.DataFrame, people: pd.DataFrame, canonicalize_franchid: Callable[[str], str] | None = None) -> None:
        self.canonicalize_franchid = canonicalize_franchid or (lambda value: str(value))
        self.memberships = self._normalize_memberships(appearances, teams)
        self.player_names = self._build_player_names(people)
        self.player_to_rosters: dict[str, set[RosterKey]] = defaultdict(set)
        self.roster_to_players: dict[RosterKey, set[str]] = defaultdict(set)
        self.roster_metadata: dict[RosterKey, dict[str, object]] = {}
        self.player_years: dict[str, tuple[int, int]] = {}
        self._component_by_player: dict[str, int] | None = None
        self._component_sizes: dict[int, int] | None = None
        self._build_indexes()

    def _normalize_memberships(self, appearances: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        required = {"playerID", "teamID", "yearID"}
        missing = required - set(appearances.columns)
        if missing:
            raise ValueError(f"Appearances is missing required columns: {sorted(missing)}")
        team_columns = [c for c in ["teamID", "yearID", "franchID", "name"] if c in teams.columns]
        merged = appearances[["playerID", "teamID", "yearID"]].copy()
        if {"teamID", "yearID"}.issubset(team_columns):
            merged = merged.merge(teams[team_columns].drop_duplicates(["teamID", "yearID"]), on=["teamID", "yearID"], how="left")
        merged = merged.dropna(subset=["playerID", "teamID", "yearID"]).copy()
        merged["playerID"] = merged["playerID"].astype(str)
        merged["teamID"] = merged["teamID"].astype(str)
        merged["yearID"] = pd.to_numeric(merged["yearID"], errors="coerce")
        merged = merged.dropna(subset=["yearID"])
        merged["yearID"] = merged["yearID"].astype(int)
        if "franchID" in merged.columns:
            merged["franchID"] = merged["franchID"].apply(lambda value: self.canonicalize_franchid(value) if pd.notna(value) else None)
        return merged.drop_duplicates(["playerID", "teamID", "yearID"]).sort_values(["yearID", "teamID", "playerID"]).reset_index(drop=True)

    @staticmethod
    def _build_player_names(people: pd.DataFrame) -> dict[str, str]:
        if "playerID" in people.columns:
            id_col, first_col, last_col = "playerID", "nameFirst", "nameLast"
        else:
            id_col, first_col, last_col = "key_bbref", "name_first", "name_last"
        names: dict[str, str] = {}
        for _, row in people.iterrows():
            player_id = str(row.get(id_col, "")).strip()
            if not player_id:
                continue
            first_value = row.get(first_col, "")
            last_value = row.get(last_col, "")
            first = str(first_value if pd.notna(first_value) else "").strip()
            last = str(last_value if pd.notna(last_value) else "").strip()
            names[player_id] = f"{first} {last}".strip() or player_id
        return names

    def _build_indexes(self) -> None:
        years: dict[str, list[int]] = defaultdict(list)
        for row in self.memberships.to_dict("records"):
            player_id = str(row["playerID"])
            roster = (str(row["teamID"]), int(row["yearID"]))
            self.player_to_rosters[player_id].add(roster)
            self.roster_to_players[roster].add(player_id)
            years[player_id].append(roster[1])
            self.roster_metadata[roster] = {"team_id": roster[0], "year": roster[1], "franchise_id": row.get("franchID"), "team_name": row.get("name")}
        self.player_years = {player: (min(values), max(values)) for player, values in years.items()}

    def name(self, player_id: str) -> str:
        return self.player_names.get(str(player_id), str(player_id))

    def player_options(self) -> list[tuple[str, str]]:
        return sorted(((player, self.name(player)) for player in self.player_to_rosters), key=lambda item: (item[1], item[0]))

    def shared_rosters(self, player_a: str, player_b: str) -> list[RosterKey]:
        return sorted(self.player_to_rosters.get(str(player_a), set()) & self.player_to_rosters.get(str(player_b), set()), key=lambda r: (r[1], r[0]))

    def are_teammates(self, player_a: str, player_b: str) -> bool:
        return bool(self.shared_rosters(player_a, player_b))

    def teammates(self, player_id: str) -> Iterable[tuple[str, RosterKey]]:
        player_id = str(player_id)
        seen: set[str] = set()
        for roster in sorted(self.player_to_rosters.get(player_id, set()), key=lambda r: (r[1], r[0])):
            for teammate in sorted(self.roster_to_players[roster]):
                if teammate != player_id and teammate not in seen:
                    seen.add(teammate)
                    yield teammate, roster

    def shortest_path(self, start: str, target: str) -> list[TeammateHop] | None:
        start, target = str(start), str(target)
        if start == target:
            return []
        if start not in self.player_to_rosters or target not in self.player_to_rosters:
            return None
        return self._shortest_path_to_any(start, {target})

    def _shortest_path_to_any(self, start: str, targets: set[str]) -> list[TeammateHop] | None:
        if start in targets:
            return []
        queue = deque([start])
        parent: dict[str, tuple[str, RosterKey] | None] = {start: None}
        while queue:
            player = queue.popleft()
            for teammate, roster in self.teammates(player):
                if teammate in parent:
                    continue
                parent[teammate] = (player, roster)
                if teammate in targets:
                    return self._reconstruct_path(parent, teammate)
                queue.append(teammate)
        return None

    def _reconstruct_path(self, parent: dict[str, tuple[str, RosterKey] | None], target: str) -> list[TeammateHop]:
        reversed_hops: list[TeammateHop] = []
        current = target
        while parent[current] is not None:
            previous, roster = parent[current]
            meta = self.roster_metadata[roster]
            reversed_hops.append(TeammateHop(previous, current, roster[0], roster[1], meta.get("franchise_id"), meta.get("team_name")))
            current = previous
        return list(reversed(reversed_hops))

    def path_to_earliest_baseball(self, start: str) -> list[TeammateHop] | None:
        start = str(start)
        if start not in self.player_to_rosters:
            return None
        earliest_year = min(roster[1] for roster in self.roster_to_players)
        targets = {player for roster, players in self.roster_to_players.items() if roster[1] == earliest_year for player in players}
        return self._shortest_path_to_any(start, targets)

    def historical_relay(self, start: str, max_hops: int = 40) -> list[TeammateHop]:
        current = str(start)
        if current not in self.player_to_rosters:
            return []
        visited = {current}
        hops: list[TeammateHop] = []
        frontier = self.player_years[current][0]
        for _ in range(max_hops):
            candidates: list[tuple[int, int, str, RosterKey]] = []
            for roster in self.player_to_rosters[current]:
                for teammate in self.roster_to_players[roster]:
                    if teammate in visited:
                        continue
                    candidates.append((self.player_years[teammate][0], roster[1], teammate, roster))
            if not candidates:
                break
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            first_year, _, teammate, roster = candidates[0]
            if first_year >= frontier:
                break
            meta = self.roster_metadata[roster]
            hops.append(TeammateHop(current, teammate, roster[0], roster[1], meta.get("franchise_id"), meta.get("team_name")))
            visited.add(teammate)
            current = teammate
            frontier = first_year
        return hops

    def connected_components(self) -> tuple[dict[str, int], dict[int, int]]:
        if self._component_by_player is not None and self._component_sizes is not None:
            return self._component_by_player, self._component_sizes
        component_by_player: dict[str, int] = {}
        component_sizes: dict[int, int] = {}
        component_id = 0
        for start in sorted(self.player_to_rosters):
            if start in component_by_player:
                continue
            queue = deque([start])
            component_by_player[start] = component_id
            size = 0
            while queue:
                player = queue.popleft()
                size += 1
                for teammate, _ in self.teammates(player):
                    if teammate not in component_by_player:
                        component_by_player[teammate] = component_id
                        queue.append(teammate)
            component_sizes[component_id] = size
            component_id += 1
        self._component_by_player = component_by_player
        self._component_sizes = component_sizes
        return component_by_player, component_sizes

    def statistics(self) -> dict[str, int | float]:
        _, sizes = self.connected_components()
        total_players = len(self.player_to_rosters)
        largest = max(sizes.values(), default=0)
        years = [roster[1] for roster in self.roster_to_players]
        return {"players": total_players, "team_seasons": len(self.roster_to_players), "memberships": len(self.memberships), "components": len(sizes), "largest_component": largest, "largest_component_pct": (100.0 * largest / total_players) if total_players else 0.0, "earliest_year": min(years) if years else 0, "latest_year": max(years) if years else 0}

    def oldest_player_walk(self, start_roster: RosterKey, oldest_players: pd.DataFrame, max_steps: int = 100) -> list[dict[str, object]]:
        if oldest_players.empty:
            return []
        player_col = "playerID" if "playerID" in oldest_players.columns else "key_bbref"
        lookup = oldest_players.copy()
        lookup["teamID"] = lookup["teamID"].astype(str)
        lookup["yearID"] = pd.to_numeric(lookup["yearID"], errors="coerce")
        lookup = lookup.dropna(subset=["yearID"])
        lookup["yearID"] = lookup["yearID"].astype(int)
        oldest_by_roster = {(str(row["teamID"]), int(row["yearID"])): str(row[player_col]) for row in lookup.to_dict("records")}
        current = (str(start_roster[0]), int(start_roster[1]))
        visited_rosters: set[RosterKey] = set()
        rows: list[dict[str, object]] = []
        for _ in range(max_steps):
            if current in visited_rosters or current not in oldest_by_roster:
                break
            visited_rosters.add(current)
            player = oldest_by_roster[current]
            rows.append({"teamID": current[0], "yearID": current[1], "playerID": player, "player_name": self.name(player)})
            earlier = [roster for roster in self.player_to_rosters.get(player, set()) if roster[1] < current[1] and roster not in visited_rosters]
            if not earlier:
                break
            current = min(earlier, key=lambda roster: (roster[1], roster[0]))
        return rows
