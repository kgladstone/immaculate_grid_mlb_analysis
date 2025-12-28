from __future__ import annotations

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Callable
from utils.constants import GRID_PLAYERS_RESTRICTED

from utils.analysis import (
    analyze_everyone_missed,
    analyze_illegal_uses,
    analyze_all_used_on_same_day,
    analyze_person_type_performance,
    analyze_team_performance,
    analyze_person_prompt_performance,
    analyze_hardest_intersections,
    analyze_most_successful_exact_intersections,
    analyze_empty_team_team_intersections,
    plot_summary_metrics,
    plot_correctness,
    plot_smoothed_metrics,
    plot_win_rates,
    plot_best_worst_scores,
    plot_best_worst_scores_30,
    analyze_top_players_by_submitter,
    analyze_grid_cell_with_shared_guesses,
    analyze_top_players_by_month,
    plot_top_n_grids,
    analyze_person_to_category,
    analyze_submitter_specific_players,
    get_favorite_player_by_attribute,
    get_players_used_for_most_teams,
    analyze_immaculate_streaks,
    analyze_splits,
    analyze_new_players_per_month,
    analyze_full_week_usage,
    analyze_shame_index,
    analyze_low_bit_high_reward,
    analyze_novelties,
    analyze_prediction_future_grid,
)

REPORT_BANK_PATH = Path(__file__).resolve().parent / "report_bank_data.json"


FUNCTIONS: Dict[str, Callable] = {
    "plot_summary_metrics": plot_summary_metrics,
    "plot_correctness": plot_correctness,
    "plot_smoothed_metrics": plot_smoothed_metrics,
    "plot_win_rates": plot_win_rates,
    "plot_best_worst_scores": plot_best_worst_scores,
    "plot_best_worst_scores_30": plot_best_worst_scores_30,
    "plot_top_n_grids": plot_top_n_grids,
    "analyze_new_players_per_month": analyze_new_players_per_month,
    "analyze_person_type_performance": analyze_person_type_performance,
    "analyze_person_to_category": analyze_person_to_category,
    "analyze_team_performance": analyze_team_performance,
    "analyze_person_prompt_performance": analyze_person_prompt_performance,
    "analyze_hardest_intersections": analyze_hardest_intersections,
    "analyze_top_players_by_submitter": analyze_top_players_by_submitter,
    "analyze_submitter_specific_players": analyze_submitter_specific_players,
    "get_favorite_player_by_attribute": get_favorite_player_by_attribute,
    "get_players_used_for_most_teams": get_players_used_for_most_teams,
    "analyze_grid_cell_with_shared_guesses": analyze_grid_cell_with_shared_guesses,
    "analyze_immaculate_streaks": analyze_immaculate_streaks,
    "analyze_splits": analyze_splits,
    "analyze_everyone_missed": analyze_everyone_missed,
    "analyze_all_used_on_same_day": analyze_all_used_on_same_day,
    "analyze_illegal_uses": analyze_illegal_uses,
    "analyze_top_players_by_month": analyze_top_players_by_month,
    "analyze_most_successful_exact_intersections": analyze_most_successful_exact_intersections,
    "analyze_empty_team_team_intersections": analyze_empty_team_team_intersections,
    "analyze_full_week_usage": analyze_full_week_usage,
    "analyze_shame_index": analyze_shame_index,
    "analyze_low_bit_high_reward": analyze_low_bit_high_reward,
    "analyze_novelties": analyze_novelties,
    "analyze_prediction_future_grid": analyze_prediction_future_grid,
    "raw_prompts": lambda prompts: prompts,
    "raw_results": lambda texts_raw: texts_raw,
    "raw_images_metadata": lambda images_raw: images_raw,
}


def _resolve_arg(token: Any, ctx: Dict[str, Any], person: Any) -> Any:
    placeholders = {
        "texts": ctx.get("texts"),
        "texts_raw": ctx.get("texts_raw"),
        "prompts": ctx.get("prompts"),
        "categories": ctx.get("categories"),
        "images": ctx.get("images"),
        "images_raw": ctx.get("images_raw"),
        "color_map": ctx.get("color_map"),
        "person": person,
        "restricted": GRID_PLAYERS_RESTRICTED,
    }
    if isinstance(token, str) and token in placeholders:
        return placeholders[token]
    return token


def run_report(entry: Dict[str, Any], ctx: Dict[str, Any], person: Any = None):
    func = FUNCTIONS.get(entry["func"])
    if not func:
        return None
    args = [_resolve_arg(arg, ctx, person) for arg in entry.get("args", [])]
    try:
        return func(*args)
    except Exception as exc:
        return {"__error__": str(exc)}


def load_report_bank() -> List[Dict[str, Any]]:
    raw = json.loads(REPORT_BANK_PATH.read_text())
    reports: List[Dict[str, Any]] = []
    for entry in raw:
        if entry.get("func") not in FUNCTIONS:
            continue
        if "category" not in entry:
            entry["category"] = "Misc"
        reports.append(entry)
    return reports
