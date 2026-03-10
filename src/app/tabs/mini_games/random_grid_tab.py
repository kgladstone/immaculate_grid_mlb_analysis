from __future__ import annotations

import pandas as pd
import streamlit as st


def render_random_grid(
    selectable_team_codes: list[str],
    team_code_to_name: dict[str, str],
    player_sets: list[set[str]],
    player_lookup_df: pd.DataFrame,
    player_franch_map: dict[str, set[str]],
    canonicalize_franchid_fn,
    generate_random_grid_puzzle_fn,
    validate_fuzzy_answer_fn,
) -> None:
    st.markdown("### Random Immaculate Grid (3x3)")
    st.caption("Generates random 3x3 team intersections where every cell has at least 1 valid player.")
    if st.button("Generate random 3x3 puzzle", key="sim_generate_grid"):
        puzzle = generate_random_grid_puzzle_fn(selectable_team_codes, player_sets)
        st.session_state["sim_random_grid_puzzle"] = puzzle

    puzzle = st.session_state.get("sim_random_grid_puzzle")
    if not puzzle:
        st.info("Click generate to create a guaranteed-solvable 3x3 puzzle.")
        return

    row_codes, col_codes, counts = puzzle
    df = pd.DataFrame(
        counts,
        index=[f"{team_code_to_name.get(c, c)} ({c})" for c in row_codes],
        columns=[f"{team_code_to_name.get(c, c)} ({c})" for c in col_codes],
    )
    st.write("Cell solution counts")
    st.dataframe(df, use_container_width=True)
    st.success("All 9 cells have at least one valid solution.")

    st.markdown("#### Fill out puzzle")
    st.caption("Type any player spelling; checker uses fuzzy matching to resolve to the closest player.")
    for r_idx, r_code in enumerate(row_codes):
        cols = st.columns(3)
        for c_idx, c_code in enumerate(col_codes):
            cell_key = f"sim_grid_answer_{r_idx}_{c_idx}"
            cell_label = (
                f"R{r_idx + 1}C{c_idx + 1}: "
                f"{team_code_to_name.get(r_code, r_code)} + {team_code_to_name.get(c_code, c_code)}"
            )
            cols[c_idx].text_input(cell_label, key=cell_key)

    if st.button("Check puzzle (3x3)", key="sim_check_random_grid"):
        result_rows = []
        solved_count = 0
        for r_idx, r_code in enumerate(row_codes):
            for c_idx, c_code in enumerate(col_codes):
                required = {canonicalize_franchid_fn(r_code), canonicalize_franchid_fn(c_code)}
                answer = st.session_state.get(f"sim_grid_answer_{r_idx}_{c_idx}", "")
                check = validate_fuzzy_answer_fn(answer, required, player_lookup_df, player_franch_map)
                solved_count += int(check["ok"])
                result_rows.append(
                    {
                        "cell": f"R{r_idx + 1}C{c_idx + 1}",
                        "required": " + ".join(sorted(required)),
                        "answer": check["answer"],
                        "matched_player": check["matched_player"],
                        "player_id": check["player_id"],
                        "confidence": f"{check['confidence']:.1%}",
                        "ok": "yes" if check["ok"] else "no",
                        "missing_franchids": ", ".join(check["missing_franchids"]),
                    }
                )
        st.write(f"Score: **{solved_count}/9**")
        if solved_count == 9:
            st.success("Puzzle solved.")
        else:
            st.warning("Puzzle not fully solved yet.")
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

