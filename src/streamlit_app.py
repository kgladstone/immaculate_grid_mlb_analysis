from __future__ import annotations

import streamlit as st

from app.operations.data_loaders import load_image_metadata_df, load_prompts_df, load_texts_df, resolve_path
from app.tabs.refresh_tab import render_refresh_tab
from app.tabs.data_viewer_tab import render_image_metadata, render_prompts_and_texts, render_data_availability
from app.tabs.analytics_tab import render_analytics
from app.tabs.simulator_tab import render_simulator_tab
from utils.constants import IMAGES_METADATA_PATH


def main():
    st.set_page_config(page_title="Immaculate Grid Data Viewer", layout="wide")
    st.title("Immaculate Grid Data Viewer")
    st.caption("Browse cached datasets and trigger selective refreshes.")

    # Load shared data once
    prompts_df = load_prompts_df()
    texts_df = load_texts_df()
    images_df = load_image_metadata_df()

    data_tab, refresh_tab, analytics_tab, simulator_tab = st.tabs(
        ["🗂 Data Viewer", "🔄 Refresh Data", "📊 Analytics", "🎮 Simulator"]
    )

    with data_tab:
        st.subheader("Datasets")
        shared_grid_key = "shared_grid_selection"
        if shared_grid_key not in st.session_state:
            st.session_state[shared_grid_key] = None
        all_grids = sorted(
            set(prompts_df.get("grid_id", []))
            | set(texts_df.get("grid_number", []))
            | set(images_df.get("grid_number", [])),
            reverse=True,
        )
        if all_grids:
            default_idx = 0
            if st.session_state[shared_grid_key] in all_grids:
                default_idx = all_grids.index(st.session_state[shared_grid_key])
            selected_grid = st.selectbox(
                "Grid ID (applies across Data Viewer tabs)",
                all_grids,
                index=default_idx,
                key="global_grid_select",
            )
            st.session_state[shared_grid_key] = selected_grid

        combined_tab, images_tab, availability_tab = st.tabs(
            ["Masked Results", "Full Results", "Data Availability"]
        )

        with combined_tab:
            st.write("Prompts alongside everyone’s results for a selected grid.")
            render_prompts_and_texts(
                prompts_df,
                texts_df,
                selected_grid=st.session_state[shared_grid_key],
                on_select=lambda gid: st.session_state.update({shared_grid_key: gid}),
                show_selector=False,
            )

        with images_tab:
            st.write("Parsed image metadata captured from shared screenshots.")
            render_image_metadata(
                images_df,
                resolve_path(IMAGES_METADATA_PATH),
                texts_df=texts_df,
                selected_grid=st.session_state[shared_grid_key],
                on_select=lambda gid: st.session_state.update({shared_grid_key: gid}),
                show_selector=False,
            )

        with availability_tab:
            st.write("Coverage of texts and image metadata by grid and player.")
            render_data_availability(prompts_df, texts_df, images_df)

    with refresh_tab:
        render_refresh_tab()

    with analytics_tab:
        render_analytics(prompts_df, texts_df, images_df)

    with simulator_tab:
        render_simulator_tab()


if __name__ == "__main__":
    main()
