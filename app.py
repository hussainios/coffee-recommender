from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_state import (
    build_scoring_features,
    initialise_review_state,
    reset_review_history,
    reset_review_history_if_data_paths_changed,
)
from coffee_service import (
    build_coffee_options,
    get_cached_url_selection,
    load_catalogue,
    normalise_optional_url,
    prepare_url_selection,
    select_catalogue_reviewed_coffee,
    selection_from_url_reviewed_coffee,
    submit_review,
)
from visualize_landscape import build_projected_score_landscape_figure


st.set_page_config(page_title="Coffee Recommender", page_icon="☕", layout="wide")

st.title("Coffee Recommender")
st.write("Pick a coffee you reviewed, describe it, and inspect landscape-based recommendations.")

initialise_review_state(st.session_state)

with st.sidebar:
    st.header("Data")
    coffees_path = Path(st.text_input("Coffee CSV", value="data/processed/coffees.csv"))
    sensory_path = Path(st.text_input("Sensory CSV", value="data/processed/coffee_sensory_vectors.csv"))
    embeddings_path = Path(st.text_input("Embeddings CSV", value="data/processed/coffee_embeddings.csv"))
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    if st.button("Clear review history"):
        reset_review_history(st.session_state)

missing_paths = [
    path
    for path in (coffees_path, sensory_path, embeddings_path)
    if not path.exists()
]
if missing_paths:
    for path in missing_paths:
        st.error(f"Missing required file: {path}")
    if embeddings_path in missing_paths:
        st.info("Generate embeddings with: .venv/bin/python src/process_data/build_embeddings.py")
    st.stop()

catalogue = load_catalogue(coffees_path, sensory_path, embeddings_path)
reset_review_history_if_data_paths_changed(st.session_state, catalogue.data_paths_key)
coffee_options = build_coffee_options(catalogue.coffees)

input_mode = st.radio("Reviewed coffee source", options=["Catalogue coffee", "Coffee URL"], horizontal=True)
if input_mode != st.session_state.input_mode:
    st.session_state.input_mode = input_mode

selected_label = None
reviewed_coffee = None
reviewed_metadata = None
reviewed_sensory = None
url_value = ""
normalized_url_value = ""
selected_reviewed = None

if input_mode == "Catalogue coffee":
    selected_label = st.selectbox("Reviewed coffee", options=list(coffee_options))
    reviewed_coffee_id = coffee_options[selected_label]
    selected_reviewed = select_catalogue_reviewed_coffee(catalogue, reviewed_coffee_id)
else:
    url_value = st.text_input(
        "Coffee product URL",
        value=st.session_state.url_reviewed_source,
        placeholder="https://...",
    ).strip()
    try:
        normalized_url_value = normalise_optional_url(url_value)
    except ValueError:
        normalized_url_value = ""
    process_url = st.button("Process coffee URL")

    if process_url:
        try:
            st.session_state.url_reviewed_coffee = prepare_url_selection(url_value)
            st.session_state.url_reviewed_source = st.session_state.url_reviewed_coffee.url
        except Exception as exc:
            st.error(str(exc))

    selected_reviewed = get_cached_url_selection(
        url_value,
        st.session_state.url_reviewed_coffee,
        st.session_state.url_reviewed_source,
    )

if selected_reviewed is not None:
    reviewed_coffee = selected_reviewed.features
    reviewed_metadata = selected_reviewed.metadata
    reviewed_sensory = selected_reviewed.sensory

review = st.text_area(
    "Your review",
    value="I liked this coffee, but it was a little too acidic.",
    height=120,
)

if st.button("Add review and recommend", type="primary"):
    try:
        is_temporary_reviewed_coffee = input_mode == "Coffee URL"
        if input_mode == "Coffee URL":
            if not url_value:
                raise ValueError("Paste a coffee product URL before running the recommender.")
            cached = st.session_state.url_reviewed_coffee
            if cached is None or cached.url != normalized_url_value:
                cached = prepare_url_selection(url_value)
                st.session_state.url_reviewed_coffee = cached
                st.session_state.url_reviewed_source = cached.url
            selected_reviewed = selection_from_url_reviewed_coffee(cached)
            reviewed_coffee = selected_reviewed.features
            reviewed_metadata = selected_reviewed.metadata
            reviewed_sensory = selected_reviewed.sensory

        result = submit_review(
            session_state=st.session_state,
            review_text=review,
            reviewed_coffee=reviewed_coffee,
            catalogue_features=catalogue.features,
            top_k=top_k,
            is_temporary=is_temporary_reviewed_coffee,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.session_state.last_recommendations = result.recommendations

col1, col2 = st.columns([0.9, 1.1])

with col1:
    st.subheader("Reviewed Coffee")
    if reviewed_coffee is None:
        st.info("Select a catalogue coffee or process a coffee URL to inspect it here.")
    else:
        st.markdown(f"### {reviewed_coffee.name}")
        st.caption(reviewed_coffee.coffee_id)

        if reviewed_metadata is not None:
            st.write("Metadata")
            st.json(reviewed_metadata)

        st.write("Sensory")
        st.json(reviewed_sensory or reviewed_coffee.sensory)

        st.write("Process")
        st.json(reviewed_coffee.process)

    st.subheader("Latest Parsed Review Event")
    if st.session_state.last_event:
        st.json(st.session_state.last_event)
    else:
        st.info("Run the recommender to see the parsed event.")

    st.subheader("Review History")
    if not st.session_state.review_events:
        st.info("Added reviews will appear here.")
    else:
        scoring_features = build_scoring_features(catalogue.features, st.session_state.reviewed_feature_overrides)
        for index, event in enumerate(st.session_state.review_events, start=1):
            reviewed = scoring_features.get(str(event.get("coffee_id", "")))
            title = reviewed.name if reviewed else "Unknown coffee"
            with st.expander(f"{index}. {title}"):
                st.caption(event.get("coffee_id", ""))
                st.write(f"Overall: `{event.get('overall', 0.0)}`")
                st.write("Change requests")
                st.json(event.get("change_requests", {}))
                st.write("Attribute opinions")
                st.json(event.get("attribute_opinions", {}))

with col2:
    st.subheader("Recommendations")
    if not st.session_state.last_recommendations:
        st.info("Run the recommender to see ranked coffees.")
    else:
        for item in st.session_state.last_recommendations:
            with st.container(border=True):
                st.markdown(f"### {item['name']}")
                st.caption(item["coffee_id"])
                st.write(f"Score: `{item['score']}`")
                st.write(f"Temperature: `{item['temperature']}`")

                debug = item["debug"]
                with st.expander("Structured coffee representation"):
                    st.json(debug["candidate"])

                with st.expander("Score debug"):
                    st.json({"reviews": debug["reviews"]})

st.subheader("Projected Score Landscape")
st.caption(
    "A 2D PCA projection of the high-dimensional coffee feature space. "
    "Height and color show the active recommendation score."
)
show_landscape_surface = st.checkbox(
    "Show interpolated surface",
    value=True,
    help="Visual aid only: the surface is interpolated between real scored coffees.",
)
if not st.session_state.review_events:
    st.info("Add at least one review to plot the score landscape.")
else:
    scoring_features = build_scoring_features(catalogue.features, st.session_state.reviewed_feature_overrides)
    figure = build_projected_score_landscape_figure(
        catalogue_features=catalogue.features,
        scoring_features=scoring_features,
        reviews=st.session_state.review_events,
        top_recommendations=st.session_state.last_recommendations,
        show_surface=show_landscape_surface,
    )
    if figure is None:
        st.info("Need at least three coffees to project the score landscape.")
    else:
        st.plotly_chart(figure, use_container_width=True)
