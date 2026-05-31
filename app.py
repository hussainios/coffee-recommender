from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from landscape import load_feature_index, recommend_from_landscape
from parse_review import parse_review_event
from reviewed_coffee_url import normalise_source_url, prepare_reviewed_coffee_from_url
from visualize_landscape import build_projected_score_landscape_figure


def reset_review_history(session_state) -> None:
    session_state.review_events = []
    session_state.reviewed_feature_overrides = {}
    session_state.last_event = None
    session_state.last_recommendations = []


def build_scoring_features(features, reviewed_feature_overrides):
    scoring_features = dict(features)
    scoring_features.update(reviewed_feature_overrides)
    return scoring_features


def append_review_event(session_state, event, reviewed_coffee, *, is_temporary: bool) -> None:
    session_state.review_events.append(event)
    session_state.last_event = event
    if is_temporary:
        session_state.reviewed_feature_overrides[reviewed_coffee.coffee_id] = reviewed_coffee


st.set_page_config(page_title="Coffee Recommender", page_icon="☕", layout="wide")

st.title("Coffee Recommender")
st.write("Pick a coffee you reviewed, describe it, and inspect landscape-based recommendations.")

if "last_event" not in st.session_state:
    st.session_state.last_event = None
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = []
if "review_events" not in st.session_state:
    st.session_state.review_events = []
if "reviewed_feature_overrides" not in st.session_state:
    st.session_state.reviewed_feature_overrides = {}
if "url_reviewed_coffee" not in st.session_state:
    st.session_state.url_reviewed_coffee = None
if "url_reviewed_source" not in st.session_state:
    st.session_state.url_reviewed_source = ""
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Catalogue coffee"
if "data_paths_key" not in st.session_state:
    st.session_state.data_paths_key = None

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

coffees = pd.read_csv(coffees_path)
features = load_feature_index(coffees_path, sensory_path, embeddings_path)
data_paths_key = (str(coffees_path), str(sensory_path), str(embeddings_path))
if st.session_state.data_paths_key is None:
    st.session_state.data_paths_key = data_paths_key
elif st.session_state.data_paths_key != data_paths_key:
    st.session_state.data_paths_key = data_paths_key
    reset_review_history(st.session_state)

coffee_options = {
    f"{row['name']} ({row['coffee_id']})": str(row["coffee_id"])
    for _, row in coffees.sort_values("name").iterrows()
}

input_mode = st.radio("Reviewed coffee source", options=["Catalogue coffee", "Coffee URL"], horizontal=True)
if input_mode != st.session_state.input_mode:
    st.session_state.input_mode = input_mode

selected_label = None
reviewed_coffee = None
reviewed_metadata = None
reviewed_sensory = None

if input_mode == "Catalogue coffee":
    selected_label = st.selectbox("Reviewed coffee", options=list(coffee_options))
    reviewed_coffee_id = coffee_options[selected_label]
    reviewed_coffee = features[reviewed_coffee_id]
    reviewed_metadata = coffees.loc[coffees["coffee_id"].astype(str) == reviewed_coffee_id].iloc[0].to_dict()
else:
    url_value = st.text_input(
        "Coffee product URL",
        value=st.session_state.url_reviewed_source,
        placeholder="https://...",
    ).strip()
    try:
        normalized_url_value = normalise_source_url(url_value) if url_value else ""
    except ValueError:
        normalized_url_value = ""
    process_url = st.button("Process coffee URL")

    if process_url:
        try:
            st.session_state.url_reviewed_coffee = prepare_reviewed_coffee_from_url(url_value)
            st.session_state.url_reviewed_source = st.session_state.url_reviewed_coffee.url
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.url_reviewed_coffee is not None:
        cached = st.session_state.url_reviewed_coffee
        if normalized_url_value and normalized_url_value == st.session_state.url_reviewed_source:
            reviewed_coffee = cached.features
            reviewed_metadata = cached.coffee.model_dump(mode="json")
            reviewed_sensory = cached.sensory.model_dump(mode="json")

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
                cached = prepare_reviewed_coffee_from_url(url_value)
                st.session_state.url_reviewed_coffee = cached
                st.session_state.url_reviewed_source = cached.url
            reviewed_coffee = cached.features
            reviewed_metadata = cached.coffee.model_dump(mode="json")
            reviewed_sensory = cached.sensory.model_dump(mode="json")

        if reviewed_coffee is None:
            raise ValueError("Select or process a reviewed coffee before running the recommender.")
        event = parse_review_event(review, reviewed_coffee)
        append_review_event(
            st.session_state,
            event,
            reviewed_coffee,
            is_temporary=is_temporary_reviewed_coffee,
        )
        scoring_features = build_scoring_features(features, st.session_state.reviewed_feature_overrides)
        recommendations = recommend_from_landscape(scoring_features, st.session_state.review_events, top_k=top_k)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.session_state.last_recommendations = recommendations

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
        scoring_features = build_scoring_features(features, st.session_state.reviewed_feature_overrides)
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
    scoring_features = build_scoring_features(features, st.session_state.reviewed_feature_overrides)
    figure = build_projected_score_landscape_figure(
        catalogue_features=features,
        scoring_features=scoring_features,
        reviews=st.session_state.review_events,
        top_recommendations=st.session_state.last_recommendations,
        show_surface=show_landscape_surface,
    )
    if figure is None:
        st.info("Need at least three coffees to project the score landscape.")
    else:
        st.plotly_chart(figure, use_container_width=True)
