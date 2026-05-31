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


st.set_page_config(page_title="Coffee Recommender", page_icon="☕", layout="wide")

st.title("Coffee Recommender")
st.write("Pick a coffee you reviewed, describe it, and inspect landscape-based recommendations.")

if "last_event" not in st.session_state:
    st.session_state.last_event = None
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = []
if "url_reviewed_coffee" not in st.session_state:
    st.session_state.url_reviewed_coffee = None
if "url_reviewed_source" not in st.session_state:
    st.session_state.url_reviewed_source = ""
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Catalogue coffee"

with st.sidebar:
    st.header("Data")
    coffees_path = Path(st.text_input("Coffee CSV", value="data/processed/coffees.csv"))
    sensory_path = Path(st.text_input("Sensory CSV", value="data/processed/coffee_sensory_vectors.csv"))
    embeddings_path = Path(st.text_input("Embeddings CSV", value="data/processed/coffee_embeddings.csv"))
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

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

coffee_options = {
    f"{row['name']} ({row['coffee_id']})": str(row["coffee_id"])
    for _, row in coffees.sort_values("name").iterrows()
}

input_mode = st.radio("Reviewed coffee source", options=["Catalogue coffee", "Coffee URL"], horizontal=True)
if input_mode != st.session_state.input_mode:
    st.session_state.input_mode = input_mode
    st.session_state.last_event = None
    st.session_state.last_recommendations = []

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
            st.session_state.last_event = None
            st.session_state.last_recommendations = []
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

if st.button("Parse review and recommend", type="primary"):
    try:
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
        scoring_features = dict(features)
        scoring_features[reviewed_coffee.coffee_id] = reviewed_coffee
        recommendations = recommend_from_landscape(scoring_features, [event], top_k=top_k)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.session_state.last_event = event
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

    st.subheader("Parsed Review Event")
    if st.session_state.last_event:
        st.json(st.session_state.last_event)
    else:
        st.info("Run the recommender to see the parsed event.")

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
