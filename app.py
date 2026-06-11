from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from coffee_recommender.api_models import LandscapeRequest, SubmitReviewRequest
from coffee_recommender.app_state import initialise_review_state, reset_review_history
from coffee_recommender.config import get_api_base_url
from coffee_recommender.streamlit_api_client import ApiClientError, StreamlitApiClient


st.set_page_config(page_title="Coffee Recommender", page_icon="☕", layout="wide")

st.title("Coffee Recommender")
st.write("Pick a coffee you reviewed, describe it, and inspect landscape-based recommendations.")

initialise_review_state(st.session_state)
client = StreamlitApiClient()

with st.sidebar:
    st.header("Backend")
    st.caption(f"API: `{get_api_base_url()}`")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    if st.button("Clear review history"):
        reset_review_history(st.session_state)

try:
    coffee_summaries = client.list_catalogue_coffees()
except ApiClientError as exc:
    st.error(str(exc))
    st.stop()

coffee_options = {
    f"{summary.name} ({summary.coffee_id})": summary.coffee_id
    for summary in coffee_summaries
}
catalogue_name_lookup = {summary.coffee_id: summary.name for summary in coffee_summaries}

input_mode = st.radio("Reviewed coffee source", options=["Catalogue coffee", "Coffee URL"], horizontal=True)
if input_mode != st.session_state.input_mode:
    st.session_state.input_mode = input_mode

selected_label = None
reviewed_coffee = None
reviewed_metadata = None
reviewed_sensory = None
selected_reviewed = None
url_value = st.session_state.url_reviewed_source

if input_mode == "Catalogue coffee":
    selected_label = st.selectbox("Reviewed coffee", options=list(coffee_options))
    try:
        selected_reviewed = client.get_catalogue_coffee(coffee_options[selected_label])
    except ApiClientError as exc:
        st.error(str(exc))
        st.stop()
else:
    url_value = st.text_input(
        "Coffee product URL",
        value=st.session_state.url_reviewed_source,
        placeholder="https://...",
    ).strip()
    process_url = st.button("Process coffee URL")

    if process_url:
        try:
            processed = client.process_reviewed_coffee_url(url_value)
            st.session_state.url_reviewed_coffee = processed.reviewed_coffee
            st.session_state.url_reviewed_source = processed.normalized_url
        except ApiClientError as exc:
            st.error(str(exc))

    if st.session_state.url_reviewed_coffee and url_value == st.session_state.url_reviewed_source:
        selected_reviewed = st.session_state.url_reviewed_coffee
    else:
        selected_reviewed = None

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
        if input_mode == "Coffee URL":
            if not url_value:
                raise ValueError("Paste a coffee product URL before running the recommender.")
            if st.session_state.url_reviewed_coffee is None or url_value != st.session_state.url_reviewed_source:
                processed = client.process_reviewed_coffee_url(url_value)
                st.session_state.url_reviewed_coffee = processed.reviewed_coffee
                st.session_state.url_reviewed_source = processed.normalized_url
            selected_reviewed = st.session_state.url_reviewed_coffee

        if selected_reviewed is None:
            raise ValueError("Select or process a reviewed coffee before running the recommender.")

        result = client.submit_review(
            SubmitReviewRequest(
                review_text=review,
                reviewed_coffee=selected_reviewed,
                top_k=top_k,
                review_session=st.session_state.review_session,
            )
        )
    except (ApiClientError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.session_state.review_session = result.review_session

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
    if st.session_state.review_session.last_event:
        st.json(st.session_state.review_session.last_event.model_dump(mode="json"))
    else:
        st.info("Run the recommender to see the parsed event.")

    st.subheader("Review History")
    if not st.session_state.review_session.review_events:
        st.info("Added reviews will appear here.")
    else:
        override_name_lookup = {
            coffee_id: payload.name
            for coffee_id, payload in st.session_state.review_session.reviewed_feature_overrides.items()
        }
        for index, event in enumerate(st.session_state.review_session.review_events, start=1):
            title = (
                catalogue_name_lookup.get(event.coffee_id)
                or override_name_lookup.get(event.coffee_id)
                or "Unknown coffee"
            )
            with st.expander(f"{index}. {title}"):
                st.caption(event.coffee_id)
                st.write(f"Overall: `{event.overall}`")
                st.write("Change requests")
                st.json(event.change_requests)
                st.write("Attribute opinions")
                st.json(event.attribute_opinions)

with col2:
    st.subheader("Recommendations")
    if not st.session_state.review_session.last_recommendations:
        st.info("Run the recommender to see ranked coffees.")
    else:
        for item in st.session_state.review_session.last_recommendations:
            with st.container(border=True):
                st.markdown(f"### {item.name}")
                st.caption(item.coffee_id)
                st.write(f"Score: `{item.score}`")
                st.write(f"Temperature: `{item.temperature}`")

                debug = item.debug
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
if not st.session_state.review_session.review_events:
    st.info("Add at least one review to plot the score landscape.")
else:
    try:
        landscape = client.build_landscape(
            LandscapeRequest(
                review_session=st.session_state.review_session,
                show_surface=show_landscape_surface,
            )
        )
    except ApiClientError as exc:
        st.error(str(exc))
        st.stop()

    if landscape.figure is None:
        st.info(landscape.message or "Need at least three coffees to project the score landscape.")
    else:
        st.plotly_chart(go.Figure(landscape.figure), use_container_width=True)
