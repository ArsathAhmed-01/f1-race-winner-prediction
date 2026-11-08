import streamlit as st
import plotly.express as px
import predictor

st.set_page_config(
    page_title="F1 Race Winner Predictor",
    page_icon="🏎️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.95)),
        url("https://media.craiyon.com/2025-07-08/cYi7RJ1kRiu4Y-DgIxJFww.webp");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# LOAD ARTIFACTS (cached so this only runs once per session)
@st.cache_resource
def get_bundle():
    return predictor.ModelBundle()

bundle = get_bundle()

# QUALIFYING FETCHER (Streamlit-cached wrapper around predictor.load_qualifing)
@st.cache_data(show_spinner=False)
def load_qualifing_cached(season, race_name):
    return predictor.load_qualifying(season, race_name)

# UI
st.title("🏎️ F1 Race Winner Predictor")
st.markdown(
    """
    Predict the race winner using:
    - Qualifying position
    - Driver form
    - Constructor form
    - Circuit history
    - Reliability
    """
)

col1, col2 = st.columns(2)

with col1:
    season = st.number_input(
        "Select season",
        min_value=2018,
        max_value=2030,
        value=2026
    )

with col2:
    race_name = st.selectbox(
        "Race",
        list(predictor.RACE_TO_CIRCUIT.keys())
    )

# LOAD QUALIFYING
if st.button("Load Qualifying Results"):
    try:
        with st.spinner("Loading qualifying data..."):
            quali_df = load_qualifing_cached(season, race_name)
        st.success("Qualifying data loaded successfully!")
        st.dataframe(quali_df, use_container_width=True)
        st.session_state["quali_df"] = quali_df
    except Exception as e:
        st.error(f"Error loading qualifying data: {e}")

# PREDICT
if ("quali_df" in st.session_state and st.button("Predict Race Winner")):
    circuit_id = predictor.RACE_TO_CIRCUIT[race_name]
    feature_df = predictor.build_race_features(
        st.session_state["quali_df"],
        bundle.history_df,
        circuit_id,
        bundle.le_driver,
        bundle.le_constructor,
        bundle.le_circuit,
    )
    X_pred = feature_df[bundle.features]
    raw_probability = bundle.model.predict_proba(X_pred)[:, 1]
    # scale so the whole race adds up to 100% (only one driver can win)
    feature_df["win_probability"] = raw_probability / raw_probability.sum()

    predictions = feature_df[
        ["driver_name", "constructor_name", "grid_position", "win_probability"]
    ].sort_values(by="win_probability", ascending=False)

    st.subheader("Predicted Winning Probabilities")

    winner = predictions.iloc[0]
    st.success(f"🏆 Predicted winner: {winner['driver_name']} ({winner['constructor_name']})")
    st.metric("Win Probability", f"{winner['win_probability']:.2%}")

    st.subheader("Predicted Driver Rankings")
    st.dataframe(predictions, use_container_width=True)