import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import apply_base_styles, load_sidebar, inject_footer


st.set_page_config(
    page_title="HotelEnergySense",
    page_icon="🏨",
    layout="wide",
)

apply_base_styles()
active_model = load_sidebar()

st.markdown(
    """
    <div style="margin-bottom: 30px;">
        <h1 style="font-size: 3.5rem; font-weight: 900; color: #1e3a8a;">
            HotelEnergySense
        </h1>
        <h2 style="font-size: 1.5rem; font-weight: 400; color: #64748b;">
            Energy Efficiency Optimization using Hotel Booking Demand Prediction
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Project Overview")

    st.write(
        """
        **HotelEnergySense** is a data science product prototype designed to help hotel managers
        forecast customer demand and estimate energy consumption more intelligently.

        The system uses two predictive components:

        1. A **hotel booking demand model** that estimates occupancy probability.
        2. An **energy efficiency model** that predicts heating and cooling loads.

        The two models are then logically connected using predicted occupancy level. This allows
        the system to estimate adjusted energy consumption based on expected hotel usage.
        """
    )

    st.markdown(
        """
        **Integrated Formula:**

        `Adjusted Energy Consumption = Predicted Energy Load × Occupancy Probability`
        """
    )

with col_right:
    st.container(border=True)
    st.markdown("### Product Details")
    st.write("**Domain:** Hospitality and Energy Management")
    st.write("**Users:** Hotel managers, operations teams, sustainability officers")
    st.write("**Platform:** Streamlit Web App")
    st.write(f"**Active Model:** {active_model}")

st.divider()

st.subheader("How to Use HotelEnergySense")

with st.expander("Step 1: Review Analytics Dashboard", expanded=True):
    st.write(
        """
        Use the Analytics Dashboard to explore booking patterns, guest demand, seasonal
        trends, and energy-load relationships.
        """
    )

with st.expander("Step 2: Enter Hotel and Building Details"):
    st.write(
        """
        Navigate to the Energy Prediction page. Enter booking information such as month,
        guests, stay length, and room price. Then enter building parameters such as surface
        area, wall area, roof area, height, and glazing area.
        """
    )

with st.expander("Step 3: Generate Prediction"):
    st.write(
        """
        The system predicts the likelihood of hotel occupancy and estimates heating and
        cooling loads. These outputs are combined to calculate adjusted energy consumption.
        """
    )

with st.expander("Step 4: Review Model Evaluation"):
    st.write(
        """
        The Model Evaluation page shows model performance, including classification accuracy,
        regression errors, and interpretation of results.
        """
    )

st.divider()

st.subheader("Project Development Journey")

timeline_data = pd.DataFrame(
    [
        dict(Task="Dataset Selection & Research", Start="2026-03-01", Finish="2026-03-10", Phase="Planning"),
        dict(Task="Data Cleaning & Preprocessing", Start="2026-03-11", Finish="2026-03-24", Phase="Data Engineering"),
        dict(Task="Exploratory Data Analysis", Start="2026-03-25", Finish="2026-04-07", Phase="Analysis"),
        dict(Task="Hotel Demand Model Development", Start="2026-04-08", Finish="2026-04-20", Phase="AI Development"),
        dict(Task="Energy Load Model Development", Start="2026-04-21", Finish="2026-05-02", Phase="AI Development"),
        dict(Task="Model Integration", Start="2026-05-03", Finish="2026-05-12", Phase="Backend"),
        dict(Task="Streamlit UI Development", Start="2026-05-13", Finish="2026-05-24", Phase="Frontend"),
        dict(Task="Testing & Technical Report", Start="2026-05-25", Finish="2026-05-31", Phase="Submission"),
    ]
)

fig = px.timeline(
    timeline_data,
    x_start="Start",
    x_end="Finish",
    y="Task",
    color="Phase",
    template="plotly_white",
)

fig.update_yaxes(autorange="reversed")
fig.update_layout(height=430)

st.plotly_chart(fig, use_container_width=True)

inject_footer()