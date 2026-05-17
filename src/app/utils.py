import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def apply_base_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8fafc;
        }

        h1, h2, h3 {
            color: #1e3a8a;
        }

        section[data-testid="stSidebar"] {
            background-color: #eef2ff;
        }

        .metric-card {
            padding: 20px;
            background: white;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }

        .success-box {
            padding: 18px;
            background-color: #dcfce7;
            color: #166534;
            border-radius: 12px;
            border: 1px solid #86efac;
        }

        .warning-box {
            padding: 18px;
            background-color: #fef3c7;
            color: #92400e;
            border-radius: 12px;
            border: 1px solid #fcd34d;
        }

        .danger-box {
            padding: 18px;
            background-color: #fee2e2;
            color: #991b1b;
            border-radius: 12px;
            border: 1px solid #fca5a5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_sidebar():
    st.sidebar.title("HotelEnergySense")

    st.sidebar.markdown("---")
    st.sidebar.page_link("app.py", label="Home & Instructions")
    st.sidebar.page_link("pages/1_Dashboard.py", label="Analytics Dashboard")
    st.sidebar.page_link("pages/2_Prediction.py", label="Energy Prediction")
    st.sidebar.page_link("pages/3_Model_Evaluation.py", label="Model Evaluation")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Engine Config")
    model_choice = st.sidebar.selectbox(
        "Active AI Model",
        ["Random Forest", "Linear Regression"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This prototype links hotel booking demand prediction with building energy-load estimation."
    )

    return model_choice


def inject_footer():
    st.markdown("---")
    st.caption("Designed & Developed for Data Science Product Development")