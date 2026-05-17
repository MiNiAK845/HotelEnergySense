# Import Streamlit to build the web application interface
import streamlit as st

# Import joblib to load saved machine learning model metrics from .pkl files
import joblib

# Import pandas to convert model reports into DataFrames
import pandas as pd

# Import sys to modify Python's import path
import sys

# Import Path for clean and reliable file path handling
from pathlib import Path


# ---------------------------------------------------------
# Project Path Setup
# ---------------------------------------------------------

# BASE_DIR points to the main project directory.
# __file__ gives the path of the current Python file.
# resolve() converts it into an absolute path.
# parents[3] moves three folders up from the current file location.
BASE_DIR = Path(__file__).resolve().parents[3]

# SRC_DIR points to the main source code folder
SRC_DIR = BASE_DIR / "src"

# APP_DIR points to the Streamlit app folder
APP_DIR = BASE_DIR / "src" / "app"


# Add the src folder to Python's import path if it is not already added.
# This allows Python to import files located inside the src directory.
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Add the app folder to Python's import path if it is not already added.
# This allows Python to import files located inside src/app.
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


# ---------------------------------------------------------
# Local Project Imports
# ---------------------------------------------------------

# Import helper functions for styling, sidebar navigation, and footer display
from utils import apply_base_styles, load_sidebar, inject_footer


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

# Configure browser tab title, page icon, and page layout
st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🧪",
    layout="wide",
)


# ---------------------------------------------------------
# Apply App Styling and Sidebar
# ---------------------------------------------------------

# Apply custom CSS styles used across the app
apply_base_styles()

# Load the sidebar, usually used for navigation or project information
load_sidebar()


# ---------------------------------------------------------
# Page Title
# ---------------------------------------------------------

# Display the main page title
st.title("Algorithmic Integrity & Model Evaluation")


# ---------------------------------------------------------
# Load Saved Model Metrics
# ---------------------------------------------------------

# Define the file path where model evaluation metrics are saved
metrics_path = BASE_DIR / "models" / "model_metrics.pkl"

# Check whether the metrics file exists.
# If it does not exist, show an error and stop the Streamlit app.
if not metrics_path.exists():
    st.error(
        "Model metrics not found. Please train models first using `python src/ml_workflows.py`."
    )

    # Stop the app execution so the rest of the page does not run without metrics
    st.stop()

# Load the saved model metrics dictionary from the .pkl file
metrics = joblib.load(metrics_path)


# ---------------------------------------------------------
# Hotel Occupancy Demand Model Evaluation
# ---------------------------------------------------------

# Display section heading for the hotel occupancy model
st.subheader("Hotel Occupancy Demand Model")

# Create two columns:
# col1 shows the accuracy metric
# col2 explains what the hotel model does
col1, col2 = st.columns(2)

with col1:
    # Display hotel occupancy model accuracy as a percentage
    st.metric(
        "Hotel Occupancy Accuracy",
        f"{metrics['hotel_accuracy'] * 100:.2f}%",
    )

with col2:
    # Display a short explanation of the hotel occupancy classification model
    st.write(
        """
        The hotel model is a classification model that predicts whether a booking is likely
        to result in realised occupancy. This is used as the linking variable for energy
        adjustment.
        """
    )


# ---------------------------------------------------------
# Classification Report Table
# ---------------------------------------------------------

# Extract the hotel model classification report from the metrics dictionary
hotel_report = metrics["hotel_report"]

# Convert the classification report dictionary into a pandas DataFrame.
# transpose() turns classes/summary rows into table rows for easier display.
report_df = pd.DataFrame(hotel_report).transpose()

# Display the classification report as an interactive table in Streamlit
st.dataframe(report_df, use_container_width=True)

# Add a divider before the next model evaluation section
st.divider()


# ---------------------------------------------------------
# Energy Load Regression Model Evaluation
# ---------------------------------------------------------

# Display section heading for heating and cooling regression models
st.subheader("Energy Load Regression Models")

# Create four columns for regression evaluation metrics
col3, col4, col5, col6 = st.columns(4)

# Display Mean Absolute Error for the heating load model
col3.metric("Heating MAE", f"{metrics['heating_mae']:.2f}")

# Display R-squared score for the heating load model
col4.metric("Heating R²", f"{metrics['heating_r2']:.2f}")

# Display Mean Absolute Error for the cooling load model
col5.metric("Cooling MAE", f"{metrics['cooling_mae']:.2f}")

# Display R-squared score for the cooling load model
col6.metric("Cooling R²", f"{metrics['cooling_r2']:.2f}")

# Explain the purpose of the heating and cooling regression models
st.write(
    """
    The energy prediction models estimate heating and cooling loads using building parameters.
    The final system does not directly merge the hotel and energy datasets. Instead, the hotel
    occupancy prediction is used as a logical adjustment factor for base energy consumption.
    """
)

# Add divider before the integration explanation section
st.divider()


# ---------------------------------------------------------
# Model Integration Explanation
# ---------------------------------------------------------

# Display section heading for integration logic
st.subheader("Model Integration Explanation")

# Explain how hotel occupancy probability is used to adjust energy predictions
st.markdown(
    """
    The product uses the following integration logic:

    `Adjusted Heating Load = Predicted Heating Load × Occupancy Probability`

    `Adjusted Cooling Load = Predicted Cooling Load × Occupancy Probability`

    `Total Adjusted Energy = Adjusted Heating Load + Adjusted Cooling Load`

    This means the building energy estimate increases when predicted hotel occupancy is high
    and decreases when expected occupancy is low.
    """
)


# ---------------------------------------------------------
# Limitations Section
# ---------------------------------------------------------

# Display section heading for project limitations
st.subheader("Limitations")

# List important limitations of the current prototype
st.write(
    """
    - The hotel booking dataset and energy efficiency dataset are separate datasets, so integration is simulation-based.
    - Occupancy is estimated from booking demand rather than direct room sensor data.
    - Real hotels may require extra variables such as weather, HVAC type, room count, and energy tariffs.
    - The prototype demonstrates decision-support logic rather than a fully deployed enterprise system.
    """
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

# Add the custom footer at the bottom of the page
inject_footer()