# Import Streamlit for building the web application UI
import streamlit as st

# Import sys so we can modify Python's module search path
import sys

# Import Path for cleaner and safer file/folder path handling
from pathlib import Path


# ---------------------------------------------------------
# Project Path Setup
# ---------------------------------------------------------

# BASE_DIR points to the main project folder.
# __file__ is the current file path.
# resolve() converts it to an absolute path.
# parents[3] moves three levels up from this file location.
BASE_DIR = Path(__file__).resolve().parents[3]

# SRC_DIR points to the src folder where main source code files are stored
SRC_DIR = BASE_DIR / "src"

# APP_DIR points to the Streamlit app folder
APP_DIR = BASE_DIR / "src" / "app"


# Add the src folder to Python's import path if it is not already there.
# This allows imports from files such as ml_workflows.py.
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Add the app folder to Python's import path if it is not already there.
# This allows imports from files such as utils.py.
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


# ---------------------------------------------------------
# Local Project Imports
# ---------------------------------------------------------

# Import the main prediction function that combines hotel occupancy
# and building energy prediction logic
from ml_workflows import predict_integrated_energy

# Import utility functions for applying CSS styles,
# loading the sidebar, and displaying the footer
from utils import apply_base_styles, load_sidebar, inject_footer


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

# Set Streamlit page settings such as browser title, icon, and layout
st.set_page_config(
    page_title="Energy Prediction",
    page_icon="⚡",
    layout="wide",
)


# Apply custom CSS styling to the page
apply_base_styles()

# Load the app sidebar, usually used for navigation or project information
load_sidebar()


# ---------------------------------------------------------
# Page Title and Description
# ---------------------------------------------------------

# Display the main title of the prediction page
st.title("HotelEnergySense: Demand & Energy Prediction")

# Display a short explanation of what the page does
st.write(
    """
    Enter hotel booking information and building energy parameters.
    The system will predict occupancy demand, estimate heating and cooling load,
    and calculate occupancy-adjusted energy consumption.
    """
)

# Add a horizontal divider to separate the introduction from the input form
st.divider()


# ---------------------------------------------------------
# Input Form Layout
# ---------------------------------------------------------

# Create two columns:
# col1 is used for hotel booking details
# col2 is used for building energy details
col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Hotel Booking Details Inputs
# ---------------------------------------------------------

with col1:
    # Section heading for hotel-related input fields
    st.subheader("Hotel Booking Details")

    # Dropdown for selecting the hotel type
    hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])

    # Dropdown for selecting arrival month using numbers 1 to 12
    # format_func displays the month names instead of plain numbers
    month_number = st.selectbox(
        "Arrival Month",
        list(range(1, 13)),
        format_func=lambda x: [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][x - 1],
    )

    # Input for lead time, meaning how many days before arrival the booking was made
    lead_time = st.number_input(
        "Lead Time",
        min_value=0,
        max_value=700,
        value=30,
    )

    # Input for total number of guests in the booking
    total_guests = st.number_input(
        "Number of Guests",
        min_value=1,
        max_value=10,
        value=2,
    )

    # Input for total number of nights the guest will stay
    total_stay_nights = st.number_input(
        "Length of Stay Nights",
        min_value=1,
        max_value=30,
        value=3,
    )

    # Input for Average Daily Rate, usually the average room price per night
    adr = st.number_input(
        "Average Daily Rate",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
    )

    # Input for how many changes were made to the booking
    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        max_value=20,
        value=0,
    )

    # Input for number of parking spaces required by the guest
    parking_spaces = st.number_input(
        "Required Parking Spaces",
        min_value=0,
        max_value=5,
        value=0,
    )

    # Input for number of special requests made by the guest
    special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        max_value=10,
        value=1,
    )


# ---------------------------------------------------------
# Building Energy Details Inputs
# ---------------------------------------------------------

with col2:
    # Section heading for building-related input fields
    st.subheader("Building Energy Details")

    # Input for relative compactness of the building
    # This describes how compact the building shape is
    relative_compactness = st.number_input(
        "Relative Compactness",
        min_value=0.5,
        max_value=1.0,
        value=0.76,
    )

    # Input for total surface area of the building
    surface_area = st.number_input(
        "Surface Area",
        min_value=400.0,
        max_value=900.0,
        value=650.0,
    )

    # Input for wall area of the building
    wall_area = st.number_input(
        "Wall Area",
        min_value=200.0,
        max_value=500.0,
        value=320.0,
    )

    # Input for roof area of the building
    roof_area = st.number_input(
        "Roof Area",
        min_value=100.0,
        max_value=400.0,
        value=220.0,
    )

    # Dropdown for building height
    overall_height = st.selectbox("Overall Height", [3.5, 7.0])

    # Dropdown for orientation value
    # This usually represents building direction/orientation category
    orientation = st.selectbox("Orientation", [2, 3, 4, 5])

    # Dropdown for glazing area, meaning window/glass area proportion
    glazing_area = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])

    # Dropdown for glazing area distribution category
    glazing_area_distribution = st.selectbox(
        "Glazing Area Distribution",
        [0, 1, 2, 3, 4, 5],
    )


# Add a divider before the prediction button
st.divider()


# ---------------------------------------------------------
# Prediction Button and Model Execution
# ---------------------------------------------------------

# When the user clicks the button, run the prediction workflow
if st.button("Predict Occupancy and Energy Consumption", type="primary"):
    try:
        # Call the integrated prediction function.
        # This function uses the hotel booking inputs and building energy inputs
        # to calculate occupancy probability, heating load, cooling load,
        # adjusted energy values, and a business recommendation.
        result = predict_integrated_energy(
            hotel_type=hotel_type,
            month_number=month_number,
            lead_time=lead_time,
            total_guests=total_guests,
            total_stay_nights=total_stay_nights,
            adr=adr,
            booking_changes=booking_changes,
            parking_spaces=parking_spaces,
            special_requests=special_requests,
            relative_compactness=relative_compactness,
            surface_area=surface_area,
            wall_area=wall_area,
            roof_area=roof_area,
            overall_height=overall_height,
            orientation=orientation,
            glazing_area=glazing_area,
            glazing_area_distribution=glazing_area_distribution,
        )

        # Display heading for prediction output
        st.subheader("Prediction Results")

        # Extract occupancy probability from the result dictionary
        occupancy_probability = result["occupancy_probability"]


        # ---------------------------------------------------------
        # Occupancy Risk Category Display
        # ---------------------------------------------------------

        # If occupancy probability is 75% or higher,
        # show a high occupancy warning box
        if occupancy_probability >= 0.75:
            st.markdown(
                f"""
                <div class="danger-box">
                    <strong>High Occupancy Expected</strong><br>
                    Occupancy Probability: {occupancy_probability * 100:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        # If occupancy probability is between 45% and 74.9%,
        # show a moderate occupancy warning box
        elif occupancy_probability >= 0.45:
            st.markdown(
                f"""
                <div class="warning-box">
                    <strong>Moderate Occupancy Expected</strong><br>
                    Occupancy Probability: {occupancy_probability * 100:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        # If occupancy probability is below 45%,
        # show a low occupancy success box
        else:
            st.markdown(
                f"""
                <div class="success-box">
                    <strong>Low Occupancy Expected</strong><br>
                    Occupancy Probability: {occupancy_probability * 100:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Add divider before showing energy metrics
        st.divider()


        # ---------------------------------------------------------
        # Energy Prediction Metrics
        # ---------------------------------------------------------

        # Create three columns for base heating, base cooling,
        # and total adjusted energy values
        c1, c2, c3 = st.columns(3)

        # Display predicted base heating load
        c1.metric("Base Heating Load", f"{result['predicted_heating_load']:.2f}")

        # Display predicted base cooling load
        c2.metric("Base Cooling Load", f"{result['predicted_cooling_load']:.2f}")

        # Display total energy after occupancy adjustment
        c3.metric("Total Adjusted Energy", f"{result['total_adjusted_energy']:.2f}")

        # Create two columns for adjusted heating and cooling loads
        c4, c5 = st.columns(2)

        # Display heating load adjusted by occupancy prediction
        c4.metric("Adjusted Heating Load", f"{result['adjusted_heating_load']:.2f}")

        # Display cooling load adjusted by occupancy prediction
        c5.metric("Adjusted Cooling Load", f"{result['adjusted_cooling_load']:.2f}")


        # ---------------------------------------------------------
        # Business Recommendation
        # ---------------------------------------------------------

        # Display a business recommendation generated by the prediction workflow
        st.subheader("Business Recommendation")
        st.info(result["recommendation"])


    # ---------------------------------------------------------
    # Error Handling
    # ---------------------------------------------------------

    # This error appears if required trained model files are missing
    except FileNotFoundError:
        st.error(
            "Model files were not found. Please run `python src/ml_workflows.py` first to train the models."
        )

    # This catches any other unexpected prediction error
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

# Add the custom footer at the bottom of the Streamlit page
inject_footer()