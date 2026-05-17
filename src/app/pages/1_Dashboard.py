# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project Path Setup
# ---------------------------------------------------------

# Get the main project directory.
# __file__ refers to the current file.
# parents[3] moves three folders up from the current file location.
BASE_DIR = Path(__file__).resolve().parents[3]

# Define the source code directory
SRC_DIR = BASE_DIR / "src"

# Define the Streamlit app directory
APP_DIR = BASE_DIR / "src" / "app"

# Add src directory to Python path so files like data_prep.py can be imported
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Add app directory to Python path so files like utils.py can be imported
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

# ---------------------------------------------------------
# Local Module Imports
# ---------------------------------------------------------

# Import helper functions for loading and preparing data
from data_prep import load_hotel_data, load_energy_data, get_month_name

# Import UI utility functions for styling, sidebar, and footer
from utils import apply_base_styles, load_sidebar, inject_footer


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

# Configure the Streamlit page title, icon, and layout
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# Apply custom CSS styles
apply_base_styles()

# Load the sidebar navigation/content
load_sidebar()

# Main dashboard title
st.title("HotelEnergySense: Analytics Dashboard")


# ---------------------------------------------------------
# Load Datasets
# ---------------------------------------------------------

# Load hotel booking dataset from the data folder
hotel_df = load_hotel_data(str(BASE_DIR / "data" / "hotel_bookings.csv"))

# Load energy efficiency dataset from the data folder
energy_df = load_energy_data(str(BASE_DIR / "data" / "energy_efficiency.csv"))

# Convert numeric month values into readable month names
hotel_df["month_name"] = hotel_df["arrival_month_number"].apply(get_month_name)


# ---------------------------------------------------------
# Dashboard Summary Metrics
# ---------------------------------------------------------

# Create four columns to display key summary statistics
col1, col2, col3, col4 = st.columns(4)

# Display total number of hotel booking records
col1.metric("Hotel Records", f"{len(hotel_df):,}")

# Display total number of energy dataset records
col2.metric("Energy Records", f"{len(energy_df):,}")

# Display average number of guests per booking
col3.metric("Average Guests", f"{hotel_df['total_guests'].mean():.2f}")

# Display average heating load from the energy dataset
col4.metric("Avg Heating Load", f"{energy_df['heating_load'].mean():.2f}")

# Add a horizontal divider
st.divider()


# ---------------------------------------------------------
# Hotel Booking Demand Analysis
# ---------------------------------------------------------

# Section heading for hotel booking analysis
st.subheader("Hotel Booking Demand Analysis")

# Calculate average occupancy demand for each month
monthly_demand = (
    hotel_df.groupby("month_name")["occupancy_demand"]
    .mean()
    .reset_index()
)

# Define the correct order of months for sorting
month_order = [
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
]

# Convert month_name column into an ordered categorical column
# This ensures months appear in calendar order instead of alphabetical order
monthly_demand["month_name"] = pd.Categorical(
    monthly_demand["month_name"],
    categories=month_order,
    ordered=True,
)

# Sort the monthly demand data by month order
monthly_demand = monthly_demand.sort_values("month_name")

# Create a bar chart showing average occupancy demand by month
fig_month = px.bar(
    monthly_demand,
    x="month_name",
    y="occupancy_demand",
    title="Average Occupancy Demand by Month",
    labels={
        "month_name": "Month",
        "occupancy_demand": "Occupancy Demand Probability",
    },
    template="plotly_white",
)

# Display the monthly demand bar chart in Streamlit
st.plotly_chart(fig_month, use_container_width=True)

# Create two columns for side-by-side hotel analysis charts
col_a, col_b = st.columns(2)

# Left column: histogram of total guests
with col_a:
    # Create a histogram to show how many guests are included in bookings
    fig_guests = px.histogram(
        hotel_df,
        x="total_guests",
        nbins=10,
        title="Distribution of Total Guests",
        labels={"total_guests": "Total Guests"},
        template="plotly_white",
    )

    # Display the guest distribution chart
    st.plotly_chart(fig_guests, use_container_width=True)

# Right column: histogram of stay length
with col_b:
    # Create a histogram to show distribution of total stay nights
    fig_stay = px.histogram(
        hotel_df,
        x="total_stay_nights",
        nbins=20,
        title="Distribution of Stay Length",
        labels={"total_stay_nights": "Total Stay Nights"},
        template="plotly_white",
    )

    # Display the stay length distribution chart
    st.plotly_chart(fig_stay, use_container_width=True)

# Add a horizontal divider before the next section
st.divider()


# ---------------------------------------------------------
# Energy Efficiency Analysis
# ---------------------------------------------------------

# Section heading for energy efficiency analysis
st.subheader("Energy Efficiency Analysis")

# Create two columns for side-by-side energy analysis charts
col_c, col_d = st.columns(2)

# Left column: surface area vs heating load scatter plot
with col_c:
    # Create a scatter plot to analyze relationship between surface area and heating load
    # Color represents overall building height
    fig_heat = px.scatter(
        energy_df,
        x="surface_area",
        y="heating_load",
        color="overall_height",
        title="Surface Area vs Heating Load",
        labels={
            "surface_area": "Surface Area",
            "heating_load": "Heating Load",
            "overall_height": "Overall Height",
        },
        template="plotly_white",
    )

    # Display the heating load scatter plot
    st.plotly_chart(fig_heat, use_container_width=True)

# Right column: roof area vs cooling load scatter plot
with col_d:
    # Create a scatter plot to analyze relationship between roof area and cooling load
    # Color represents glazing area
    fig_cool = px.scatter(
        energy_df,
        x="roof_area",
        y="cooling_load",
        color="glazing_area",
        title="Roof Area vs Cooling Load",
        labels={
            "roof_area": "Roof Area",
            "cooling_load": "Cooling Load",
            "glazing_area": "Glazing Area",
        },
        template="plotly_white",
    )

    # Display the cooling load scatter plot
    st.plotly_chart(fig_cool, use_container_width=True)


# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

# Select important numerical columns from the energy dataset
# These columns are used to calculate correlations
corr_df = energy_df[
    [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "glazing_area",
        "heating_load",
        "cooling_load",
    ]
].corr()

# Create a heatmap to show correlations between energy-related features
fig_corr = px.imshow(
    corr_df,
    text_auto=True,
    title="Energy Dataset Correlation Heatmap",
    template="plotly_white",
)

# Display the correlation heatmap
st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

# Inject custom footer at the bottom of the dashboard
inject_footer()