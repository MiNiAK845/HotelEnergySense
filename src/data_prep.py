"""
Description:
This script contains helper functions for loading and preparing two datasets:

1. Hotel booking dataset:
   - Reads hotel booking data from a CSV file.
   - Handles missing values in important columns.
   - Converts month names into month numbers.
   - Creates new useful columns such as:
        - total_guests
        - total_stay_nights
        - occupancy_demand
   - Removes invalid records where guests or stay nights are zero.
   - Selects only the relevant columns required for further analysis or modeling.

2. Energy efficiency dataset:
   - Reads building energy efficiency data from a CSV file.
   - Renames coded column names such as X1, X2, Y1, and Y2 into meaningful names.
   - Selects important input and output columns.
   - Removes missing values.

The script also includes a helper function to convert a month number back into its month name.
"""

import pandas as pd
import numpy as np


# Dictionary used to convert month names into month numbers
MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def load_hotel_data(path="data/hotel_bookings.csv"):
    """
    Load and clean the hotel booking dataset.

    Parameters:
        path (str): File path of the hotel booking CSV file.

    Returns:
        pandas.DataFrame: Cleaned hotel booking data with selected columns.
    """

    # Read the hotel booking CSV file into a DataFrame
    df = pd.read_csv(path)

    # Replace missing values in the children column with 0
    df["children"] = df["children"].fillna(0)

    # Replace missing country values with "Unknown"
    df["country"] = df["country"].fillna("Unknown")

    # Replace missing agent values with 0 if the column exists
    # If the column does not exist, create it with value 0
    df["agent"] = df["agent"].fillna(0) if "agent" in df.columns else 0

    # Replace missing company values with 0 if the column exists
    # If the column does not exist, create it with value 0
    df["company"] = df["company"].fillna(0) if "company" in df.columns else 0

    # Convert arrival month name into month number
    df["arrival_month_number"] = df["arrival_date_month"].map(MONTH_MAP)

    # Calculate the total number of guests in each booking
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Calculate the total number of stay nights
    df["total_stay_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )

    # Remove records where total guests are zero
    df = df[df["total_guests"] > 0]

    # Remove records where total stay nights are zero
    df = df[df["total_stay_nights"] > 0]

    # Create an occupancy demand column
    # If booking is not canceled, occupancy_demand = 1
    # If booking is canceled, occupancy_demand = 0
    df["occupancy_demand"] = 1 - df["is_canceled"]

    # Select only the columns needed for analysis or machine learning
    selected_cols = [
        "hotel",
        "arrival_month_number",
        "lead_time",
        "total_guests",
        "total_stay_nights",
        "adr",
        "booking_changes",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "occupancy_demand",
    ]

    # Keep selected columns only and remove rows with missing values //////
    df = df[selected_cols].dropna()

    # Return the cleaned hotel booking DataFrame
    return df


def load_energy_data(path="data/energy_efficiency.csv"):
    """
    Load and clean the energy efficiency dataset.

    Parameters:
        path (str): File path of the energy efficiency CSV file.

    Returns:
        pandas.DataFrame: Cleaned energy efficiency data with renamed columns.
    """

    # Read the energy efficiency CSV file into a DataFrame
    df = pd.read_csv(path)

    # Rename original dataset column names into meaningful names
    rename_map = {
        "X1": "relative_compactness",
        "X2": "surface_area",
        "X3": "wall_area",
        "X4": "roof_area",
        "X5": "overall_height",
        "X6": "orientation",
        "X7": "glazing_area",
        "X8": "glazing_area_distribution",
        "Y1": "heating_load",
        "Y2": "cooling_load",
    }

    # Apply the column renaming
    df = df.rename(columns=rename_map)

    # Select only the columns required for analysis or modeling
    selected_cols = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_area_distribution",
        "heating_load",
        "cooling_load",
    ]

    # Keep selected columns only and remove rows with missing values
    df = df[selected_cols].dropna()

    # Return the cleaned energy efficiency DataFrame
    return df


def get_month_name(month_number):
    """
    Convert a month number into its month name.

    Parameters:
        month_number (int): Month number from 1 to 12.

    Returns:
        str: Month name if the number is valid, otherwise "Unknown".
    """

    # Create a reverse dictionary from MONTH_MAP
    # Example: 1 becomes "January"
    reverse_map = {v: k for k, v in MONTH_MAP.items()}

    # Return the month name if found, otherwise return "Unknown"
    return reverse_map.get(month_number, "Unknown")