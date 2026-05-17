"""
Description:
This script trains and uses machine learning models for two related tasks:

1. Hotel occupancy prediction:
   - Loads cleaned hotel booking data.
   - Trains a Random Forest Classifier to predict whether a booking will result in realised occupancy.
   - Saves the trained hotel occupancy model.
   - Calculates classification metrics such as accuracy and classification report.

2. Energy load prediction:
   - Loads cleaned energy efficiency data.
   - Trains two Random Forest Regression models:
        - one for heating load prediction
        - one for cooling load prediction
   - Saves both trained energy models.
   - Calculates regression metrics such as MAE and R² score.    

3. Integrated prediction:
   - Loads all trained models.
   - Predicts hotel occupancy probability.
   - Predicts heating and cooling loads.
   - Adjusts predicted energy demand based on the occupancy probability.
   - Gives an energy management recommendation based on expected occupancy.

When this file is run directly, it trains all models and prints the evaluation metrics.
"""

# Used for creating folders and handling file paths
import os

# Used for saving and loading trained machine learning models
import joblib

# Used for working with tabular data
import pandas as pd

# Used for numerical operations
import numpy as np

# Used to split the dataset into training and testing parts
from sklearn.model_selection import train_test_split

# Used to convert categorical text values into numerical form
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Used to apply different preprocessing steps to different columns
from sklearn.compose import ColumnTransformer

# Used to combine preprocessing and model training into one workflow
from sklearn.pipeline import Pipeline

# Evaluation metrics for classification and regression models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)

# Machine learning models used for classification and regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Imported but not currently used in this script
from sklearn.linear_model import LinearRegression

# Custom data loading functions from the data preparation file
from data_prep import load_hotel_data, load_energy_data


# Folder where trained models and metrics will be saved
MODEL_DIR = "models"


def train_hotel_occupancy_model():
    """
    Train a hotel occupancy classification model.

    Returns:
        dict: Accuracy score and classification report for the hotel occupancy model.
    """

    # Create the models folder if it does not already exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load the cleaned hotel booking dataset
    df = load_hotel_data()

    # Separate input features from the target column
    X = df.drop(columns=["occupancy_demand"])

    # Target variable: 1 means booking was not cancelled, 0 means booking was cancelled
    y = df["occupancy_demand"]

    # Categorical column that needs one-hot encoding
    categorical_features = ["hotel"]

    # Numerical columns used directly by the model
    numerical_features = [
        "arrival_month_number",
        "lead_time",
        "total_guests",
        "total_stay_nights",
        "adr",
        "booking_changes",
        "required_car_parking_spaces",
        "total_of_special_requests",
    ]

    # Define preprocessing:
    # - Convert hotel type into numerical columns using OneHotEncoder
    # - Pass numerical columns without changing them
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Define the Random Forest classification model
    # class_weight="balanced" helps when classes are imbalanced
    model = RandomForestClassifier(
        n_estimators=120,     #120 Dt use 
        max_depth=10,          #each tree goes !0 depth
        random_state=42,
        class_weight="balanced",
    )

    # Create a pipeline that first preprocesses the data, then trains the model
    # preprocessor mean handle missing values convert strng int numeric split 
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Split data into training and testing sets
    # stratify=y keeps the same class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict occupancy demand on the test data
    preds = pipeline.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, preds)

    # Generate a detailed classification report as a dictionary
    report = classification_report(y_test, preds, output_dict=True)

    # Save the trained hotel occupancy model to a file
    joblib.dump(pipeline, f"{MODEL_DIR}/hotel_occupancy_model.pkl")

    # Return model evaluation results
    return {
        "hotel_accuracy": accuracy,
        "hotel_report": report,
    }


def train_energy_models():
    """
    Train heating load and cooling load regression models.

    Returns:
        dict: MAE and R² metrics for both heating and cooling models.
    """

    # Create the models folder if it does not already exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load the cleaned energy efficiency dataset
    df = load_energy_data()

    # Input features are all columns except heating and cooling targets
    X = df.drop(columns=["heating_load", "cooling_load"])

    # Target variable for heating load prediction
    y_heat = df["heating_load"]

    # Target variable for cooling load prediction
    y_cool = df["cooling_load"]

    # Split the data for heating model training and testing
    X_train, X_test, y_heat_train, y_heat_test = train_test_split(
        X,
        y_heat,
        test_size=0.2,
        random_state=42,
    )

    # Split the cooling target using the same random_state
    # This keeps the row split aligned with the heating split
    _, _, y_cool_train, y_cool_test = train_test_split(
        X,
        y_cool,
        test_size=0.2,
        random_state=42,
    )

    # StandardScaler is used to normalize numerical input features
    scaler = StandardScaler()

    # Pipeline for heating load prediction
    # First scales the features, then applies Random Forest Regression
    heat_model = Pipeline(
        steps=[
            ("scaler", scaler),
            ("model", RandomForestRegressor(n_estimators=120, random_state=42)),
        ]
    )

    # Pipeline for cooling load prediction
    # A separate scaler is used for this model
    cool_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=120, random_state=42)),
        ]
    )

    # Train the heating model
    heat_model.fit(X_train, y_heat_train)

    # Train the cooling model
    cool_model.fit(X_train, y_cool_train)

    # Predict heating load on test data
    heat_preds = heat_model.predict(X_test)

    # Predict cooling load on test data
    cool_preds = cool_model.predict(X_test)

    # Calculate Mean Absolute Error for heating prediction
    heat_mae = mean_absolute_error(y_heat_test, heat_preds)

    # Calculate Mean Absolute Error for cooling prediction
    cool_mae = mean_absolute_error(y_cool_test, cool_preds)

    # Calculate R² score for heating prediction
    heat_r2 = r2_score(y_heat_test, heat_preds)

    # Calculate R² score for cooling prediction
    cool_r2 = r2_score(y_cool_test, cool_preds)

    # Save the trained heating model
    joblib.dump(heat_model, f"{MODEL_DIR}/energy_heating_model.pkl")

    # Save the trained cooling model
    joblib.dump(cool_model, f"{MODEL_DIR}/energy_cooling_model.pkl")

    # Return model evaluation results
    return {
        "heating_mae": heat_mae,
        "cooling_mae": cool_mae,
        "heating_r2": heat_r2,
        "cooling_r2": cool_r2,
    }


def train_all_models():
    """
    Train all models and save their evaluation metrics.

    Returns:
        dict: Combined metrics from the hotel, heating, and cooling models.
    """

    # Train the hotel occupancy model
    hotel_metrics = train_hotel_occupancy_model()

    # Train the heating and cooling energy models
    energy_metrics = train_energy_models()

    # Combine all metrics into one dictionary
    all_metrics = {
        **hotel_metrics,
        **energy_metrics,
    }

    # Save all model metrics to a file
    joblib.dump(all_metrics, f"{MODEL_DIR}/model_metrics.pkl")

    # Return the combined metrics
    return all_metrics


def load_models():
    """
    Load all saved machine learning models.

    Returns:
        tuple: Hotel occupancy model, heating model, and cooling model.
    """

    # Load the trained hotel occupancy model
    hotel_model = joblib.load(f"{MODEL_DIR}/hotel_occupancy_model.pkl")

    # Load the trained heating load model
    heating_model = joblib.load(f"{MODEL_DIR}/energy_heating_model.pkl")

    # Load the trained cooling load model
    cooling_model = joblib.load(f"{MODEL_DIR}/energy_cooling_model.pkl")

    # Return all loaded models
    return hotel_model, heating_model, cooling_model

#Hotel booking inputs
def predict_integrated_energy(
    hotel_type,
    month_number,
    lead_time,
    total_guests,
    total_stay_nights,
    adr,
    booking_changes,
    parking_spaces,
    special_requests,
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    overall_height,
    orientation,
    glazing_area,
    glazing_area_distribution,
):
    """
    Predict occupancy and adjusted energy demand using trained models.

    Parameters:
        hotel_type: Type of hotel, such as "City Hotel" or "Resort Hotel".
        month_number: Arrival month number from 1 to 12.
        lead_time: Number of days between booking and arrival.
        total_guests: Total number of guests.
        total_stay_nights: Total number of nights stayed.
        adr: Average daily rate.
        booking_changes: Number of changes made to the booking.
        parking_spaces: Number of required parking spaces.
        special_requests: Number of special requests.
        relative_compactness: Building compactness value.
        surface_area: Building surface area.
        wall_area: Building wall area.
        roof_area: Building roof area.
        overall_height: Building height.
        orientation: Building orientation value.
        glazing_area: Window/glazing area.
        glazing_area_distribution: Distribution of glazing area.

    Returns:
        dict: Occupancy prediction, energy predictions, adjusted energy values,
              and energy usage recommendation.
    """

    # Load the saved hotel, heating, and cooling models
    hotel_model, heating_model, cooling_model = load_models()

    # Create one-row input data for the hotel occupancy model
    hotel_input = pd.DataFrame(
        [
            {
                "hotel": hotel_type,
                "arrival_month_number": month_number,
                "lead_time": lead_time,
                "total_guests": total_guests,
                "total_stay_nights": total_stay_nights,
                "adr": adr,
                "booking_changes": booking_changes,
                "required_car_parking_spaces": parking_spaces,
                "total_of_special_requests": special_requests,
            }
        ]
    )

    # Predict occupancy class:
    # 1 means occupancy is expected, 0 means occupancy is not expected
    occupancy_class = hotel_model.predict(hotel_input)[0]

    # Predict probability of realised occupancy
    occupancy_probability = hotel_model.predict_proba(hotel_input)[0][1]

    # Create one-row input data for the energy models
    energy_input = pd.DataFrame(
        [
            {
                "relative_compactness": relative_compactness,
                "surface_area": surface_area,
                "wall_area": wall_area,
                "roof_area": roof_area,
                "overall_height": overall_height,
                "orientation": orientation,
                "glazing_area": glazing_area,
                "glazing_area_distribution": glazing_area_distribution,
            }
        ]
    )

    # Predict heating load using the trained heating model
    predicted_heating = heating_model.predict(energy_input)[0]

    # Predict cooling load using the trained cooling model
    predicted_cooling = cooling_model.predict(energy_input)[0]

    # Adjust heating load according to occupancy probability
    adjusted_heating = predicted_heating * occupancy_probability

    # Adjust cooling load according to occupancy probability
    adjusted_cooling = predicted_cooling * occupancy_probability

    # Calculate total adjusted energy demand
    total_adjusted_energy = adjusted_heating + adjusted_cooling

    # Generate recommendation based on occupancy probability
    if occupancy_probability >= 0.75:
        recommendation = (
            "High occupancy expected. Increase energy readiness for heating and cooling systems."
        )
    elif occupancy_probability >= 0.45:
        recommendation = (
            "Moderate occupancy expected. Use balanced energy scheduling and monitor demand peaks."
        )
    else:
        recommendation = (
            "Low occupancy expected. Reduce unnecessary heating and cooling to save energy."
        )

    # Return all prediction results in dictionary form
    return {
        "occupancy_class": int(occupancy_class),
        "occupancy_probability": float(occupancy_probability),
        "predicted_heating_load": float(predicted_heating),
        "predicted_cooling_load": float(predicted_cooling),
        "adjusted_heating_load": float(adjusted_heating),
        "adjusted_cooling_load": float(adjusted_cooling),
        "total_adjusted_energy": float(total_adjusted_energy),
        "recommendation": recommendation,
    }


# This block only runs when this file is executed directly
if __name__ == "__main__":

    # Train all models and collect their evaluation metrics
    metrics = train_all_models()

    # Print the metrics after training is complete
    print(metrics)