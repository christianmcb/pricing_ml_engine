import pytest
import pandas as pd
from src.data_processing import validate_inference_dataframe
from src.feature_engineering import build_preprocessor


def test_preprocessor_fits_on_sample_data():
    df = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Age": [44, 32],
        "Driving_License": [1, 1],
        "Region_Code": [28.0, 3.0],
        "Previously_Insured": [0, 1],
        "Vehicle_Age": ["> 2 Years", "1-2 Year"],
        "Vehicle_Damage": ["Yes", "No"],
        "Annual_Premium": [40454.0, 33536.0],
        "Policy_Sales_Channel": [26.0, 152.0],
        "Vintage": [217, 183],
    })

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 2


def test_validate_inference_dataframe_drops_id_and_preserves_required_columns():
    df = pd.DataFrame({
        "id": [1],
        "Gender": ["Male"],
        "Age": [44],
        "Driving_License": [1],
        "Region_Code": [28.0],
        "Previously_Insured": [0],
        "Vehicle_Age": ["> 2 Years"],
        "Vehicle_Damage": ["Yes"],
        "Annual_Premium": [40454.0],
        "Policy_Sales_Channel": [26.0],
        "Vintage": [217],
    })

    validated = validate_inference_dataframe(df)

    assert "id" not in validated.columns
    assert validated.shape[1] == 10


def test_validate_inference_dataframe_raises_on_missing_columns():
    df = pd.DataFrame({
        "Gender": ["Male"],
        "Age": [44],
    })

    with pytest.raises(ValueError):
        validate_inference_dataframe(df)