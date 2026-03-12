from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import load_config
from src.data_processing import validate_inference_dataframe


config = load_config()
model = joblib.load(config["artifacts"]["model_path"])

app = FastAPI(title="Pricing ML Engine API")


class PricingRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=120)
    Driving_License: int = Field(..., ge=0, le=1)
    Region_Code: float
    Previously_Insured: int = Field(..., ge=0, le=1)
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(..., ge=0)
    Policy_Sales_Channel: float
    Vintage: int = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/quote")
def quote(request: PricingRequest):
    input_df = pd.DataFrame([request.model_dump()])
    input_df = validate_inference_dataframe(input_df)

    conversion_probability = float(model.predict_proba(input_df)[:, 1][0])
    predicted_conversion = int(model.predict(input_df)[0])

    base_premium = config["pricing"]["base_premium"]
    demand_multiplier = config["pricing"]["demand_multiplier"]

    demand_adjustment = conversion_probability * demand_multiplier
    recommended_premium = round(base_premium + demand_adjustment, 2)

    if conversion_probability >= 0.75:
        price_segment = "high-conversion"
    elif conversion_probability >= 0.40:
        price_segment = "mid-conversion"
    else:
        price_segment = "low-conversion"

    return {
        "conversion_probability": round(conversion_probability, 4),
        "predicted_conversion": predicted_conversion,
        "base_premium": base_premium,
        "demand_adjustment": round(demand_adjustment, 2),
        "recommended_premium": recommended_premium,
        "price_segment": price_segment,
    }
