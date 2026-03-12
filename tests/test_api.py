from fastapi.testclient import TestClient
from api.pricing_api import app

client = TestClient(app)


def test_quote_endpoint_returns_prediction():
    payload = {
        "Gender": "Male",
        "Age": 44,
        "Driving_License": 1,
        "Region_Code": 28.0,
        "Previously_Insured": 0,
        "Vehicle_Age": "> 2 Years",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 35000.0,
        "Policy_Sales_Channel": 26.0,
        "Vintage": 217,
    }

    response = client.post("/quote", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "conversion_probability" in body
    assert "predicted_conversion" in body
    assert "recommended_premium" in body
