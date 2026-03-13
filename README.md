# Pricing ML Engine

A production-style machine learning project demonstrating how models can be **trained, evaluated, versioned, and deployed via an API**.

The model predicts whether an existing **health insurance customer will purchase vehicle insurance** and uses this signal to generate a simple pricing recommendation.

---

# Dataset

This project uses the **Health Insurance Cross-Sell Prediction dataset**, which contains demographic and insurance-related information about customers with existing health insurance policies.

The objective is to predict whether a customer would be interested in purchasing **vehicle insurance**.

Dataset source:

https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Reference:

> Kumar, A. (2021). *Health Insurance Cross Sell Prediction Dataset*. Kaggle.

Target variable:

```
Response
```

Where:

```
1 → Customer interested in vehicle insurance  
0 → Customer not interested  
```

The positive rate in the dataset is roughly **12%**, making this a moderately imbalanced classification task.

---

## 🚀 Live API Demo

The trained model is deployed as a **FastAPI inference service** and can be queried directly.

Interactive API documentation:

https://YOUR-RAILWAY-URL/docs


## Example Results

Model performance using 5-fold cross-validation on the training set and evaluation on a held-out test set.

| Model | CV ROC-AUC (Train) | Test ROC-AUC | Notes |
|------|------|------|------|
| RandomForest | 0.8543 | 0.8544 | Strong baseline |
| LightGBM | 0.8576 | 0.8580 | Captures nonlinear interactions |
| XGBoost | 0.8577 | 0.8583 | Best overall performance |

**Selected model for deployment:** XGBoost  

---

# Project Structure

```
pricing-ml-engine/
│
├── data/
│   └── train.csv
│
├── models/
│   ├── registry/
│   │   └── <RUN_ID>/
│   │       ├── model.joblib
│   │       ├── model_metadata.json
│   │       ├── model_comparison.csv
│   │       ├── best_params.json
│   │       └── feature_importance.csv
│   │
│   └── current/
│       ├── model.joblib
│       └── model_metadata.json
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── serve_api.py
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── evaluate_model.py
│   └── logger.py
│
├── tests/
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

# Quickstart

Clone the repository, install dependencies, train the model, and start the API:

```bash
git clone <repo-url>
cd pricing_ml_engine
pip install -r requirements.txt

make train
make list-models
make evaluate RUN_ID=<RUN_ID>
make promote RUN_ID=<RUN_ID>

make api
```

API documentation:

```
http://localhost:8000/docs
```

---

# Training Pipeline

Training runs a full ML workflow:

1. Load dataset  
2. Validate input schema  
3. Build preprocessing pipeline  
4. Train multiple candidate models  
5. Tune hyperparameters  
6. Evaluate performance  
7. Save versioned artifacts  

Supported models:

- RandomForest  
- XGBoost  
- LightGBM  

Each training run produces a **timestamped model version** stored in:

```
models/registry/<RUN_ID>/
```

Example:

```
models/registry/20260312T193312Z/
```

Artifacts saved per run:

```
model.joblib
model_metadata.json
model_comparison.csv
best_params.json
feature_importance.csv
```

This allows experiment reproducibility and safe deployment decisions.

---

# Model Evaluation & Promotion

Models are evaluated before deployment and promoted manually.

Example workflow:

```bash
make list-models
make evaluate RUN_ID=<RUN_ID>
make promote RUN_ID=<RUN_ID>
```

The promoted model becomes the **active production model**:

```
models/current/model.joblib
```

All prediction services load from this location.

---

# API

Start the FastAPI inference service:

```bash
make api
```

Interactive API documentation:

```
http://localhost:8000/docs
```

Example request:

```json
{
  "Gender": "Male",
  "Age": 44,
  "Driving_License": 1,
  "Region_Code": 28.0,
  "Previously_Insured": 0,
  "Vehicle_Age": "> 2 Years",
  "Vehicle_Damage": "Yes",
  "Annual_Premium": 35000.0,
  "Policy_Sales_Channel": 26.0,
  "Vintage": 217
}
```

Example response:

```json
{
  "conversion_probability": 0.08,
  "predicted_conversion": 0,
  "base_premium": 300,
  "demand_adjustment": 16,
  "recommended_premium": 316
}
```

---

# Docker

Build and run the API inside a container:

```bash
make docker-build
make docker-run
```

The API will be available at:

```
http://localhost:8000/docs
```

---

# Typical Workflow

```
train → evaluate → promote → serve
```

Commands:

```bash
make train
make list-models
make evaluate RUN_ID=<RUN_ID>
make promote RUN_ID=<RUN_ID>
make api
```

---

## Author

Christian McBride  
Manchester, UK  

GitHub: https://github.com/christianmcb  
LinkedIn: https://linkedin.com/in/christianmcb8

---

## License

This project is licensed under the MIT License.