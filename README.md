# Pricing ML Engine

A production-style machine learning project demonstrating how models can be **trained, evaluated, versioned, and deployed via an API**.

The system predicts whether an existing **health insurance customer will purchase vehicle insurance** and uses this signal to generate a simple pricing recommendation.

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

Typical positive rate in the dataset is around **12%**, making this a moderately imbalanced classification problem.

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
│
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

# Quickstart

Clone the repository:

```bash
git clone <repo-url>
cd pricing-ml-engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
make train
```

List trained runs:

```bash
make list-models
```

Evaluate a candidate model:

```bash
make evaluate RUN_ID=<RUN_ID>
```

Promote a model for production:

```bash
make promote RUN_ID=<RUN_ID>
```

Start the API:

```bash
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
2. Validate data schema
3. Build preprocessing pipeline
4. Train multiple models
5. Tune hyperparameters
6. Evaluate performance
7. Save versioned artifacts

Supported models:

- RandomForest
- XGBoost
- LightGBM

Training command:

```bash
make train
```

Each run generates a **timestamped run ID** and stores artifacts in the **model registry**.

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

This allows:

- experiment reproducibility
- historical comparison of models
- safe deployment decisions

---

# Model Evaluation

Evaluate a trained model before deployment.

List available runs:

```bash
make list-models
```

Evaluate a specific run:

```bash
make evaluate RUN_ID=<RUN_ID>
```

Evaluation metrics include:

- ROC AUC
- test set performance
- model comparison across algorithms

---

# Model Promotion

Only evaluated models should be promoted.

Promote a model to production:

```bash
make promote RUN_ID=<RUN_ID>
```

This copies the selected model to:

```
models/current/model.joblib
```

All prediction services load from this location.

---

# Batch Prediction

Generate predictions on new data:

```bash
make predict
```

or run directly:

```bash
python -m scripts.predict
```

Predictions typically include:

```
conversion_probability
predicted_conversion
base_premium
demand_adjustment
recommended_premium
```

---

# Pricing Logic

The ML model predicts the probability that a customer will purchase vehicle insurance.

Example:

```
conversion_probability = 0.08
```

A simple pricing rule converts this into a premium recommendation.

Example calculation:

```
base_premium = 300
demand_adjustment = conversion_probability × price_factor
recommended_premium = base_premium + demand_adjustment
```

This demonstrates how **machine learning outputs can feed into pricing logic**, similar to demand-based pricing systems.

---

# API

Start the API:

```bash
make api
```

or directly:

```bash
uvicorn scripts.serve_api:app --reload
```

Open the interactive documentation:

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

# Docker Deployment

Build the container:

```bash
make docker-build
```

Run the container:

```bash
make docker-run
```

The API will be available at:

```
http://localhost:8000
```

---

# Testing

Run unit tests:

```bash
make test
```

or:

```bash
pytest tests/
```

---

# Typical Workflow

```
make train
make list-models
make evaluate RUN_ID=<RUN_ID>
make promote RUN_ID=<RUN_ID>
make api
```

This workflow demonstrates a simplified **ML training → evaluation → promotion → deployment pipeline**, similar to those used in production ML systems.

---

# Technologies Used

- Python
- Scikit-learn
- XGBoost
- LightGBM
- FastAPI
- Docker
- Pandas
- NumPy
- TQDM
```