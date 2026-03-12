# Pricing ML Engine

A production-style machine learning project demonstrating how pricing and customer response models can be trained, evaluated, versioned, and deployed into a real-time quote generation service.

The system predicts the probability that a customer will respond to an insurance offer and converts this signal into a pricing recommendation that could be used inside a quoting or pricing engine.

This repository is designed to showcase **production-oriented ML engineering practices**, including:

- reproducible training pipelines
- model versioning and artifact management
- evaluation gates before deployment
- controlled model promotion
- API-based inference
- containerised deployment

---

# Quick Setup (Recommended)

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd pricing-ml-engine
pip install -r requirements.txt
```

Train the model:

```bash
make train
```

List trained model runs:

```bash
make list-models
```

Evaluate a candidate model:

```bash
make evaluate RUN_ID=<RUN_ID>
```

Promote the model to production:

```bash
make promote RUN_ID=<RUN_ID>
```

Start the API:

```bash
make api
```

API documentation will be available at:

```
http://localhost:8000/docs
```

---

# Example Full Workflow

```bash
make train
make list-models
make evaluate RUN_ID=<RUN_ID>
make promote RUN_ID=<RUN_ID>
make api
```

Pipeline steps:

1. Train candidate models
2. Evaluate performance
3. Promote approved model
4. Serve model via API

---

# Project Structure

```
pricing-ml-engine/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── evaluate_model.py
│   └── logger.py
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── serve_api.py
│
├── tests/
│   └── test_features.py
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
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

# ML Pipeline Architecture

This project separates **training**, **evaluation**, and **serving**.

Models are first trained and stored in a **model registry**, then evaluated and promoted to production.

```
Train → Evaluate → Promote → Serve
```

This architecture mirrors common **MLOps workflows** used in production ML systems.

---

# Model Registry

Each training run creates a timestamped model version:

```
models/registry/<RUN_ID>/
```

Example:

```
models/registry/20260312T193312Z/
```

Artifacts stored per run:

```
model.joblib
model_metadata.json
model_comparison.csv
best_params.json
feature_importance.csv
```

This allows:

- reproducible experiments
- historical model comparison
- safe model deployment decisions

---

# Business Objective

Insurance pricing and quote optimisation systems must balance:

- customer demand
- conversion likelihood
- risk-based pricing

This project models **customer conversion probability** and converts it into a **pricing recommendation**.

Example workflow:

1. Customer requests a quote
2. Model predicts probability of conversion
3. Pricing logic adjusts premium
4. Recommended premium returned

Applications include:

- insurance quote optimisation
- demand-based pricing
- price sensitivity modelling
- customer propensity modelling

---

# Dataset

The project uses a public insurance dataset containing customer and policy information.

Example features:

| Feature | Description |
|------|------|
| Gender | Customer gender |
| Age | Customer age |
| Driving_License | Whether the customer has a driving license |
| Region_Code | Geographic region |
| Previously_Insured | Whether the customer already has insurance |
| Vehicle_Age | Age of vehicle |
| Vehicle_Damage | Prior vehicle damage |
| Policy_Sales_Channel | Sales channel |
| Vintage | Days since customer association |

Target variable:

**Response**

Indicates whether the customer accepted the insurance offer.

---

# Training Models

The training pipeline performs:

1. Load dataset
2. Build preprocessing pipeline
3. Train multiple candidate models
4. Tune hyperparameters
5. Evaluate models
6. Compare performance
7. Save versioned artifacts

Supported models:

- RandomForest
- XGBoost
- LightGBM

Run training:

```bash
make train
```

---

# Batch Prediction

Run predictions on new CSV data:

```bash
make predict
```

or

```bash
python -m scripts.predict
```

Outputs include:

- conversion_probability
- predicted_conversion
- risk_loading
- recommended_premium

---

# Running the API

Start the quote generation API:

```bash
make api
```

or

```bash
uvicorn scripts.serve_api:app --reload
```

API documentation:

```
http://localhost:8000/docs
```

---

# Example Quote Request

POST `/quote`

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

---

# Example Quote Response

```json
{
  "conversion_probability": 0.81,
  "predicted_conversion": 1,
  "base_premium": 300.0,
  "demand_adjustment": 162.46,
  "recommended_premium": 462.46,
  "price_segment": "high-conversion"
}
```

---

# Testing

Run unit tests:

```bash
make test
```

or

```bash
pytest tests/
```

---

# Docker Deployment

Build container:

```bash
make docker-build
```

Run container:

```bash
make docker-run
```

API available at:

```
http://localhost:8000
```

---

# Technologies Used

Python  
Scikit-learn  
XGBoost  
LightGBM  
FastAPI  
Docker  
Pandas  
NumPy  
TQDM  

---

# Future Improvements

Potential extensions:

- MLflow experiment tracking
- Feature store integration
- Model monitoring and drift detection
- CI/CD pipeline for automated training
- Cloud deployment (AWS / GCP)
- Online feature pipelines

---

# License

MIT License