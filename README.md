# Pricing ML Engine

Real-world ML portfolio project focused on one question:

How do you turn a classification model into a pricing workflow that can be trained, promoted, served, monitored, and retrained without manual chaos?

This project predicts whether a health insurance customer is likely to buy vehicle insurance, then uses that probability to recommend a premium.

## The Business Problem

Insurance teams have limited outbound budget and need to decide:

- Which customers are worth targeting for cross-sell.
- How much premium to quote without leaving money on the table.
- When the model can no longer be trusted because live data has drifted.

This repo implements the complete loop, not just notebook experimentation.

## What I Built

- Data validation and preprocessing for tabular data (numeric + categorical).
- Model training pipeline with three candidates: RandomForest, XGBoost, LightGBM.
- Hyperparameter search with stratified CV and ROC-AUC selection.
- Model registry with timestamped runs and explicit promotion to production.
- FastAPI service for real-time scoring and premium recommendation.
- Batch live inference simulation, drift monitoring, and retrain decision logic.
- Automated tests for API, features, model loading, and registry behavior.

## Why These Choices

- ROC-AUC is the primary metric because the target is imbalanced (about 12% positive).
- Tree ensembles handle nonlinear interactions in tabular data with minimal feature hand-crafting.
- Stratified folds avoid unstable validation caused by class imbalance.
- Promotion is manual by design to keep deployment decisions auditable.
- PSI-based monitoring provides an interpretable drift signal for operations.

## Current Results (From Latest Production Run)

Run ID: 20260320T123908Z

| Model | Best CV ROC-AUC | Test ROC-AUC |
|---|---:|---:|
| RandomForest | 0.8543 | 0.8544 |
| LightGBM | 0.8576 | 0.8580 |
| XGBoost (promoted) | 0.8577 | 0.8583 |

Latest monitoring snapshot shows drift detection is working in practice:

- max feature PSI: 0.6802 (threshold: 0.2)
- retrain decision: triggered (dry-run mode)

## Repository Layout

```text
pricing_ml_engine/
├── api/                # FastAPI app
├── data/               # train/test and simulated live batches
├── models/             # registry and current promoted model
├── outputs/            # predictions, monitoring, retrain decisions
├── scripts/            # train, serve, monitor, retrain orchestration
├── src/                # core ML modules
├── tests/              # pytest suite
├── Makefile            # one-command workflows
└── config.yaml
```

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train candidates and register a run

```bash
make train
make list-models
```

3. Evaluate and promote a specific run

```bash
make evaluate RUN_ID=<run_id>
make promote RUN_ID=<run_id>
```

4. Start the API

```bash
make api
```

Deeploi dashboard: http://localhost:8000/
https://github.com/christianmcb/deeploi

## End-to-End Workflow

```bash
make simulate      # generate live batches
make live-infer    # score unprocessed batches
make monitor       # compute drift + performance summary
make retrain       # evaluate retrain triggers (dry-run)
```

To execute retraining automatically when triggered:

```bash
python -m scripts.retrain_if_needed --execute
```

## API Example

Request:

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

Response:

```json
{
  "conversion_probability": 0.08,
  "predicted_conversion": 0,
  "price_segment": "low-conversion",
  "base_premium": 300,
  "demand_adjustment": 16,
  "recommended_premium": 316
}
```

## Tests

```bash
make test
```

Coverage focus:

- input schema and feature pipeline behavior
- model loading and prediction validity
- API health and quote endpoint behavior
- registry and promotion flow

## What I Would Improve Next

- Add calibration checks (and optional calibrated probabilities) before pricing.
- Add CI pipeline to run tests and quality gates on every push.
- Add richer observability (request latency, model version tags, alerting hooks).
- Add temporal validation strategy if timestamped customer history becomes available.

## Author

Christian McBride  
Manchester, UK

GitHub: https://github.com/christianmcb  
LinkedIn: https://linkedin.com/in/christianmcb8

## License

MIT License