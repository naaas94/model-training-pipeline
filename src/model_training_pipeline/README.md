# Model Training Pipeline

This module provides a modular, production-grade pipeline for training and evaluating models for the Privacy Case Classifier (PCC) system.

## Structure

- `data.py` — Data ingestion and validation
- `preprocessing.py` — Embedding generation, data balancing (SMOTE), and train/test split
- `training.py` — Model training (Logistic Regression, config-driven), hyperparameter search, and model persistence
- `evaluation.py` — Model evaluation (F1, ROC-AUC, PR-AUC, accuracy, confusion matrix), prediction output, and monitoring log
- `persistence.py` — Model saving, prediction output, and monitoring log writing
- `utils.py` — Config loading and logging utilities

## Input Data Format

The pipeline expects training data in CSV, Parquet, or PKL format with the following schema:

| Column           | Type         | Description                                 |
|------------------|--------------|---------------------------------------------|
| case_id          | string       | Unique identifier for each case             |
| embedding_vector | list[float]  | 384-dim MiniLM embedding vector             |
| timestamp        | timestamp    | ISO 8601 timestamp for the case             |
| label            | string       | Target label (PC, NOT_PC)                   |

**Sample Input Row:**
```json
{
  "case_id": "CASE_000001",
  "embedding_vector": [0.12, -0.34, ..., 0.56],
  "timestamp": "2025-01-01T00:00:00.000",
  "label": "PC"
}
```

- Embeddings are synthetic by default (N(0,1), 384 dims), but the pipeline is modular and can be extended to use real or text-derived embeddings.
- Invalid rows (missing fields, wrong shape/type, NaNs) are dropped and logged.

## Pipeline Stages

### Preprocessing
- **Class Balancing:** SMOTE-based balancing (configurable)
- **Train/Test Split:** Stratified by label (configurable)
- **Logging:** Class distributions and split sizes

### Model Training
- **Model:** Logistic Regression (L1 penalty, C=10, random_state=42 by default, config-driven)
- **Grid Search/CV:** Optional, config-driven
- **Persistence:** Model saved as .joblib, metadata as .json

### Evaluation
- **Metrics:** F1, ROC-AUC, PR-AUC, accuracy, confusion matrix
- **Predictions Output:** Schema-compliant DataFrame, saved as Parquet/CSV
- **Metrics Output:** JSON
- **Monitoring Log:** JSONL, schema-compliant

## Output Formats

- **Model:** `models/pcc_model.joblib`
- **Model Metadata:** `models/metadata.json`
- **Predictions:** `output/predictions.parquet` or `output/predictions.csv`
- **Metrics:** `output/metrics.json`
- **Monitoring Log:** `logs/monitoring_log.jsonl`

## Config Options (config.yaml)

See `config.yaml` for all configurable options, including:
- Data path, schema, stratification, test size
- Preprocessing: embedding model, balancing method
- Model: type, hyperparameters, grid search, cross-validation
- Output: directories, filenames, log paths
- Logging: level

## Testing

- Unit tests for data loading, preprocessing, training, and evaluation modules in `tests/`

## Example Usage

```bash
python src/train_pipeline.py --data data/curated_training_data.csv
```

## PCC Output Schema

| Field              | Type      | Description                                 |
|--------------------|-----------|---------------------------------------------|
| case_id            | string    | Unique identifier for each case             |
| predicted_label    | string    | Predicted class (e.g., PC, NOT_PC)          |
| subtype_label      | string    | Subtype (nullable, for future use)          |
| confidence         | float     | Model confidence score                      |
| model_version      | string    | Version of the model used                   |
| embedding_model    | string    | Name of the embedding model                 |
| inference_timestamp| timestamp | When the prediction was made                |
| prediction_notes   | string    | Additional notes (nullable)                 |
| ingestion_time     | timestamp | When the record was written                 |

## Monitoring Log Schema

| Field                        | Type      | Description                                 |
|------------------------------|-----------|---------------------------------------------|
| run_id                       | string    | Unique identifier for each pipeline run      |
| model_version                | string    | Version of the model used                   |
| embedding_model              | string    | Name of the embedding model                 |
| partition_date               | date      | Date partition being processed              |
| runtime_ts                   | timestamp | When the pipeline run started               |
| status                       | string    | Run status (success, failed, empty)         |
| total_cases                  | int       | Total number of cases processed             |
| passed_validation            | int       | Number of cases that passed validation      |
| dropped_cases                | int       | Number of cases dropped                     |
| notes                        | string    | Additional notes about the run              |
| ingestion_time               | timestamp | When the log record was written             |
| processing_duration_seconds  | float     | Total processing time in seconds            |
| error_message                | string    | Error details if the run failed (nullable)  | 