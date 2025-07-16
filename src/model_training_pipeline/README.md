# Model Training Pipeline

This module provides a modular, production-grade pipeline for training and evaluating models for the Privacy Case Classifier (PCC) system.

## Structure

- `data.py` — Data ingestion and validation
- `preprocessing.py` — Embedding generation and data balancing
- `training.py` — Model training and hyperparameter search
- `evaluation.py` — Model evaluation and visualization
- `persistence.py` — Model saving, prediction output, and monitoring log writing
- `utils.py` — Config loading and logging utilities

## Input Data Format

The pipeline expects training data in CSV, Parquet, or PKL format with the following schema:

| Column           | Type         | Description                                 |
|------------------|--------------|---------------------------------------------|
| case_id          | string       | Unique identifier for each case             |
| embedding_vector | list[float]  | 384-dim MiniLM embedding vector             |
| timestamp        | timestamp    | ISO 8601 timestamp for the case             |

**Sample Input Row:**
```json
{
  "case_id": "CASE_000001",
  "embedding_vector": [0.12, -0.34, ..., 0.56],
  "timestamp": "2025-01-01T00:00:00.000"
}
```

- Embeddings are synthetic by default (N(0,1), 384 dims), but the pipeline is modular and can be extended to use real or text-derived embeddings.
- Invalid rows (missing fields, wrong shape/type, NaNs) are dropped and logged.

## Output Prediction Schema

Predictions are saved in Parquet or CSV with the following fields (matching PCC output schema):

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

Monitoring logs are written as JSON lines with the following fields:

| Field                     | Type      | Description                                 |
|---------------------------|-----------|---------------------------------------------|
| run_id                    | string    | Unique identifier for each pipeline run      |
| model_version             | string    | Version of the model used                   |
| embedding_model           | string    | Name of the embedding model                 |
| partition_date            | date      | Date partition being processed              |
| runtime_ts                | timestamp | When the pipeline run started               |
| status                    | string    | Run status (success, failed, empty)         |
| total_cases               | int       | Total number of cases processed             |
| passed_validation         | int       | Number of cases that passed validation      |
| dropped_cases             | int       | Number of cases dropped                     |
| notes                     | string    | Additional notes about the run              |
| ingestion_time            | timestamp | When the log record was written             |
| processing_duration_seconds| float    | Total processing time in seconds            |
| error_message             | string    | Error details if the run failed (nullable)  |

## Embedding Assumptions
- Default: Synthetic, sampled from N(0,1), 384 dimensions (MiniLM-compatible).
- Modular: The pipeline can be extended to generate embeddings from synthetic or real text.
- All embedding parameters and generation methods should be logged for reproducibility.

## Modularity & Configurability
- All pipeline parameters (data path, embedding model, model hyperparameters, etc.) are set in `config.yaml`.
- Embedding generation/validation is modular and can be swapped as needed.
- Output and monitoring schemas are enforced for compatibility with the PCC system.

## Usage

Run the training pipeline from the project root:

```bash
python scripts/train_pipeline.py
```

## Next Steps
- Implement each module step-by-step
- Add unit and integration tests
- Integrate with BigQuery and production data sources
- Ensure compatibility with the PCC inference pipeline 