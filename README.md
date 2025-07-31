---
noteId: "bf2c0cd0625611f084f08ddb7af7baaa"
tags: []

---

# Model Training Pipeline System

A modular, production-ready pipeline for training and versioning privacy intent classification models. Designed for integration with the PCC system and ready for plug-and-play use.

## Quick Start

### 1. Prepare Your Data
Place your data file (CSV, Parquet, or PKL) in the path specified by `data.path` in `config.yaml`. Ensure your data matches the required schema:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Input text for classification |
| `intent` | string | Classification label |
| `confidence` | float | Confidence score |
| `timestamp` | string | ISO 8601 timestamp |
| `text_length` | int | Character count |
| `word_count` | int | Word count |
| `has_personal_info` | boolean | Personal info flag |
| `formality_score` | float | Formality metric |
| `urgency_score` | float | Urgency metric |
| `embeddings` | list[float] | 584-dim embeddings (384 sentence + 200 TF-IDF) |

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python scripts/train_pipeline.py
```

**Outputs:**
- Model: `models/pcc_model.joblib` (local) + GCS backup
- Metadata: `models/metadata.json` + GCS backup  
- Predictions: `output/predictions.parquet`
- Metrics: `output/metrics.json`
- Registry: `models/model_registry.json`

## Performance Expectations

### Test Results
The pipeline achieves **perfect scores (1.0000 accuracy/F1)** on synthetic test data due to:
- Clear, non-overlapping intent patterns
- Rich 584-dimensional feature space
- Balanced class distribution

### Real-World Performance
With actual user data, expect **0.7-0.9 accuracy/F1** due to:
- Natural language ambiguity
- Class imbalance
- Text variation and noise
- Edge cases and typos

## Project Structure

```
model-training-pipeline/
├── config.yaml                # Pipeline configuration
├── data/                      # Input data directory
├── scripts/
│   ├── train_pipeline.py      # Main pipeline script
│   └── manage_models.py       # Model registry CLI
├── src/model_training_pipeline/
│   ├── data.py                # Data ingestion & validation
│   ├── preprocessing.py       # Embedding, balancing, splitting
│   ├── training.py            # Model training & optimization
│   ├── evaluation.py          # Model evaluation & metrics
│   ├── persistence.py         # Model & output saving
│   ├── model_registry.py      # Model tracking & registry
│   └── utils.py               # Config & logging utilities
├── tests/                     # Unit tests
├── models/                    # Saved models & registry
├── output/                    # Predictions & metrics
└── logs/                      # Monitoring logs
```

## Pipeline Stages

1. **Data Ingestion** → Load from CSV/Parquet/GCS
2. **Data Validation** → Schema & embedding validation
3. **Preprocessing** → SMOTE balancing + stratified split
4. **Model Training** → Logistic Regression + hyperparameter optimization
5. **Evaluation** → F1, Accuracy, ROC-AUC, PR-AUC metrics
6. **Persistence** → Save model, predictions, logs
7. **Registry** → Track runs, metadata, GCS paths

## Configuration

All pipeline behavior is controlled via `config.yaml`:

```yaml
# Key sections
data:
  path: "data/your_dataset.csv"
  test_size: 0.2

preprocessing:
  balancing: "smote"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

model:
  type: "LogisticRegression"
  search_method: "random"  # 'grid', 'random', 'none'
  n_iter: 20              # For random search
  cross_val: false
  scoring: "f1"

output:
  model_dir: "models/"
  predictions_file: "output/predictions.parquet"
  metrics_file: "output/metrics.json"
```

## Hyperparameter Optimization

### Available Methods

| Method | Speed | Effectiveness | Best For |
|--------|-------|---------------|----------|
| **Random Search** | Fast | Better | **Recommended** |
| Grid Search | Medium | Good | Small parameter spaces |
| No Search | Fastest | Basic | Quick testing |

### Random Search (Recommended)
```yaml
model:
  search_method: 'random'
  n_iter: 20
```

**Advantages:**
- Tests continuous parameter ranges (C: 0.01 to 1000)
- Often finds better hyperparameters than grid search
- Faster exploration of parameter space

### Grid Search
```yaml
model:
  search_method: 'grid'
```

**Advantages:**
- Exhaustive search of predefined combinations
- Reproducible results
- Good for small parameter spaces

## Model Registry Management

Use the CLI tool to manage models:

```bash
# List recent model runs
python scripts/manage_models.py list

# Show latest successful model
python scripts/manage_models.py latest

# Get performance summary
python scripts/manage_models.py summary

# Export registry to CSV
python scripts/manage_models.py export

# Get details for specific model version
python scripts/manage_models.py get --version v20250729084352
```

### Registry Features
- **Run Tracking**: Every pipeline run logged with full metadata
- **Performance History**: Track F1 scores, accuracy over time
- **GCS Integration**: Automatic tracking of model locations
- **Export Capabilities**: Export registry data for analysis
- **Failure Tracking**: Failed runs logged for debugging

## GCS Integration

Models are automatically saved to Google Cloud Storage:

```
gs://pcc-datasets/pcc-models/
├── v20250729084352/
│   ├── model.joblib
│   └── metadata.json
├── v20250729084415/
│   ├── model.joblib
│   └── metadata.json
└── ...
```

**Features:**
- Automatic versioning with timestamp-based versions
- Complete metadata storage alongside models
- Easy retrieval from GCS paths
- Local + GCS backup strategy

## Testing

Run all unit tests:
```bash
pytest
```

## Key Features

- **Config-driven**: All behavior controlled via `config.yaml`
- **Modular**: Each stage is a separate, swappable module
- **Schema-compliant**: Ready for downstream PCC integration
- **Secure**: Uses `ast.literal_eval()` for safe data parsing
- **Extensible**: Add new models/features with minimal changes
- **Unit tested**: Core modules covered by tests
- **Logging**: Comprehensive logging throughout pipeline

## Production Workflow

1. **New Dataset** → Place in `data/` directory
2. **Configure Search** → Set `search_method: 'random'` for best results
3. **Run Pipeline** → `python scripts/train_pipeline.py`
4. **Model Saved** → Automatically saved to GCS with versioning
5. **Registry Updated** → Run tracked with performance metrics
6. **Monitor Performance** → Use CLI tools to track over time

## Recent Enhancements

- **Schema Consistency**: Fixed column naming across modules
- **Security**: Replaced `eval()` with `ast.literal_eval()`
- **Dependencies**: Added missing packages (imbalanced-learn, PyYAML, google-cloud-storage)
- **Random Search**: Added for better hyperparameter optimization
- **GCS Integration**: Automatic model saving to Google Cloud Storage
- **Model Registry**: Complete tracking system for all runs
- **CLI Tools**: Management interface for model registry

## Future Enhancements

### High Priority
- **Enhanced Metrics**: Training/validation scores, per-class analysis
- **Bayesian Optimization**: Optuna integration for advanced hyperparameter tuning
- **Cross-Validation**: Detailed CV metrics across folds

### Medium Priority  
- **Automated Workflows**: GitHub Actions for CI/CD
- **Model Monitoring**: Drift detection and performance tracking
- **Data Quality**: Enhanced validation and profiling

### Low Priority
- **Advanced MLOps**: Docker, Kubernetes deployment
- **API Endpoints**: RESTful model serving
- **Security**: Encryption, access control, compliance

*Each enhancement can be implemented incrementally without breaking existing functionality.*

---

**This pipeline is ready for production use with full GCS integration, model tracking, and advanced hyperparameter optimization!**

For questions or customization, see the code comments and config file, or contact the pipeline author. 
