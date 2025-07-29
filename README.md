---
noteId: "bf2c0cd0625611f084f08ddb7af7baaa"
tags: []

---

# Model Training Pipeline System

A modular, production-ready pipeline for training and versioning privacy intent classification models. Designed for integration with the PCC system and ready for plug-and-play use.

---

## 🚀 Plug and Play Usage

1. **Prepare your data**
   - Place your data file (CSV, Parquet, or PKL) in the path specified by `data.path` in `config.yaml`.
   - Ensure your data matches the required schema:
     - `text` (string)
     - `intent` (string: classification label)
     - `confidence` (float)
     - `timestamp` (ISO 8601 string)
     - `text_length` (int)
     - `word_count` (int)
     - `has_personal_info` (boolean)
     - `formality_score` (float)
     - `urgency_score` (float)
     - `embeddings` (list[float], 384-dim MiniLM)

2. **Configure the pipeline**
   - Edit `config.yaml` to set model type, balancing, output locations, and other parameters. All options are documented in the config file.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**
   ```bash
   python scripts/train_pipeline.py
   ```
   - Outputs (model, metrics, predictions, logs) will be saved to the locations specified in `config.yaml`.
   - **Models are automatically saved to GCS**: `gs://pcc-datasets/pcc-models/`
   - **All runs are tracked in the model registry**

5. **Check outputs**
   - Model: `models/pcc_model.joblib` (local) + `gs://pcc-datasets/pcc-models/{version}/model.joblib` (GCS)
   - Metadata: `models/metadata.json` (local) + `gs://pcc-datasets/pcc-models/{version}/metadata.json` (GCS)
   - Predictions: `output/predictions.parquet`
   - Metrics: `output/metrics.json`
   - Monitoring log: `logs/monitoring_log.jsonl`
   - Model Registry: `models/model_registry.json`

---

## 📁 Project Structure

```
model-training-pipeline/
├── config.yaml                # All pipeline configuration
├── data/                      # Input data directory
├── notebooks/                 # EDA and exploratory notebooks
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── train_pipeline.py      # Main pipeline entry script
│   └── manage_models.py       # CLI tool for model registry
├── src/
│   └── model_training_pipeline/
│       ├── __init__.py
│       ├── data.py            # Data ingestion and validation
│       ├── evaluation.py      # Model evaluation and metrics
│       ├── model_registry.py  # Model tracking and registry
│       ├── persistence.py     # Model and output saving (GCS)
│       ├── preprocessing.py   # Embedding, balancing, splitting
│       ├── training.py        # Model training and search
│       └── utils.py           # Config and logging utilities
├── tests/
│   ├── test_data_loader.py    # Data and preprocessing tests
│   └── test_utils.py          # Config and logger tests
├── models/                    # Saved models and registry (output)
├── output/                    # Predictions, metrics, etc. (output)
├── logs/                      # Monitoring logs (output)
└── README.md                  # System documentation (this file)
```

---

## ✨ Features

- **Config-driven**: All pipeline behavior is controlled via `config.yaml`.
- **Modular**: Each stage (preprocessing, training, evaluation) is a separate, swappable module.
- **Schema-compliant outputs**: Ready for downstream PCC integration.
- **Logging**: All steps log actions, errors, and warnings.
- **Unit tested**: Core modules are covered by tests in `tests/`.
- **Extensible**: Add new models, features, or preprocessing steps with minimal changes.
- **Secure**: Uses `ast.literal_eval()` instead of `eval()` for safe data parsing.
- **GCS Integration**: Automatic model saving to Google Cloud Storage.
- **Model Registry**: Complete tracking of all model runs and performance metrics.
- **Hyperparameter Optimization**: Grid search and random search support.

---

## 🛠️ Pipeline Stages

### Data Ingestion & Validation
- Loads data from CSV, Parquet, or PKL files
- Validates embedding vectors (384-dimensional)
- Drops invalid rows and logs issues

### Preprocessing
- SMOTE-based class balancing (configurable)
- Stratified train/test split
- Logging of class distributions and split sizes

### Model Training
- Logistic Regression with hyperparameter optimization
- **Grid Search**: Tests predefined parameter combinations
- **Random Search**: Tests continuous parameter ranges (often more effective)
- Model and metadata saving (local + GCS)

### Evaluation
- F1, ROC-AUC, PR-AUC, accuracy, confusion matrix
- Schema-compliant predictions and metrics outputs
- Monitoring log (JSONL)

### Model Registry
- Tracks all training runs with metadata
- Performance metrics over time
- GCS path tracking
- Export capabilities for analysis

---

## 🔧 Hyperparameter Optimization

The pipeline supports multiple hyperparameter search strategies:

### **Random Search (Recommended)**
```yaml
model:
  search_method: 'random'  # Default - often finds better hyperparameters
  n_iter: 20              # Number of random trials
```

**Advantages:**
- Tests continuous parameter ranges (C: 0.01 to 1000)
- Often finds better hyperparameters than grid search
- Faster than grid search for large parameter spaces
- More exploration of the parameter space

### **Grid Search**
```yaml
model:
  search_method: 'grid'    # Tests specific parameter combinations
```

**Advantages:**
- Exhaustive search of predefined combinations
- Reproducible results
- Good for small parameter spaces

### **No Search (Fixed Parameters)**
```yaml
model:
  search_method: 'none'    # Uses fixed hyperparameters
```

### **Search Comparison:**

| Method | C Range | Penalty | Solver | Trials | Speed | Effectiveness |
|--------|---------|---------|--------|--------|-------|---------------|
| Grid | [0.1, 1, 10, 100] | [l1, l2] | [liblinear, saga] | 16 | Medium | Good |
| Random | [0.01, 1000] | [l1, l2] | [liblinear, saga] | 20 | Fast | Better |
| None | Fixed | Fixed | Fixed | 1 | Fastest | Basic |

---

## 🔧 Model Registry Management

The pipeline includes a comprehensive model registry that tracks all training runs. Use the CLI tool to manage models:

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

### Registry Features:
- **Run Tracking**: Every pipeline run is logged with full metadata
- **Performance History**: Track F1 scores, accuracy, and other metrics over time
- **GCS Integration**: Automatic tracking of model locations in GCS
- **Export Capabilities**: Export registry data for analysis
- **Failure Tracking**: Failed runs are also logged for debugging

---

## ☁️ GCS Integration

Models are automatically saved to Google Cloud Storage with the following structure:

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

### GCS Features:
- **Automatic Versioning**: Each model gets a unique timestamp-based version
- **Metadata Storage**: Complete model metadata stored alongside models
- **Easy Retrieval**: Models can be loaded directly from GCS paths
- **Backup Strategy**: Local + GCS storage for redundancy

---

## ⚙️ Configuration (`config.yaml`)

All options are documented in the config file. Key sections:
- `data`: Path, schema, stratification, test size
- `preprocessing`: Embedding model, balancing method
- `model`: Type, hyperparameters, search method, cross-validation
- `output`: Directories, filenames, log paths
- `logging`: Level

### **Hyperparameter Search Options:**
```yaml
model:
  type: 'LogisticRegression'
  hyperparameters:
    penalty: 'l1'
    C: 10
    solver: 'liblinear'
    random_state: 42
  search_method: 'random'  # 'grid', 'random', 'none'
  n_iter: 20              # Number of iterations for random search
  cross_val: false
  cv_folds: 5
  scoring: 'f1'
```

---

## 🧪 Testing

Run all unit tests:
```bash
pytest
```

---

## 🔧 Recent Fixes & Enhancements

- **Schema Consistency**: Fixed column naming inconsistencies across all modules
- **Security**: Replaced `eval()` with `ast.literal_eval()` for safe data parsing
- **Dependencies**: Added missing packages (imbalanced-learn, PyYAML, google-cloud-storage)
- **Grid Search**: Fixed parameter grid structure for proper hyperparameter tuning
- **Random Search**: Added random search for better hyperparameter optimization
- **ROC AUC**: Improved handling of binary vs multiclass classification
- **Pipeline Completion**: Implemented full end-to-end pipeline flow
- **Error Handling**: Added proper validation and error handling throughout
- **GCS Integration**: Automatic model saving to Google Cloud Storage
- **Model Registry**: Complete tracking system for all model runs
- **CLI Tools**: Management interface for model registry

---

## 🚀 Production Workflow

1. **New Dataset Available**: Place data in `data/` directory
2. **Configure Search**: Set `search_method: 'random'` for best results
3. **Run Pipeline**: `python scripts/train_pipeline.py`
4. **Model Saved**: Automatically saved to GCS with versioning
5. **Registry Updated**: Run tracked with performance metrics
6. **Monitor Performance**: Use CLI tools to track model performance over time

**This pipeline is now ready for production use with full GCS integration, model tracking, and advanced hyperparameter optimization!**

For questions or further customization, see the code comments and config file, or contact the pipeline author. 