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
     - `case_id` (string)
     - `embedding_vector` (list[float], 384-dim MiniLM)
     - `timestamp` (ISO 8601 string)
     - `label` (string: `PC` or `NOT_PC`)

2. **Configure the pipeline**
   - Edit `config.yaml` to set model type, balancing, output locations, and other parameters. All options are documented in the config file.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**
   ```bash
   python src/train_pipeline.py --data data/your_data.csv
   ```
   - Outputs (model, metrics, predictions, logs) will be saved to the locations specified in `config.yaml`.

5. **Check outputs**
   - Model: `models/pcc_model.joblib`
   - Metadata: `models/metadata.json`
   - Predictions: `output/predictions.parquet` or `.csv`
   - Metrics: `output/metrics.json`
   - Monitoring log: `logs/monitoring_log.jsonl`

---

## 📁 Project Structure

```
model-training-pipeline/
├── config.yaml                # All pipeline configuration
├── data/                      # Input data directory
├── notebooks/                 # EDA and exploratory notebooks
├── requirements.txt           # Python dependencies
├── scripts/
│   └── train_pipeline.py      # Main pipeline entry script
├── src/
│   └── model_training_pipeline/
│       ├── __init__.py
│       ├── data.py            # Data ingestion and validation
│       ├── evaluation.py      # Model evaluation and metrics
│       ├── persistence.py     # Model and output saving
│       ├── preprocessing.py   # Embedding, balancing, splitting
│       ├── training.py        # Model training and search
│       └── utils.py           # Config and logging utilities
├── tests/
│   ├── test_data_loader.py    # Data and preprocessing tests
│   └── test_utils.py          # Config and logger tests
├── models/                    # Saved models (output)
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

---

## 🛠️ Pipeline Stages

### Preprocessing
- SMOTE-based class balancing (configurable)
- Stratified train/test split
- Logging of class distributions and split sizes

### Model Training
- Logistic Regression (L1 penalty, C=10, random_state=42 by default, config-driven)
- Optional grid search and cross-validation
- Model and metadata saving

### Evaluation
- F1, ROC-AUC, PR-AUC, accuracy, confusion matrix
- Schema-compliant predictions and metrics outputs
- Monitoring log (JSONL)

---

## ⚙️ Configuration (`config.yaml`)

All options are documented in the config file. Key sections:
- `data`: Path, schema, stratification, test size
- `preprocessing`: Embedding model, balancing method
- `model`: Type, hyperparameters, grid search, cross-validation
- `output`: Directories, filenames, log paths
- `logging`: Level

---

## 🧪 Testing

Run all unit tests:
```bash
pytest
```

---

## ➡️ Next Steps
- Integrate with the next pipeline or system as needed
- Extend with new models or features as requirements evolve

---

**This pipeline is now ready for handoff and production use.**

For questions or further customization, see the code comments and config file, or contact the pipeline author. 