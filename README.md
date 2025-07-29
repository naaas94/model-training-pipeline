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
   # From project root (recommended)
   python scripts/train_pipeline.py
   
   # Or from scripts directory
   cd scripts
   python train_pipeline.py
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

## **🔧 Pipeline Stages**

1. **Data Ingestion**: Load data from CSV/Parquet files or Google Cloud Storage
2. **Data Validation**: Schema validation, embedding vector validation (584-dimensional: 384 sentence + 200 TF-IDF)
3. **Preprocessing**: 
   - **Balancing**: SMOTE for class imbalance
   - **Splitting**: Stratified train-test split
4. **Model Training**: Logistic Regression with hyperparameter optimization
5. **Evaluation**: F1, Accuracy, ROC-AUC, PR-AUC metrics
6. **Persistence**: Save model, predictions, and monitoring logs
7. **Model Registry**: Track all runs, metadata, and GCS paths

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

## 🚀 Advanced Hyperparameter Optimization (Future Enhancements)

While the current pipeline uses grid search and random search for simplicity, there are more advanced methods that could be implemented for even better hyperparameter optimization:

### **Bayesian Optimization**
```python
# Example with Optuna
import optuna

def objective(trial):
    C = trial.suggest_float('C', 0.01, 1000, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    # ... model training and evaluation
```

**Advantages:**
- Uses previous trial results to guide next trials
- Often finds optimal hyperparameters in fewer trials
- More efficient than random search
- Can handle complex parameter spaces

**Why we're not using it:**
- Requires additional dependencies (Optuna, scikit-optimize)
- More complex to understand and debug
- Overkill for simple LogisticRegression models

### **Tree-structured Parzen Estimators (TPE)**
```python
# Example with Hyperopt
from hyperopt import fmin, tpe, hp

space = {
    'C': hp.loguniform('C', np.log(0.01), np.log(1000)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'solver': hp.choice('solver', ['liblinear', 'saga'])
}
```

**Advantages:**
- Very efficient for expensive evaluations
- Good for deep learning and complex models
- Adaptive sampling based on previous results

**Why we're not using it:**
- Complex implementation
- Requires Hyperopt dependency
- Better suited for expensive model training

### **Sequential Model-Based Optimization (SMBO)**
```python
# Example with scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Categorical

space = [
    Real(0.01, 1000, name='C', prior='log-uniform'),
    Categorical(['l1', 'l2'], name='penalty'),
    Categorical(['liblinear', 'saga'], name='solver')
]
```

**Advantages:**
- Gaussian Process-based optimization
- Good for continuous and categorical parameters
- Efficient exploration-exploitation balance

**Why we're not using it:**
- Requires scikit-optimize dependency
- More complex than needed for our use case
- Better for expensive objective functions

### **Comparison of All Methods:**

| Method | Efficiency | Complexity | Dependencies | Best For |
|--------|------------|------------|--------------|----------|
| **Random Search** | Medium | Low | None | ✅ **Our Choice** |
| Grid Search | Low | Low | None | Small spaces |
| Bayesian (Optuna) | High | Medium | Optuna | Complex models |
| TPE (Hyperopt) | High | High | Hyperopt | Expensive training |
| SMBO (skopt) | High | Medium | scikit-optimize | Continuous params |

### **When to Consider Advanced Methods:**

**Stick with Random Search if:**
- ✅ Simple models (LogisticRegression, RandomForest)
- ✅ Quick iteration cycles
- ✅ Team prefers simplicity
- ✅ Limited computational resources

**Consider Advanced Methods if:**
- 🔄 Training deep learning models
- 🔄 Expensive model training (>1 hour per trial)
- 🔄 Complex parameter spaces (>10 parameters)
- 🔄 Need maximum performance optimization
- 🔄 Have expertise in advanced optimization

### **Future Implementation Path:**

If you later want to add advanced optimization:

1. **Start with Optuna** (easiest to implement)
2. **Add as optional dependency** (don't break existing functionality)
3. **Keep random search as default** (for simplicity)
4. **Add configuration option** for advanced users

```yaml
# Future config option
model:
  search_method: 'bayesian'  # 'random', 'grid', 'bayesian', 'tpe'
  optimization_library: 'optuna'  # 'sklearn', 'optuna', 'hyperopt'
```

**For now, random search provides excellent results with minimal complexity!** 🎯

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

## **🚀 Next Steps & Future Enhancements**

### **📊 Enhanced Scoring & Metrics**
- **Training Set Scores**: Add metrics computation on training set after model fitting
- **Validation Set Scores**: Implement train/val/test splits for proper validation
- **Cross-Validation Scores**: Enable and log detailed CV metrics across folds
- **Per-Class Metrics**: Expand per-class precision, recall, and confusion matrix analysis
- **Model Interpretability**: Add feature importance analysis and SHAP values

### **🔧 Advanced Hyperparameter Optimization**
Currently using Grid Search and Random Search. Future implementations:

| Method | Library | Pros | Cons | When to Use |
|--------|---------|------|------|-------------|
| **Bayesian Optimization** | Optuna | Efficient, handles continuous params | Complex setup | Limited compute budget |
| **Tree-structured Parzen Estimators (TPE)** | Hyperopt | Good for categorical params | Slower convergence | Mixed parameter types |
| **Sequential Model-Based Optimization (SMBO)** | scikit-optimize | Robust, good defaults | Less flexible | General purpose |

**Implementation Path:**
```python
# Example: Bayesian Optimization with Optuna
import optuna

def objective(trial):
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
    }
    # Train and evaluate model
    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### **🤖 Automated Pipeline Workflows**

#### **GitHub Actions Workflow**
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline Automation

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  check-new-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for new dataset
        run: |
          TODAY=$(date +%Y%m%d)
          if [ -f "src/data/balanced_dataset_${TODAY}.csv" ]; then
            echo "NEW_DATA=true" >> $GITHUB_ENV
          fi
      
      - name: Run ML Pipeline
        if: env.NEW_DATA == 'true'
        run: |
          python scripts/train_pipeline.py
          
      - name: Deploy Model
        if: env.NEW_DATA == 'true'
        run: |
          # Deploy to production
          # Update model registry
          # Send notifications
```

#### **Local Automation Script**
```python
# scripts/auto_pipeline.py
import os
from datetime import datetime
from scripts.train_pipeline import main as run_pipeline

def check_and_run():
    today = datetime.now().strftime('%Y%m%d')
    data_path = f'src/data/balanced_dataset_{today}.csv'
    
    if os.path.exists(data_path):
        print(f"New data found: {data_path}")
        run_pipeline()
        # Send notification
    else:
        print(f"No new data for {today}")

if __name__ == '__main__':
    check_and_run()
```

### **📈 Model Monitoring & MLOps**
- **Model Performance Tracking**: Monitor model drift and performance degradation
- **A/B Testing Framework**: Compare model versions in production
- **Alerting System**: Notify on model performance issues
- **Model Versioning**: Enhanced version control with semantic versioning
- **Experiment Tracking**: Integration with MLflow or Weights & Biases

### **🔍 Data Quality & Validation**
- **Data Drift Detection**: Monitor changes in data distribution
- **Schema Validation**: Enhanced data validation with Pydantic
- **Data Quality Metrics**: Completeness, consistency, accuracy scores
- **Automated Data Profiling**: Generate data quality reports

### **⚡ Performance Optimizations**
- **Parallel Processing**: Multi-core data processing
- **GPU Acceleration**: CUDA support for large datasets
- **Caching Layer**: Redis/Memcached for intermediate results
- **Streaming Processing**: Handle large datasets without loading into memory

### **🔐 Security & Compliance**
- **Data Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based permissions for model access
- **Audit Logging**: Comprehensive logging for compliance
- **Data Privacy**: GDPR/CCPA compliance features

### **🌐 Production Deployment**
- **Docker Containerization**: Containerized pipeline deployment
- **Kubernetes Orchestration**: Scalable model serving
- **API Endpoints**: RESTful API for model predictions
- **Load Balancing**: High-availability model serving

### **📊 Advanced Analytics**
- **Model Interpretability**: SHAP, LIME integration
- **Feature Engineering Pipeline**: Automated feature selection
- **Ensemble Methods**: Stacking, voting, bagging
- **Time Series Analysis**: Temporal pattern recognition

### **🔄 Continuous Integration/Deployment**
- **Automated Testing**: Unit, integration, and performance tests
- **Model Validation**: Pre-deployment model checks
- **Rollback Mechanisms**: Quick model version rollback
- **Blue-Green Deployment**: Zero-downtime model updates

---

## **🎯 Implementation Priority**

1. **High Priority**: Training/Validation scores, Bayesian optimization
2. **Medium Priority**: Automated workflows, model monitoring
3. **Low Priority**: Advanced MLOps, production deployment

*Each enhancement can be implemented incrementally without breaking existing functionality.* 