import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training_pipeline.data import load_data
from model_training_pipeline.preprocessing import balance_data, stratified_train_test_split
from model_training_pipeline.training import train_model
from model_training_pipeline.evaluation import evaluate_model

def test_load_data(tmp_path):
    # Create a sample DataFrame with valid and invalid rows
    valid_row = {
        'case_id': 'CASE_001',
        'embedding_vector': list(np.random.randn(384)),
        'timestamp': '2025-01-01T00:00:00.000'
    }
    invalid_row_shape = {
        'case_id': 'CASE_002',
        'embedding_vector': list(np.random.randn(100)),  # wrong shape
        'timestamp': '2025-01-01T01:00:00.000'
    }
    invalid_row_nan = {
        'case_id': 'CASE_003',
        'embedding_vector': [np.nan]*384,
        'timestamp': '2025-01-01T02:00:00.000'
    }
    df = pd.DataFrame([valid_row, invalid_row_shape, invalid_row_nan])
    csv_path = tmp_path / 'test_data.csv'
    df.to_csv(csv_path, index=False)
    # Load data using the loader
    loaded = load_data(str(csv_path))
    # Only the valid row should remain
    assert len(loaded) == 1
    assert loaded.iloc[0]['case_id'] == 'CASE_001'
    assert isinstance(loaded.iloc[0]['embedding_vector'], list)
    assert len(loaded.iloc[0]['embedding_vector']) == 384 


def test_balance_data():
    # Create imbalanced data
    df = pd.DataFrame({
        'case_id': [f'CASE_{i:03d}' for i in range(20)],
        'embedding_vector': [list(np.random.randn(384)) for _ in range(20)],
        'timestamp': ['2025-01-01T00:00:00.000']*20,
        'label': ['PC']*5 + ['NOT_PC']*15
    })
    df_bal = balance_data(df, label_col='label', method='SMOTE', random_state=42)
    assert abs(df_bal['label'].value_counts()['PC'] - df_bal['label'].value_counts()['NOT_PC']) <= 1


def test_stratified_train_test_split():
    df = pd.DataFrame({
        'case_id': [f'CASE_{i:03d}' for i in range(20)],
        'embedding_vector': [list(np.random.randn(384)) for _ in range(20)],
        'timestamp': ['2025-01-01T00:00:00.000']*20,
        'label': ['PC']*10 + ['NOT_PC']*10
    })
    X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_col='label', test_size=0.2, random_state=42)
    assert len(X_train) == 16
    assert len(X_test) == 4
    # Check stratification
    assert set(np.unique(y_train)) == set(np.unique(y_test))


def test_train_and_evaluate():
    df = pd.DataFrame({
        'case_id': [f'CASE_{i:03d}' for i in range(40)],
        'embedding_vector': [list(np.random.randn(384)) for _ in range(40)],
        'timestamp': ['2025-01-01T00:00:00.000']*40,
        'label': ['PC']*20 + ['NOT_PC']*20
    })
    X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_col='label', test_size=0.25, random_state=42)
    config = {
        'type': 'LogisticRegression',
        'hyperparameters': {'penalty': 'l1', 'C': 10, 'solver': 'liblinear', 'random_state': 42},
        'grid_search': False,
        'cross_val': False,
        'cv_folds': 3,
        'scoring': 'f1',
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    model, metadata = train_model(X_train, y_train, config)
    assert 'model_version' in metadata
    metrics = evaluate_model(model, X_test, y_test, case_ids=[f'CASE_{i:03d}' for i in range(len(y_test))], model_version=metadata['model_version'], embedding_model='all-MiniLM-L6-v2')
    assert 'f1' in metrics
    assert 'accuracy' in metrics 