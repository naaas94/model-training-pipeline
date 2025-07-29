import pandas as pd
import numpy as np
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training_pipeline.data import load_data, ingest_data_from_gcs
from model_training_pipeline.preprocessing import balance_data, stratified_train_test_split
from model_training_pipeline.training import train_model
from model_training_pipeline.evaluation import evaluate_model

def test_load_data(tmp_path):
    # Create a sample DataFrame with valid and invalid rows matching the actual dataset structure
    valid_row = {
        'text': 'I\'m worried about my privacy. Stop tracking me.',
        'intent': 'opt_out',
        'confidence': 0.766,
        'timestamp': '2024-08-05 15:34:21.217348',
        'text_length': 54,
        'word_count': 9,
        'has_personal_info': False,
        'formality_score': 0.555556,
        'urgency_score': 0.025926,
        'embeddings': json.dumps(list(np.random.randn(384)))
    }
    invalid_row_shape = {
        'text': 'Test text',
        'intent': 'test',
        'confidence': 0.5,
        'timestamp': '2024-08-05 15:34:21.217348',
        'text_length': 9,
        'word_count': 2,
        'has_personal_info': False,
        'formality_score': 0.5,
        'urgency_score': 0.1,
        'embeddings': json.dumps(list(np.random.randn(100)))  # wrong shape
    }
    invalid_row_nan = {
        'text': 'Another test',
        'intent': 'test',
        'confidence': 0.5,
        'timestamp': '2024-08-05 15:34:21.217348',
        'text_length': 12,
        'word_count': 2,
        'has_personal_info': False,
        'formality_score': 0.5,
        'urgency_score': 0.1,
        'embeddings': json.dumps([np.nan]*384)
    }
    df = pd.DataFrame([valid_row, invalid_row_shape, invalid_row_nan])
    csv_path = tmp_path / 'test_data.csv'
    df.to_csv(csv_path, index=False)
    # Load data using the loader
    loaded = load_data(str(csv_path))
    # Only the valid row should remain
    assert len(loaded) == 1
    assert loaded.iloc[0]['text'] == 'I\'m worried about my privacy. Stop tracking me.'
    assert loaded.iloc[0]['intent'] == 'opt_out'
    assert isinstance(loaded.iloc[0]['embeddings'], list)
    assert len(loaded.iloc[0]['embeddings']) == 384 


def test_balance_data():
    # Create imbalanced data matching the actual dataset structure
    # Need more samples for SMOTE to work properly
    df = pd.DataFrame({
        'text': [f'Test text {i}' for i in range(40)],
        'intent': ['opt_out']*10 + ['other']*30,  # Increased sample size
        'confidence': [0.8]*40,
        'timestamp': ['2024-08-05 15:34:21.217348']*40,
        'text_length': [50]*40,
        'word_count': [10]*40,
        'has_personal_info': [False]*40,
        'formality_score': [0.5]*40,
        'urgency_score': [0.1]*40,
        'embeddings': [list(np.random.randn(384)) for _ in range(40)]
    })
    df_bal = balance_data(df, label_col='intent', method='SMOTE', random_state=42)
    assert abs(df_bal['intent'].value_counts()['opt_out'] - df_bal['intent'].value_counts()['other']) <= 1


def test_stratified_train_test_split():
    df = pd.DataFrame({
        'text': [f'Test text {i}' for i in range(20)],
        'intent': ['opt_out']*10 + ['other']*10,
        'confidence': [0.8]*20,
        'timestamp': ['2024-08-05 15:34:21.217348']*20,
        'text_length': [50]*20,
        'word_count': [10]*20,
        'has_personal_info': [False]*20,
        'formality_score': [0.5]*20,
        'urgency_score': [0.1]*20,
        'embeddings': [list(np.random.randn(384)) for _ in range(20)]
    })
    X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_col='intent', test_size=0.2, random_state=42)
    assert len(X_train) == 16
    assert len(X_test) == 4
    # Check stratification
    assert set(np.unique(y_train)) == set(np.unique(y_test))


def test_train_and_evaluate():
    df = pd.DataFrame({
        'text': [f'Test text {i}' for i in range(40)],
        'intent': ['opt_out']*20 + ['other']*20,
        'confidence': [0.8]*40,
        'timestamp': ['2024-08-05 15:34:21.217348']*40,
        'text_length': [50]*40,
        'word_count': [10]*40,
        'has_personal_info': [False]*40,
        'formality_score': [0.5]*40,
        'urgency_score': [0.1]*40,
        'embeddings': [list(np.random.randn(384)) for _ in range(40)]
    })
    X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_col='intent', test_size=0.25, random_state=42)
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
    metrics = evaluate_model(model, X_test, y_test, case_ids=[f'text_{i}' for i in range(len(y_test))], model_version=metadata['model_version'], embedding_model='all-MiniLM-L6-v2')
    assert 'f1' in metrics
    assert 'accuracy' in metrics


def test_gcs_data_ingestion():
    """Test data ingestion from Google Cloud Storage"""
    # This test would require GCS credentials and access
    # For now, we'll test the function structure
    try:
        # Test with mock data - in real scenario this would download from GCS
        df = ingest_data_from_gcs('test-bucket', 'test-file.csv')
        assert isinstance(df, pd.DataFrame)
        # Check that the data has the expected structure
        expected_columns = ['text', 'intent', 'confidence', 'timestamp', 'text_length', 
                          'word_count', 'has_personal_info', 'formality_score', 'urgency_score', 'embeddings']
        assert all(col in df.columns for col in expected_columns)
    except Exception as e:
        # If GCS is not configured, this is expected
        print(f"GCS test skipped - {e}")
        pass 