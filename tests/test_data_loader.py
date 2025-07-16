import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training_pipeline.data import load_data

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