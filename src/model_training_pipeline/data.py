import pandas as pd
import numpy as np
import logging
import ast
import json
from typing import Tuple, List

from .utils import get_logger
from google.cloud import storage

logger = get_logger('data')

REQUIRED_COLUMNS = ['text', 'intent', 'confidence', 'timestamp', 'text_length', 'word_count', 'has_personal_info', 'formality_score', 'urgency_score', 'embeddings']
EMBEDDING_DIM = 384

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV, Parquet, or PKL file.
    Returns a DataFrame with only valid rows (invalid rows are dropped and logged).
    """
    if path.endswith('.csv'):
        # Try different parsing methods for embeddings
        try:
            df = pd.read_csv(path, converters={'embeddings': lambda x: json.loads(x) if pd.notna(x) else [np.nan]*EMBEDDING_DIM})
        except (json.JSONDecodeError, ValueError):
            try:
                df = pd.read_csv(path, converters={'embeddings': lambda x: ast.literal_eval(x) if pd.notna(x) else [np.nan]*EMBEDDING_DIM})
            except (ValueError, SyntaxError):
                # Fallback: read as string and parse manually
                df = pd.read_csv(path)
                df['embeddings'] = df['embeddings'].apply(lambda x: json.loads(x) if pd.notna(x) else [np.nan]*EMBEDDING_DIM)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.pkl'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Loaded object is not a DataFrame: {type(df)}")

    logger.info(f"Loaded {len(df)} rows from {path}")
    df_valid, dropped_rows = validate_data(df)
    logger.info(f"{len(df_valid)} valid rows, {len(dropped_rows)} dropped rows after validation")
    if dropped_rows:
        logger.warning(f"Dropped rows indices: {dropped_rows}")
    return df_valid.reset_index(drop=True)

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Validate that DataFrame matches required schema and embedding vectors are correct.
    Returns (valid DataFrame, list of dropped row indices).
    """
    # Check for required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame({col: [] for col in REQUIRED_COLUMNS}), list(df.index)
    
    # Vectorized validation for better performance
    valid_mask = pd.Series(True, index=df.index)
    
    # Validate embedding vectors
    for idx, embeddings in df['embeddings'].items():
        if not isinstance(embeddings, (list, np.ndarray)):
            valid_mask[idx] = False
            continue
        
        vec = np.asarray(embeddings)
        if vec.shape != (EMBEDDING_DIM,) or np.isnan(vec).any():
            valid_mask[idx] = False
    
    valid_df = df[valid_mask].copy()
    dropped_indices = df[~valid_mask].index.tolist()
    
    logger.info(f"Embedding validation complete: {len(valid_df)} valid, {len(dropped_indices)} dropped")
    return valid_df, dropped_indices


def ingest_data_from_gcs(bucket_name, source_blob_name):
    """
    Ingest data from a GCS bucket and store it in the 'data' directory.
    """
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    destination_file_name = 'data/' + source_blob_name.split('/')[-1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(destination_file_name)
    return df


# Example usage
# df = ingest_data_from_gcs('pcc-datasets', 'balanced_dataset_20250728.csv') 