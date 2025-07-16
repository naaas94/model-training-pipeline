import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

from .utils import get_logger

logger = get_logger('data')

REQUIRED_COLUMNS = ['case_id', 'embedding_vector', 'timestamp']
EMBEDDING_DIM = 384

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV, Parquet, or PKL file.
    Returns a DataFrame with only valid rows (invalid rows are dropped and logged).
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path, converters={'embedding_vector': lambda x: eval(x, {"nan": np.nan})})
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
    valid_rows = []
    dropped_indices = []
    for idx, row in df.iterrows():
        # Check required columns
        if not all(col in row for col in REQUIRED_COLUMNS):
            dropped_indices.append(idx)
            continue
        # Validate embedding vector
        vec = row['embedding_vector']
        if not isinstance(vec, (list, np.ndarray)):
            dropped_indices.append(idx)
            continue
        vec = np.asarray(vec)
        if vec.shape != (EMBEDDING_DIM,):
            dropped_indices.append(idx)
            continue
        if np.isnan(vec).any():
            dropped_indices.append(idx)
            continue
        # Convert row to dict to avoid Series issues
        valid_rows.append({col: row[col] for col in REQUIRED_COLUMNS})
    logger.info(f"Embedding validation complete: {len(valid_rows)} valid, {len(dropped_indices)} dropped")
    if valid_rows:
        valid_df = pd.DataFrame(valid_rows)
    else:
        valid_df = pd.DataFrame({col: [] for col in REQUIRED_COLUMNS})
    return valid_df, dropped_indices 