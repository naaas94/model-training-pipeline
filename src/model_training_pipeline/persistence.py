import pandas as pd
import json
from datetime import datetime, date
from typing import Optional, List

from .utils import get_logger

logger = get_logger('persistence')

# PCC Output Schema fields
OUTPUT_FIELDS = [
    'case_id', 'predicted_label', 'subtype_label', 'confidence', 'model_version',
    'embedding_model', 'inference_timestamp', 'prediction_notes', 'ingestion_time'
]

# Monitoring Log Schema fields
MONITOR_FIELDS = [
    'run_id', 'model_version', 'embedding_model', 'partition_date', 'runtime_ts', 'status',
    'total_cases', 'passed_validation', 'dropped_cases', 'notes', 'ingestion_time',
    'processing_duration_seconds', 'error_message'
]

def save_predictions(df: pd.DataFrame, path: str, model_version: str, embedding_model: str, notes: Optional[str] = None):
    """
    Save predictions DataFrame to Parquet or CSV, ensuring all required fields are present.
    Adds ingestion_time and fills missing fields as needed.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input to save_predictions must be a DataFrame, got {type(df)}")
    now = pd.Timestamp.utcnow()
    df = df.copy()
    df['model_version'] = model_version
    df['embedding_model'] = embedding_model
    df['inference_timestamp'] = now
    df['ingestion_time'] = now
    if 'prediction_notes' not in df:
        df['prediction_notes'] = notes or ''
    if 'subtype_label' not in df:
        df['subtype_label'] = None
    # Ensure all required fields are present
    for col in OUTPUT_FIELDS:
        if col not in df:
            df[col] = None
    df = df[OUTPUT_FIELDS]
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    elif path.endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format for predictions: {path}")
    logger.info(f"Saved predictions to {path} ({len(df)} rows)")

def save_monitoring_log(log: dict, path: str):
    """
    Save a monitoring log dict to a JSON file, ensuring all required fields are present.
    Adds ingestion_time and partition_date if missing.
    """
    log = log.copy()
    now = datetime.utcnow()
    if 'ingestion_time' not in log:
        log['ingestion_time'] = now.isoformat()
    if 'partition_date' not in log:
        log['partition_date'] = date.today().isoformat()
    for col in MONITOR_FIELDS:
        if col not in log:
            log[col] = None
    log = {k: log[k] for k in MONITOR_FIELDS}
    # Append to file as a JSON line
    with open(path, 'a') as f:
        f.write(json.dumps(log) + '\n')
    logger.info(f"Appended monitoring log to {path}") 