# evaluation.py: Model evaluation and visualization

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from datetime import datetime
from .persistence import save_predictions, save_monitoring_log
from .utils import get_logger

logger = get_logger('evaluation')

# Placeholder for evaluation function
def evaluate_model(model, X_test, y_test, case_ids=None, model_version=None, embedding_model=None, output_path=None, metrics_path=None, monitor_log_path=None, notes=None):
    """
    Evaluate the model and return/save metrics. Optionally save predictions and monitoring log.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    # Determine pos_label for f1_score
    unique_labels = np.unique(y_test)
    if all(isinstance(l, (int, np.integer)) for l in unique_labels):
        pos_label = int(max(unique_labels))
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)  # type: ignore
    elif all(isinstance(l, str) for l in unique_labels):
        pos_label = 'PC' if 'PC' in unique_labels else str(unique_labels[0])
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)  # type: ignore
    else:
        raise ValueError('Mixed or unknown label types in y_test for f1_score.')
    metrics = {
        'f1': f1,
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'pr_auc': average_precision_score(y_test, y_proba) if y_proba is not None else None,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    logger.info(f"Evaluation metrics: {metrics}")
    # Save metrics
    if metrics_path:
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    # Save predictions
    if output_path and case_ids is not None:
        df_pred = pd.DataFrame({
            'case_id': case_ids,
            'predicted_label': y_pred,
            'confidence': y_proba if y_proba is not None else np.nan
        })
        save_predictions(df_pred, output_path, model_version or '', embedding_model or '', notes)
    # Save monitoring log
    if monitor_log_path:
        log = {
            'run_id': f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'model_version': model_version,
            'embedding_model': embedding_model,
            'runtime_ts': datetime.now().isoformat(),
            'status': 'success',
            'total_cases': len(y_test),
            'passed_validation': int(np.sum(y_pred == y_test)),
            'dropped_cases': int(np.sum(y_pred != y_test)),
            'notes': notes or '',
            'processing_duration_seconds': None,
            'error_message': None
        }
        save_monitoring_log(log, monitor_log_path)
    return metrics

# Placeholder for visualization function
def plot_metrics(metrics):
    """Plot evaluation metrics (ROC, PR curves, etc.)."""
    pass 