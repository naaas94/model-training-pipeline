# evaluation.py: Model evaluation and visualization

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report
from datetime import datetime
from .persistence import save_predictions, save_monitoring_log
from .utils import get_logger
from typing import List
import json

logger = get_logger('evaluation')

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, case_ids: List[str] = None, 
                  model_version: str = None, embedding_model: str = None) -> dict:
    """
    Evaluate a trained model and return metrics.
    """
    logger = get_logger('evaluation')
    
    # Prepare test data
    X = np.array(X_test['embeddings'].tolist())
    y = y_test.values
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
    else:
        y_proba = None
    
    # Calculate metrics
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y, y_pred)
    metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted')
    metrics['f1_macro'] = f1_score(y, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y, y_pred, average='micro')
    
    # Per-class F1 scores
    unique_labels = np.unique(y)
    per_class_f1 = f1_score(y, y_pred, average=None, labels=unique_labels)
    for i, label in enumerate(unique_labels):
        metrics[f'f1_{label}'] = per_class_f1[i]
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=unique_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    # ROC AUC and PR AUC for multi-class (one-vs-rest)
    if y_proba is not None and len(unique_labels) > 2:
        try:
            # One-vs-rest ROC AUC
            metrics['roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
            # One-vs-rest PR AUC
            metrics['pr_auc_ovr'] = average_precision_score(y, y_proba, average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC/PR AUC: {e}")
            metrics['roc_auc_ovr'] = None
            metrics['pr_auc_ovr'] = None
    elif y_proba is not None and len(unique_labels) == 2:
        # Binary classification
        try:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
            metrics['pr_auc'] = average_precision_score(y, y_proba[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate ROC/PR AUC: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'case_id': case_ids if case_ids else [f'case_{i}' for i in range(len(y))],
        'true_label': y,
        'predicted_label': y_pred,
        'confidence': np.max(y_proba, axis=1) if y_proba is not None else [1.0] * len(y)
    })
    
    # Add prediction probabilities for each class
    if y_proba is not None:
        for i, label in enumerate(unique_labels):
            predictions_df[f'prob_{label}'] = y_proba[:, i]
    
    # Save predictions and metrics
    # Get output paths from config
    from .utils import load_config
    import os
    
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)
    predictions_path = config['output']['predictions']
    metrics_path = config['output']['metrics']
    
    # Save predictions
    save_predictions(predictions_df, predictions_path, model_version, embedding_model)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    return metrics

# Placeholder for visualization function
def plot_metrics(metrics):
    """Plot evaluation metrics (ROC, PR curves, etc.)."""
    pass 