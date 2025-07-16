# training.py: Model training and hyperparameter search

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from datetime import datetime
import json
import os
from .utils import get_logger

logger = get_logger('training')

# training.py: Model training and hyperparameter search

# Placeholder for model training function
def train_model(X, y, config, model_dir='models/', model_name='pcc_model.joblib', metadata_name='metadata.json'):
    """
    Train a model using the provided data and config. Optionally perform grid search/CV.
    Save model and metadata.
    Returns trained model and best params.
    """
    model_type = config.get('type', 'LogisticRegression')
    params = config.get('hyperparameters', {})
    grid_search_enabled = config.get('grid_search', False)
    cv_folds = config.get('cv_folds', 5)
    scoring = config.get('scoring', 'f1')
    random_state = params.get('random_state', 42)
    os.makedirs(model_dir, exist_ok=True)

    if model_type == 'LogisticRegression':
        model = LogisticRegression(**params)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    best_params = params
    if grid_search_enabled:
        logger.info(f"Running grid search with params: {params}")
        grid = GridSearchCV(model, param_grid=params, cv=cv_folds, scoring=scoring, n_jobs=-1)
        grid.fit(X, y)
        model = grid.best_estimator_
        best_params = grid.best_params_
        logger.info(f"Best params from grid search: {best_params}")
    else:
        logger.info(f"Training model with params: {params}")
        model.fit(X, y)

    # Optionally, cross-validation
    if config.get('cross_val', False):
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        logger.info(f"Cross-validation {scoring} scores: {scores}")

    # Save model
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save metadata
    metadata = {
        'model_version': f"v{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'trained_on': datetime.now().isoformat(),
        'model_type': model_type,
        'embedding_model': config.get('embedding_model', 'all-MiniLM-L6-v2'),
        'hyperparameters': best_params,
        'cv_folds': cv_folds if grid_search_enabled or config.get('cross_val', False) else None,
        'scoring': scoring,
        'train_shape': list(X.shape),
        'notes': config.get('notes', '')
    }
    metadata_path = os.path.join(model_dir, metadata_name)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")
    return model, metadata


def grid_search(X, y, model, param_grid, cv=5, scoring='f1'):
    """
    Perform grid search for hyperparameter tuning.
    Returns best estimator and best params.
    """
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_ 