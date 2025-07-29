# training.py: Model training and hyperparameter search

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from datetime import datetime
import json
import os
from .utils import get_logger

logger = get_logger('training')

def train_model(X, y, config, model_dir='models/', model_name='pcc_model.joblib', metadata_name='metadata.json'):
    """
    Train a model using the provided data and config. Optionally perform grid search/CV.
    Save model and metadata.
    Returns trained model and best params.
    """
    model_type = config.get('type', 'LogisticRegression')
    params = config.get('hyperparameters', {})
    search_method = config.get('search_method', 'none')  # 'grid', 'random', 'none'
    n_iter = config.get('n_iter', 20)  # Number of iterations for random search
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
    search_info = {}
    
    if search_method == 'grid':
        logger.info("Running grid search for hyperparameter optimization...")
        # Grid search with predefined parameter combinations
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        grid = GridSearchCV(model, param_grid=param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
        grid.fit(X, y)
        model = grid.best_estimator_
        best_params = grid.best_params_
        search_info = {
            'method': 'grid_search',
            'best_score': grid.best_score_,
            'n_combinations': len(grid.cv_results_['params'])
        }
        logger.info(f"Grid search completed. Best params: {best_params}")
        logger.info(f"Best CV score: {grid.best_score_:.4f}")
        
    elif search_method == 'random':
        logger.info(f"Running random search for hyperparameter optimization ({n_iter} iterations)...")
        # Random search with continuous parameter ranges
        param_distributions = {
            'C': np.logspace(-2, 3, 1000),  # 0.01 to 1000
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_distributions, 
            n_iter=n_iter,
            cv=cv_folds, 
            scoring=scoring, 
            n_jobs=-1,
            random_state=random_state
        )
        random_search.fit(X, y)
        model = random_search.best_estimator_
        best_params = random_search.best_params_
        search_info = {
            'method': 'random_search',
            'best_score': random_search.best_score_,
            'n_iterations': n_iter
        }
        logger.info(f"Random search completed. Best params: {best_params}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
    else:
        logger.info(f"Training model with fixed params: {params}")
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
        'search_method': search_method,
        'search_info': search_info,
        'cv_folds': cv_folds if search_method != 'none' or config.get('cross_val', False) else None,
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


def random_search(X, y, model, param_distributions, n_iter=20, cv=5, scoring='f1', random_state=42):
    """
    Perform random search for hyperparameter tuning.
    Returns best estimator and best params.
    """
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_distributions, 
        n_iter=n_iter,
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1,
        random_state=random_state
    )
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_params_ 