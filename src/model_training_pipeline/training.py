# training.py: Model training and hyperparameter search

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from datetime import datetime
import json
import os
from .utils import get_logger
import pandas as pd
from typing import Tuple, Any

logger = get_logger('training')

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> Tuple[Any, dict]:
    """
    Train a model with the given configuration.
    Returns (trained_model, metadata_dict).
    """
    logger = get_logger('training')
    
    # Extract configuration
    model_type = config.get('type', 'LogisticRegression')
    params = config.get('hyperparameters', {})
    grid_search = config.get('grid_search', False)
    search_method = config.get('search_method', 'grid')
    n_iter = config.get('n_iter', 20)
    cross_val = config.get('cross_val', False)
    cv_folds = config.get('cv_folds', 5)
    scoring = config.get('scoring', 'f1_weighted')  # Changed to weighted for multi-class
    random_state = params.get('random_state', 42)
    
    # Prepare features (embeddings)
    X = np.array(X_train['embeddings'].tolist())
    y = y_train.values
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Target classes: {np.unique(y)}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Data validation checks
    logger.info(f"Feature statistics - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    logger.info(f"Feature range - Min: {X.min():.4f}, Max: {X.max():.4f}")
    
    # Check for potential data issues
    if X.std() < 1e-6:
        logger.warning("Very low feature variance detected - possible data leakage or preprocessing issue")
    
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class found in training data")
    
    # Check for class imbalance
    class_counts = pd.Series(y).value_counts()
    min_class_ratio = class_counts.min() / class_counts.max()
    if min_class_ratio < 0.1:
        logger.warning(f"Severe class imbalance detected: {min_class_ratio:.3f}")
    
    # Initialize model
    if model_type == 'LogisticRegression':
        # Remove random_state from params to avoid duplicate argument
        model_params = {k: v for k, v in params.items() if k != 'random_state'}
        model = LogisticRegression(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Hyperparameter optimization
    best_params = params
    search_info = {}
    
    if search_method == 'grid':
        logger.info("Running grid search for hyperparameter optimization...")
        param_grid = {
            'C': [0.1, 1, 5],  # More conservative range
            'penalty': ['l2'],  # Only l2 to avoid solver conflicts
            'solver': ['liblinear', 'lbfgs'],  # Only compatible solvers
            'max_iter': [1000, 2000]  # Higher max_iter values
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
        
        # Check for suspiciously high CV scores
        if grid.best_score_ > 0.99:
            logger.warning(f"Suspiciously high CV score ({grid.best_score_:.4f}) - possible overfitting or data leakage")
    
    elif search_method == 'random':
        logger.info(f"Running random search for hyperparameter optimization ({n_iter} iterations)...")
        param_distributions = {
            'C': np.logspace(-2, 1, 1000),  # 0.01 to 10 (more conservative)
            'penalty': ['l2'],  # Only l2 to avoid solver conflicts
            'solver': ['liblinear', 'lbfgs'],  # Removed saga, only compatible solvers
            'max_iter': [1000, 2000]  # Higher max_iter values
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
        
        # Check for suspiciously high CV scores
        if random_search.best_score_ > 0.99:
            logger.warning(f"Suspiciously high CV score ({random_search.best_score_:.4f}) - possible overfitting or data leakage")
    
    else:
        logger.info(f"Training model with fixed params: {params}")
        model.fit(X, y)
    
    # Create metadata
    metadata = {
        'model_type': model_type,
        'hyperparameters': best_params,
        'search_method': search_method,
        'search_info': search_info,
        'training_samples': len(X),
        'feature_dimensions': X.shape[1],
        'classes': np.unique(y).tolist(),
        'class_distribution': pd.Series(y).value_counts().to_dict(),
        'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'training_timestamp': datetime.now().isoformat()
    }
    
    # Save model locally
    model_path = 'models/pcc_model.joblib'
    metadata_path = 'models/metadata.json'
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved model to {model_path}")
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