import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from .utils import get_logger

logger = get_logger('preprocessing')

# preprocessing.py: data balancing for model training pipeline

def balance_data(df, label_col='intent', method='SMOTE', random_state=42):
    """
    Balance the dataset using SMOTE or similar technique.
    Logs class distribution before and after balancing.
    Returns balanced DataFrame.
    """
    logger.info(f"Class distribution before balancing: {df[label_col].value_counts().to_dict()}")
    X = np.vstack(df['embeddings'].values)
    y = df[label_col].values
    if method == 'SMOTE':
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Create a new DataFrame with the resampled data
        df_res = pd.DataFrame({
            'embeddings': list(X_res),
            label_col: y_res
        })
        
        # Copy other columns from the original DataFrame (use first row as template)
        for col in df.columns:
            if col not in ['embeddings', label_col]:
                # Fill with the first value from original DataFrame for all rows
                df_res[col] = df[col].iloc[0]
        
        logger.info(f"Class distribution after balancing: {pd.Series(y_res).value_counts().to_dict()}")
        return df_res
    else:
        logger.warning(f"Unknown balancing method: {method}. Returning original DataFrame.")
        return df


def stratified_train_test_split(df, label_col='intent', test_size=0.2, random_state=42):
    """
    Split data into train/test sets, stratified by label.
    Logs split sizes and class distributions.
    Returns X_train, X_test, y_train, y_test.
    """
    X = np.vstack(df['embeddings'].values)
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    logger.info(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test

def preprocess_for_training(df, embedding_col='embeddings', intent_col='intent'):
    """
    Preprocess the dataset for training.
    - Extract embeddings
    - One-hot encode the intent column
    Returns processed DataFrame.
    """
    # Extract embeddings
    X = np.vstack(df[embedding_col].values)
    
    # One-hot encode intent
    intents = pd.get_dummies(df[intent_col])
    
    # Combine embeddings and intents
    processed_df = pd.concat([pd.DataFrame(X), intents], axis=1)
    
    return processed_df 