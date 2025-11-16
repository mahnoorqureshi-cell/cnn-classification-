import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(use_default=True, uploaded_file=None):
    '''
    Loads and preprocesses the Breast Cancer dataset.
    
    Parameters:
    - use_default (bool): Whether to use sklearn's built-in dataset.
    - uploaded_file (str or None): Path to a CSV dataset if not using default.
    
    Returns:
    - X_tr_scaled, X_val_scaled, X_test_scaled: Scaled feature sets
    - y_tr, y_val, y_test: Target arrays
    - feature_names: Feature column names
    - target_names: Target class names
    '''
    if use_default:
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    else:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            X = df.drop("target", axis=1).values
            y = df["target"].values
            feature_names = df.drop("target", axis=1).columns
            target_names = ["benign", "malignant"]
        else:
            raise ValueError("Uploaded file is None and use_default is False.")
    
    # Stratified split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    # StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_tr_scaled, X_val_scaled, X_test_scaled, y_tr, y_val, y_test, feature_names, target_names