# 3_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scipy import stats

def handle_missing_values(df, strategy='mean'):
    """
    Handles missing values in the dataset.
    strategy: mean, median, most_frequent
    """
    imputer = SimpleImputer(strategy=strategy)
    df[df.columns] = imputer.fit_transform(df)
    return df

def encode_categoricals(df):
    """
    Automatically detects and label-encodes categorical columns.
    """
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_data(df, method='standard'):
    """
    Scales numeric data using standard or min-max scaler.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)
    return df

def remove_duplicates(df):
    """Removes duplicate rows"""
    return df.drop_duplicates()

def remove_outliers(df, z_thresh=3):
    """
    Removes outliers using Z-score method.
    Only numeric columns are considered.
    """
    numeric_df = df.select_dtypes(include='number')
    z_scores = stats.zscore(numeric_df)
    mask = (abs(z_scores) < z_thresh).all(axis=1)
    return df[mask]

def preprocess_pipeline(df, missing_strategy='mean', scale_method='standard'):
    """
    Complete preprocessing pipeline that applies:
    - Duplicate removal
    - Missing value imputation
    - Encoding
    - Scaling
    - Outlier removal
    """
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    df, encoders = encode_categoricals(df)
    df = scale_data(df, method=scale_method)
    df = remove_outliers(df)
    return df
