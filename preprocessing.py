import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats
from sklearn.impute import SimpleImputer



def handle_missing_values_auto(df):
    """
    Handles missing values with the most suitable strategy per column:
    - Mean or median for numeric columns based on skewness
    - Most frequent for categorical columns
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # Skip columns without missing values

        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = df[col].skew()
            if abs(skewness) < 1:
                strategy = 'mean'
            else:
                strategy = 'median'
        else:
            strategy = 'most_frequent'

        imputer = SimpleImputer(strategy=strategy)
        df[[col]] = imputer.fit_transform(df[[col]])

    return df





def encode_categoricals(df):
    """
    Detects and label-encodes categorical columns.
    Returns encoded DataFrame and dictionary of encoders.
    """
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def scale_data(df, method='standard'):
    """
    Scales numeric columns only using the specified method.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def remove_duplicates(df):
    """Removes duplicate rows"""
    return df.drop_duplicates()

def remove_outliers(df, z_thresh=3):
    """
    Removes rows where any numeric column has a Z-score above threshold.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number')
    z_scores = stats.zscore(numeric_cols)
    mask = (abs(z_scores) < z_thresh).all(axis=1)
    return df[mask]

def preprocess_pipeline(df, scale_method='standard'):
    """
    Full pipeline:
    - Remove duplicates
    - Handle missing values
    - Encode categoricals
    - Scale numeric features
    - Remove outliers
    """
    df = remove_duplicates(df)
    df = handle_missing_values_auto(df)   # auto-handled
    df, encoders = encode_categoricals(df)
    df = scale_data(df, method=scale_method)
    df = remove_outliers(df)
    return df

