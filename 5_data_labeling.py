
# 5_data_labeling.py

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def label_encode_column(df, column):
    """Label encodes a specific column"""
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

def one_hot_encode(df, columns):
    """Applies one-hot encoding to specified columns"""
    return pd.get_dummies(df, columns=columns)

def get_target_column(df):
    """Ask user to specify the dependent variable"""
    print("Available columns:", df.columns.tolist())
    return input("Enter the name of the dependent column: ")
