# 5_data_labeling.py

from sklearn.preprocessing import LabelEncoder
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

def label_data(df, target_col):
    """Ask user how to encode and apply transformation"""
    print(f"\nHow would you like to encode the target column '{target_col}'?")
    print("1. Label Encoding (e.g., Cat → 0, Dog → 1)")
    print("2. One-Hot Encoding (e.g., Cat → [1,0], Dog → [0,1])")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        df = label_encode_column(df, target_col)
    elif choice == "2":
        df = one_hot_encode(df, [target_col])
    else:
        print("Invalid choice. Applying label encoding by default.")
        df = label_encode_column(df, target_col)

    return df
