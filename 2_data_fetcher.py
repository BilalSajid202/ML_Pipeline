# 2_data_fetcher.py

import pandas as pd
import mysql.connector

def fetch_data_from_file(file_path="loaded_data.csv"):
    """Loads data from local file"""
    return pd.read_csv(file_path)

def fetch_data_from_db():
    """Loads data from MySQL database using credentials"""
    print("Enter MySQL credentials to fetch data:")
    host = input("Host: ")
    user = input("User: ")
    password = input("Password: ")
    db = input("Database: ")
    table = input("Table Name: ")

    conn = mysql.connector.connect(host=host, user=user, password=password, database=db)
    df = pd.read_sql(f"SELECT * FROM {table}", con=conn)
    conn.close()
    return df

def data_summary(df):
    """Performs basic data analysis"""
    print("\nData Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Stats:\n", df.describe())
    print("\nSample Rows:\n", df.head())

def main():
    with open("data_origin.txt", "r") as f:
        origin = f.read().strip()

    df = fetch_data_from_db() if origin == "db" else fetch_data_from_file()
    print("\n✅ Step 2 complete: Data fetched.")
    data_summary(df)

    return df

if __name__ == "__main__":
    main()
