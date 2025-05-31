import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
import json

def fetch_data_from_file(file_path="loaded_data.csv"):
    """Loads data from a local CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded from file: {file_path}")
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please check the path and try again.")
        return None
    except Exception as e:
        print(f"Error loading data from file: {e}")
        return None

def fetch_data_from_db_using_saved_credentials(creds_file="db_credentials.json"):
    """Loads data from MySQL database using saved credentials."""
    if not os.path.exists(creds_file):
        print(f"❌ Credentials file '{creds_file}' not found. Please run Step 1 first.")
        return None

    try:
        with open(creds_file, "r") as f:
            creds = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read credentials file: {e}")
        return None

    try:
        print(f"🔌 Connecting to MySQL on {creds['host']}:{creds['port']} using saved credentials...")

        conn = mysql.connector.connect(
            host=creds['host'],
            port=int(creds['port']),
            user=creds['username'],
            password=creds['password'],
            database=creds['database']
        )

        query = f"SELECT * FROM {creds['table_name']}"
        df = pd.read_sql(query, con=conn)
        conn.close()

        print(f"✅ Data loaded from database '{creds['database']}', table '{creds['table_name']}'.")
        return df

    except Error as e:
        print(f"❌ Database connection or query failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def data_summary(df):
    """Performs basic data analysis and prints summary."""
    if df is None or df.empty:
        print("No data to summarize.")
        return
    print("\nData Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Statistics:\n", df.describe(include='all'))
    print("\nSample Rows:\n", df.head())

def load_data_based_on_origin():
    origin_file = "data_origin.txt"

    if os.path.exists(origin_file):
        with open(origin_file, "r") as f:
            origin = f.read().strip().lower()
    else:
        print(f"⚠️ File '{origin_file}' not found. Defaulting to 'file'.")
        origin = "file"

    if origin == "db":
        df = fetch_data_from_db_using_saved_credentials()
    elif origin == "file":
        df = fetch_data_from_file()
    else:
        print(f"❌ Invalid origin '{origin}' in '{origin_file}'. Use 'db' or 'file'.")
        return None

    return df

if __name__ == "__main__":
    print("📖 Step 2: Understanding Dataset")
    df = load_data_based_on_origin()

    if df is not None:
        print("\n✅ Data loaded successfully. Summary below:\n")
        data_summary(df)
    else:
        print("⚠️ Failed to load data. Please check previous steps or data source.")
