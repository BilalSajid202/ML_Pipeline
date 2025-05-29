import pandas as pd
import mysql.connector
from mysql.connector import Error

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

def fetch_data_from_db():
    """Loads data from MySQL database using user-provided credentials."""
    try:
        print("Enter MySQL credentials to fetch data:")
        host = input("Host (default 'localhost'): ").strip() or "localhost"
        user = input("User: ").strip()
        password = input("Password: ").strip()
        db = input("Database: ").strip()
        table = input("Table Name: ").strip()

        conn = mysql.connector.connect(host=host, user=user, password=password, database=db)
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, con=conn)
        conn.close()
        print(f"Data loaded from database '{db}', table '{table}'.")
        return df

    except Error as e:
        print(f"Database connection or query failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
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

def main():
    try:
        with open("data_origin.txt", "r") as f:
            origin = f.read().strip().lower()
    except FileNotFoundError:
        print("File 'data_origin.txt' not found. Please create this file with 'db' or 'file' inside.")
        return None
    except Exception as e:
        print(f"Error reading 'data_origin.txt': {e}")
        return None

    if origin == "db":
        df = fetch_data_from_db()
    elif origin == "file":
        df = fetch_data_from_file()
    else:
        print(f"Invalid data origin '{origin}' specified in 'data_origin.txt'. Use 'db' or 'file'.")
        return None

    if df is not None:
        print("\n✅ Step 2 complete: Data fetched successfully.")
    else:
        print("\n⚠️ Step 2 failed: Could not fetch data.")

    data_summary(df)
    return df

if __name__ == "__main__":
    main()
