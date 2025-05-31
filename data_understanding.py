import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
import json

class DataUnderstanding:
    def __init__(self, creds_file="db_credentials.json", local_file="loaded_data.csv", origin_file="data_origin.txt"):
        self.creds_file = creds_file
        self.local_file = local_file
        self.origin_file = origin_file

    def fetch_data_from_file(self):
        """Loads data from a local CSV file."""
        try:
            df = pd.read_csv(self.local_file)
            print(f"📂 Data loaded from file: {self.local_file}")
            return df
        except FileNotFoundError:
            print(f"❌ File '{self.local_file}' not found.")
            return None
        except Exception as e:
            print(f"❌ Error loading data from file: {e}")
            return None

    def fetch_data_from_db_using_saved_credentials(self):
        """Loads data from MySQL database using saved credentials."""
        if not os.path.exists(self.creds_file):
            print(f"❌ Credentials file '{self.creds_file}' not found. Please run Step 1 first.")
            return None

        try:
            with open(self.creds_file, "r") as f:
                creds = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read credentials file: {e}")
            return None

        try:
            print(f"🔌 Connecting to MySQL on {creds['host']}:{creds['port']}...")

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

            print(f"✅ Data loaded from DB '{creds['database']}', table '{creds['table_name']}'.")
            return df

        except Error as e:
            print(f"❌ Database connection or query failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def data_summary(self, df):
        """Performs basic data analysis and prints summary."""
        if df is None or df.empty:
            print("⚠️ No data to summarize.")
            return

        print("\n📊 Data Shape:", df.shape)
        print("\n🧬 Data Types:\n", df.dtypes)
        print("\n❓ Missing Values:\n", df.isnull().sum())
        print("\n📈 Descriptive Statistics:\n", df.describe(include='all'))
        print("\n🔍 Sample Rows:\n", df.head())

    def load_data_based_on_origin(self):
        """Load data depending on the saved origin (file/db)."""
        if os.path.exists(self.origin_file):
            with open(self.origin_file, "r") as f:
                origin = f.read().strip().lower()
        else:
            print(f"⚠️ File '{self.origin_file}' not found. Defaulting to 'file'.")
            origin = "file"

        if origin == "db":
            return self.fetch_data_from_db_using_saved_credentials()
        elif origin == "file":
            return self.fetch_data_from_file()
        else:
            print(f"❌ Invalid origin '{origin}' in '{self.origin_file}'. Expected 'db' or 'file'.")
            return None

# 📌 Step 2: Understanding Dataset
if __name__ == "__main__":
    print("📖 Step 2: Understanding Dataset")

    du = DataUnderstanding()
    df = du.load_data_based_on_origin()

    if df is not None:
        print("\n✅ Data loaded successfully. Summary below:\n")
        du.data_summary(df)
    else:
        print("⚠️ Failed to load data. Please check previous steps or data source.")
