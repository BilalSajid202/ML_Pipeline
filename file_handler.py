import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean
import getpass
import json
import os

class FileHandler:
    def __init__(self, credentials_file="db_credentials.json"):
        self.credentials_file = credentials_file

    def map_dtype_to_sqlalchemy(self, dtype):
        """Map pandas dtype to SQLAlchemy column types."""
        if pd.api.types.is_integer_dtype(dtype):
            return Integer
        elif pd.api.types.is_float_dtype(dtype):
            return Float
        elif pd.api.types.is_bool_dtype(dtype):
            return Boolean
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return DateTime
        else:
            # Default to String for object or categorical types
            return String(255)

    def save_db_credentials(self, creds):
        """Save database credentials to a JSON file."""
        try:
            with open(self.credentials_file, "w") as f:
                json.dump(creds, f)
            print(f"🔐 Database credentials saved to '{self.credentials_file}'")
        except Exception as e:
            print(f"❌ Failed to save DB credentials: {e}")

    def load_data(self):
        """Load data from CSV/Excel and optionally save to database."""
        print("Enter path of CSV or Excel file:")
        path = input().strip()

        # Load data based on file extension
        if path.endswith('.csv'):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return None, {'db': False}
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            try:
                df = pd.read_excel(path)
            except Exception as e:
                print(f"Error loading Excel file: {e}")
                return None, {'db': False}
        else:
            print("Unsupported file format. Please provide CSV or Excel.")
            return None, {'db': False}

        print("\nData loaded successfully with columns:")
        print(df.columns.tolist())

        # Ask if user wants to save in database
        print("\nDo you want to save this data in a database? (yes/no)")
        save_db = input().strip().lower()

        if save_db == 'yes':
            # Get database credentials from user
            print("\nEnter database details:")
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")
            host = input("Host (default 'localhost'): ").strip() or 'localhost'
            port = input("Port (default '3306'): ").strip() or '3306'
            database = input("Database name: ").strip()
            table_name = input("Table name to create/use: ").strip()

            db_creds = {
                "username": username,
                "password": password,
                "host": host,
                "port": port,
                "database": database,
                "table_name": table_name
            }

            self.save_db_credentials(db_creds)

            # Optional column renaming
            print("\nCurrent columns:", df.columns.tolist())
            print("Do you want to rename columns before saving? (yes/no)")
            rename = input().strip().lower()

            if rename == 'yes':
                new_columns = []
                for col in df.columns:
                    new_name = input(f"Rename column '{col}' to (press Enter to keep same): ").strip()
                    new_columns.append(new_name if new_name else col)
                df.columns = new_columns

            # Create database engine
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            try:
                engine = create_engine(connection_string)
                metadata = MetaData()

                # Define SQLAlchemy table schema based on DataFrame
                columns = []
                for col_name, dtype in zip(df.columns, df.dtypes):
                    col_type = self.map_dtype_to_sqlalchemy(dtype)
                    columns.append(Column(col_name, col_type))

                table = Table(table_name, metadata, *columns)

                # Optional: drop existing table
                print(f"\nDo you want to drop table '{table_name}' if it exists and recreate? (yes/no)")
                drop_recreate = input().strip().lower()
                if drop_recreate == 'yes':
                    metadata.drop_all(engine, tables=[table], checkfirst=True)
                    print(f"Table '{table_name}' dropped.")

                # Create table if it does not exist
                metadata.create_all(engine, checkfirst=True)
                print(f"Table '{table_name}' created or already exists.")

                # Insert DataFrame into the table
                df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
                print(f"\nData inserted successfully into table '{table_name}'.")
                return df, {'db': True, 'engine': engine, 'table': table_name}

            except Exception as e:
                print(f"Error during database operations: {e}")
                return df, {'db': False}
        else:
            print("\nProceeding with data without storing in database.")
            return df, {'db': False}

# 📌 Step 1: Load Dataset
if __name__ == "__main__":
    print("📥 Step 1: Load Dataset")

    handler = FileHandler()
    df, meta = handler.load_data()

    if df is not None:
        print("\n✅ Data loaded successfully. Here's a preview:")
        print(df.head())

        # Save metadata to track source
        with open("data_origin.txt", "w") as f:
            f.write("db" if meta.get("db") else "file")

        # Optionally save locally
        if not meta.get("db"):
            df.to_csv("loaded_data.csv", index=False)
            print("💾 Data saved locally as 'loaded_data.csv'.")
    else:
        print("❌ Failed to load data.")
