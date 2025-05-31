import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean
import getpass
import json
import os

def map_dtype_to_sqlalchemy(dtype):
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

def save_db_credentials(creds, filename="db_credentials.json"):
    try:
        with open(filename, "w") as f:
            json.dump(creds, f)
        print(f"🔐 Database credentials saved to '{filename}'")
    except Exception as e:
        print(f"❌ Failed to save DB credentials: {e}")

def load_data():
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
        print("\nEnter database details:")
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")
        host = input("Host (default 'localhost'): ").strip() or 'localhost'
        port = input("Port (default '3306'): ").strip() or '3306'
        database = input("Database name: ").strip()
        table_name = input("Table name to create/use: ").strip()

        # Save credentials for Step 2 use
        db_creds = {
            "username": username,
            "password": password,
            "host": host,
            "port": port,
            "database": database,
            "table_name": table_name
        }
        save_db_credentials(db_creds)

        # Show current columns and ask if user wants to rename
        print("\nCurrent columns:", df.columns.tolist())
        print("Do you want to rename columns before saving? (yes/no)")
        rename = input().strip().lower()

        if rename == 'yes':
            new_columns = []
            for col in df.columns:
                new_name = input(f"Rename column '{col}' to (press Enter to keep same): ").strip()
                if new_name:
                    new_columns.append(new_name)
                else:
                    new_columns.append(col)
            df.columns = new_columns

        # Create connection string and engine
        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        try:
            engine = create_engine(connection_string)
            metadata = MetaData()

            # Define table columns dynamically based on dataframe dtypes
            columns = []
            for col_name, dtype in zip(df.columns, df.dtypes):
                col_type = map_dtype_to_sqlalchemy(dtype)
                columns.append(Column(col_name, col_type))

            # Create table object
            table = Table(table_name, metadata, *columns)

            # Ask if user wants to drop table if exists and recreate
            print(f"\nDo you want to drop table '{table_name}' if it exists and recreate? (yes/no)")
            drop_recreate = input().strip().lower()
            if drop_recreate == 'yes':
                metadata.drop_all(engine, tables=[table], checkfirst=True)
                print(f"Table '{table_name}' dropped.")

            # Create table in DB if not exists
            metadata.create_all(engine, checkfirst=True)
            print(f"Table '{table_name}' created or already exists.")

            # Insert data into table
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

    df, meta = load_data()

    if df is not None:
        print("\n✅ Data loaded successfully. Here's a preview:")
        print(df.head())

        # Save metadata: store origin so it can be referenced in future steps
        with open("data_origin.txt", "w") as f:
            f.write("db" if meta.get("db") else "file")

        # Optionally store to CSV for later use if not stored in DB
        if not meta.get("db"):
            df.to_csv("loaded_data.csv", index=False)
            print("💾 Data saved locally as 'loaded_data.csv'.")

    else:
        print("❌ Failed to load data.")
