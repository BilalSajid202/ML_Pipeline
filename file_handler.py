# 1_data_loader.py

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

def load_data(file_path):
    """Loads CSV or Excel file into a DataFrame"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type.")
    return df

def rename_columns(df):
    """Allows user to rename columns if desired"""
    print("\nCurrent Columns:")
    print(df.columns.tolist())
    response = input("\nDo you want to rename columns? (y/n): ")
    if response.lower() == 'y':
        new_columns = {}
        for col in df.columns:
            new_name = input(f"Rename '{col}' to (press enter to keep same): ")
            new_columns[col] = new_name if new_name else col
        df.rename(columns=new_columns, inplace=True)
    return df

def save_to_database(df):
    """Saves DataFrame to MySQL database"""
    print("\nEnter MySQL credentials:")
    host = input("Host: ")
    user = input("User: ")
    password = input("Password: ")
    db = input("Database: ")
    table = input("Table Name: ")

    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{db}')
    df.to_sql(name=table, con=engine, if_exists='replace', index=False)
    print(f"\n✅ Data successfully saved to {db}.{table}")

def main():
    path = input("Enter path of your data file: ")
    df = load_data(path)
    df = rename_columns(df)

    choice = input("\nDo you want to work with data directly or save it to DB? (direct/db): ")
    if choice.lower() == 'db':
        save_to_database(df)
        with open("data_origin.txt", "w") as f:
            f.write("db")
    else:
        df.to_csv("loaded_data.csv", index=False)
        with open("data_origin.txt", "w") as f:
            f.write("file")
    print("\n✅ Step 1 complete: Data loaded and handled!")

if __name__ == "__main__":
    main()
