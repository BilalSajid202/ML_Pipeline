from sklearn.preprocessing import LabelEncoder
import pandas as pd

class DataLabeler:
    """
    Class for encoding target variables in a dataset.
    Supports label encoding and one-hot encoding.
    """

    def __init__(self, df):
        """
        Initialize with the DataFrame to label.
        """
        self.df = df.copy()

    def label_encode_column(self, column):
        """
        Label encode a specific column in the DataFrame.
        Converts categorical labels into integer codes.
        """
        le = LabelEncoder()
        self.df[column] = le.fit_transform(self.df[column])
        return self.df

    def one_hot_encode(self, columns):
        """
        Apply one-hot encoding to specified columns.
        Converts categorical variables into multiple binary columns.
        """
        self.df = pd.get_dummies(self.df, columns=columns)
        return self.df

    def get_target_column(self):
        """
        Prompts the user to specify the dependent variable (target column).
        Prints available columns to guide the user.
        """
        print("Available columns:", self.df.columns.tolist())
        target_col = input("Enter the name of the dependent column: ")
        return target_col

    def label_data(self, target_col):
        """
        Ask the user how to encode the target column and apply the chosen encoding.
        Options:
        1 - Label Encoding (single integer per category)
        2 - One-Hot Encoding (binary columns per category)
        Defaults to Label Encoding if invalid input.
        """
        print(f"\nHow would you like to encode the target column '{target_col}'?")
        print("1. Label Encoding (e.g., Cat → 0, Dog → 1)")
        print("2. One-Hot Encoding (e.g., Cat → [1,0], Dog → [0,1])")

        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return self.label_encode_column(target_col)
        elif choice == "2":
            return self.one_hot_encode([target_col])
        else:
            print("Invalid choice. Applying label encoding by default.")
            return self.label_encode_column(target_col)
