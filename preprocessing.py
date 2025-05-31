import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats

class DataPreprocessor:
    """
    Class to encapsulate preprocessing steps for a pandas DataFrame including:
    - Duplicate removal
    - Missing value imputation (automatic strategy)
    - Categorical encoding (label encoding)
    - Numeric feature scaling (standard or min-max)
    - Outlier removal based on Z-score threshold
    """

    def __init__(self, scale_method='standard', z_thresh=3):
        """
        Initialize the preprocessor.

        Parameters:
        - scale_method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        - z_thresh: Z-score threshold above which rows are considered outliers and removed
        """
        self.scale_method = scale_method
        self.z_thresh = z_thresh
        self.label_encoders = {}  # Store label encoders for categorical columns
        self.scaler = None        # Scaler instance (fitted during scaling)

    def handle_missing_values_auto(self, df):
        """
        Impute missing values with automatic strategy per column:
        - Numeric columns: use mean if skewness < 1 else median
        - Categorical columns: use most frequent value

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame with missing values imputed
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue  # Skip columns with no missing values

            if pd.api.types.is_numeric_dtype(df[col]):
                # Check skewness to choose mean or median
                strategy = 'mean' if abs(df[col].skew()) < 1 else 'median'
            else:
                strategy = 'most_frequent'

            imputer = SimpleImputer(strategy=strategy)
            df[[col]] = imputer.fit_transform(df[[col]])

        return df

    def encode_categoricals(self, df):
        """
        Label encode all categorical columns (object dtype).
        Stores encoders in self.label_encoders for possible inverse transforms.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - Encoded DataFrame
        """
        df = df.copy()
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def scale_data(self, df):
        """
        Scale numeric columns using specified method:
        - StandardScaler (default)
        - MinMaxScaler

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame with scaled numeric columns
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include='number').columns

        # Initialize scaler based on scale_method
        if self.scale_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        # Fit and transform numeric columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the DataFrame.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame without duplicate rows
        """
        return df.drop_duplicates()

    def remove_outliers(self, df):
        """
        Remove rows where any numeric column has Z-score exceeding the threshold.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame with outliers removed
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include='number')
        z_scores = stats.zscore(numeric_cols)

        # Keep rows where all numeric columns have Z-score less than threshold
        mask = (abs(z_scores) < self.z_thresh).all(axis=1)
        return df[mask]

    def preprocess_pipeline(self, df):
        """
        Run the complete preprocessing pipeline in order:
        1. Remove duplicates
        2. Handle missing values
        3. Encode categorical variables
        4. Scale numeric features
        5. Remove outliers based on Z-score

        Parameters:
        - df: pandas DataFrame

        Returns:
        - Fully preprocessed DataFrame
        """
        df = self.remove_duplicates(df)
        df = self.handle_missing_values_auto(df)
        df = self.encode_categoricals(df)
        df = self.scale_data(df)
        df = self.remove_outliers(df)
        return df
