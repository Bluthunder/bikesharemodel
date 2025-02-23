import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pdb

class WeekdayImputer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='dteday', weekday_column="weekday"):
        self.date_column = date_column
        self.weekday_column = weekday_column

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        X = X.copy()
        
        pdb.set_trace()
        
        # Ensure date column is in datetime format
        # X[self.date_column] = pd.to_datetime(X[self.date_column])

        print(f"Columns in DataFrame: {X.columns.tolist()}")
        if self.date_column not in X.columns:
            raise ValueError(f"Column '{self.date_column}' is missing from DataFrame!")
        
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = pd.to_datetime(X[self.date_column], errors="coerce")


        

        # Impute missing weekday values using the day name abbreviation
        X.loc[X[self.weekday_column].isna(), self.weekday_column] = (
            X.loc[X[self.weekday_column].isna(), self.date_column]
            .dt.strftime("%a")  # Get three-letter weekday abbreviation
        )

        return X
    

class WeathersitImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column="weathersit"):
        self.column = column
        self.most_frequent = None

    def fit(self, X, y=None):
        self.most_frequent = X[self.column].mode()[0]  
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].fillna(self.most_frequent)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables:str, mappings:dict):
        self.mappings = mappings
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X


class CustomOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method="iqr", factor=1.5):
        """
        Handles outliers in numerical columns by capping them at upper or lower bounds.

        :param columns: List of columns to process. If None, all numerical columns are considered.
        :param method: Outlier detection method ("iqr" for Interquartile Range).
        :param factor: Multiplier for the IQR range (default is 1.5).
        """
        self.columns = columns
        self.method = method
        self.factor = factor
        self.bounds = {}  # Dictionary to store column-wise outlier thresholds

    def fit(self, X, y=None):
        """Calculate the upper and lower bounds for outlier handling."""
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.columns:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.bounds[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X):
        """Clamp outlier values to the computed bounds."""
        X = X.copy()
        for col, (lower, upper) in self.bounds.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="weekday"):
        """
        One-hot encodes the specified weekday column.

        :param column: Name of the column to encode (default is "weekday").
        """
        self.column = column
        self.unique_values = None

    def fit(self, X, y=None):
        """Identify unique weekday values."""
        self.unique_values = sorted(X[self.column].dropna().unique())  # Get unique weekday values
        return self

    def transform(self, X):
        """Perform one-hot encoding on the weekday column."""
        X = X.copy()
        
        # One-hot encode weekday column
        one_hot_encoded = pd.get_dummies(X[self.column], prefix=self.column)

        # Drop original column and merge one-hot columns
        X = X.drop(columns=[self.column])  # ✅ Make sure to drop the original column
        X = X.join(one_hot_encoded)

        return X
