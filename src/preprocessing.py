"""
Data preprocessing module for Credit Card Approval Prediction System.

This module handles data loading, validation, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class MissingColumnsError(Exception):
    """Raised when expected columns are missing from the dataset."""
    pass


class InvalidTargetValueError(Exception):
    """Raised when target variable contains invalid values."""
    pass


class DataPreprocessor:
    """
    Handles data loading, validation, and preprocessing for credit card data.
    
    Expected columns: card, reports, age, income, share, expenditure, 
                     owner, selfemp, dependents, months, majorcards, active
    """
    
    EXPECTED_COLUMNS = [
        'card', 'reports', 'age', 'income', 'share', 'expenditure',
        'owner', 'selfemp', 'dependents', 'months', 'majorcards', 'active'
    ]
    
    CATEGORICAL_FEATURES = ['owner', 'selfemp']
    TARGET_VARIABLE = 'card'
    VALID_TARGET_VALUES = ['yes', 'no']
    
    def __init__(self, data_path: str):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        self.data = None
        self.label_encoders = {}
        self.scaler = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.ParserError: If the CSV file is invalid
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found at: {self.data_path}"
            )
        
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(
                f"Invalid CSV format in file {self.data_path}: {str(e)}"
            )
    
    def validate_columns(self, df: pd.DataFrame = None) -> bool:
        """
        Validate that all expected columns are present in the dataset.
        
        Args:
            df: DataFrame to validate (uses self.data if None)
            
        Returns:
            True if all columns are present
            
        Raises:
            MissingColumnsError: If any expected columns are missing
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        actual_columns = set(df.columns)
        expected_columns = set(self.EXPECTED_COLUMNS)
        missing_columns = expected_columns - actual_columns
        
        if missing_columns:
            raise MissingColumnsError(
                f"Missing columns: {sorted(missing_columns)}"
            )
        
        return True
    
    def identify_missing_values(self, df: pd.DataFrame = None) -> Dict[str, int]:
        """
        Identify and count missing values per column.
        
        Args:
            df: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Dictionary mapping column names to missing value counts
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        missing_counts = df.isnull().sum().to_dict()
        return missing_counts
    
    def validate_target(self, df: pd.DataFrame = None) -> bool:
        """
        Validate that the target variable contains only valid values.
        
        Args:
            df: DataFrame to validate (uses self.data if None)
            
        Returns:
            True if all target values are valid
            
        Raises:
            InvalidTargetValueError: If invalid values are found in target column
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.TARGET_VARIABLE not in df.columns:
            raise ValueError(
                f"Target variable '{self.TARGET_VARIABLE}' not found in dataset"
            )
        
        unique_values = df[self.TARGET_VARIABLE].dropna().unique()
        invalid_values = set(unique_values) - set(self.VALID_TARGET_VALUES)
        
        if invalid_values:
            raise InvalidTargetValueError(
                f"Invalid values in target variable '{self.TARGET_VARIABLE}': "
                f"{sorted(invalid_values)}. Expected only: {self.VALID_TARGET_VALUES}"
            )
        
        return True
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies.
        Uses median for numerical features and mode for categorical features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with missing values imputed
        """
        df_copy = df.copy()
        
        # Get numerical and categorical columns (excluding target)
        numerical_cols = [col for col in df_copy.columns 
                         if col not in self.CATEGORICAL_FEATURES + [self.TARGET_VARIABLE]]
        categorical_cols = [col for col in self.CATEGORICAL_FEATURES 
                           if col in df_copy.columns]
        
        # Impute numerical features with median
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])
        
        # Impute categorical features with mode (most_frequent)
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols].astype(str))
        
        # Impute target variable with mode if it has missing values
        if self.TARGET_VARIABLE in df_copy.columns and df_copy[self.TARGET_VARIABLE].isnull().any():
            target_imputer = SimpleImputer(strategy='most_frequent')
            df_copy[[self.TARGET_VARIABLE]] = target_imputer.fit_transform(df_copy[[self.TARGET_VARIABLE]].astype(str))
        
        return df_copy
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features into numerical format using LabelEncoder.
        Encodes 'owner' and 'selfemp' columns.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with categorical features encoded
        """
        df_copy = df.copy()
        
        for col in self.CATEGORICAL_FEATURES:
            if col in df_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col])
                else:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col])
        
        return df_copy
    
    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features to have mean=0 and std=1 using StandardScaler.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_copy = df.copy()
        
        # Get numerical columns (excluding target and already encoded categoricals)
        numerical_cols = [col for col in df_copy.columns 
                         if col not in self.CATEGORICAL_FEATURES + [self.TARGET_VARIABLE]]
        
        if numerical_cols:
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
            else:
                df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])
        
        return df_copy
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical features using IQR method.
        Caps outliers at the lower and upper bounds rather than removing them.
        
        Args:
            df: DataFrame to process
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
            
        Returns:
            DataFrame with outliers handled
        """
        df_copy = df.copy()
        
        # Get numerical columns (excluding target)
        numerical_cols = [col for col in df_copy.columns 
                         if col not in self.CATEGORICAL_FEATURES + [self.TARGET_VARIABLE]]
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            for col in numerical_cols:
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                if std > 0:  # Avoid division by zero
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_copy
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.TARGET_VARIABLE not in df.columns:
            raise ValueError(f"Target variable '{self.TARGET_VARIABLE}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[self.TARGET_VARIABLE])
        y = df[self.TARGET_VARIABLE]
        
        # Perform stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
