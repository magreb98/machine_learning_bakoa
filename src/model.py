"""
Model training module for Credit Card Approval Prediction System.

This module handles model training, hyperparameter tuning, predictions, and persistence.
"""

import numpy as np
import pickle
from typing import Dict, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class CreditCardModel:
    """
    Handles training, prediction, and persistence of the Decision Tree classifier
    for credit card approval prediction.
    """
    
    def __init__(self):
        """Initialize the CreditCardModel."""
        self.model = None
        self.best_params = None
        self.feature_names = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[list] = None) -> None:
        """
        Train a Decision Tree classifier on the provided training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Optional list of feature names for importance tracking
        """
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Initialize and train the Decision Tree classifier
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            cv: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
        
        # Initialize base model
        base_model = DecisionTreeClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters and model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return self.best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions on the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted class labels
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if X is None:
            raise ValueError("Input data cannot be None")
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predicted probabilities for each class
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if X is None:
            raise ValueError("Input data cannot be None")
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        importances = self.model.feature_importances_
        
        # If feature names are available, create a named dictionary
        if self.feature_names is not None:
            if len(self.feature_names) != len(importances):
                raise ValueError(
                    f"Number of feature names ({len(self.feature_names)}) "
                    f"doesn't match number of features ({len(importances)})"
                )
            return dict(zip(self.feature_names, importances))
        else:
            # Otherwise, use generic feature names
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk using pickle.
        
        Args:
            filepath: Path where the model should be saved
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data.get('best_params')
        self.feature_names = model_data.get('feature_names')
