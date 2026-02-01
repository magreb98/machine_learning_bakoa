"""
Evaluation module for Credit Card Approval Prediction System.

This module handles model evaluation, metrics calculation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sklearn_confusion_matrix
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics for model evaluation.
    
    Calculates accuracy, precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1_score
        
    Raises:
        ValueError: If inputs are empty or have mismatched lengths
    """
    if y_true is None or y_pred is None:
        raise ValueError("Input arrays cannot be None")
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label='yes', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label='yes', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label='yes', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Generate confusion matrix from true and predicted labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        2x2 numpy array representing the confusion matrix
        
    Raises:
        ValueError: If inputs are empty or have mismatched lengths
    """
    if y_true is None or y_pred is None:
        raise ValueError("Input arrays cannot be None")
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Specify labels to ensure 2x2 matrix even when only one class is present
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=['no', 'yes'])
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None) -> plt.Figure:
    """
    Create a visualization of the confusion matrix.
    
    Args:
        cm: Confusion matrix as a 2x2 numpy array
        class_names: Optional list of class names for labels (default: ['no', 'yes'])
        
    Returns:
        Matplotlib Figure object containing the confusion matrix heatmap
        
    Raises:
        ValueError: If confusion matrix is not 2x2
    """
    if cm is None:
        raise ValueError("Confusion matrix cannot be None")
    
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")
    
    if class_names is None:
        class_names = ['no', 'yes']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: list, importances: np.ndarray) -> plt.Figure:
    """
    Create a horizontal bar chart visualization of feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance scores corresponding to features
        
    Returns:
        Matplotlib Figure object containing the feature importance bar chart
        
    Raises:
        ValueError: If feature_names and importances have different lengths
    """
    if feature_names is None or importances is None:
        raise ValueError("Inputs cannot be None")
    
    if len(feature_names) == 0 or len(importances) == 0:
        raise ValueError("Inputs cannot be empty")
    
    if len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must have the same length")
    
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    ax.barh(sorted_features, sorted_importances, color='steelblue')
    
    # Set labels and title
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
    
    plt.tight_layout()
    return fig
