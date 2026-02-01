"""
Training script for Credit Card Approval Prediction System.

This script loads data, preprocesses it, trains a Decision Tree model with
hyperparameter tuning, evaluates performance, and saves the trained model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import DataPreprocessor
from src.model import CreditCardModel
from src.evaluation import calculate_metrics, generate_confusion_matrix, plot_confusion_matrix, plot_feature_importance


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Credit Card Approval Prediction - Model Training")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    data_path = "data/AER_credit_card_data.csv"
    preprocessor = DataPreprocessor(data_path)
    
    # Load data
    df = preprocessor.load_data()
    print(f"   [OK] Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Validate data
    preprocessor.validate_columns(df)
    print("   [OK] All expected columns present")
    
    preprocessor.validate_target(df)
    print("   [OK] Target variable validated")
    
    # Check for missing values
    missing_values = preprocessor.identify_missing_values(df)
    total_missing = sum(missing_values.values())
    print(f"   [OK] Identified {total_missing} missing values")
    
    # Preprocess data
    df = preprocessor.handle_missing_values(df)
    print("   [OK] Missing values imputed")
    
    df = preprocessor.handle_outliers(df)
    print("   [OK] Outliers handled")
    
    df = preprocessor.encode_categorical_features(df)
    print("   [OK] Categorical features encoded")
    
    # Split data before scaling
    X_train, X_test, y_train, y_test = preprocessor.split_data(df, test_size=0.2, random_state=42)
    print(f"   [OK] Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Scale features
    X_train_scaled = preprocessor.scale_numerical_features(X_train)
    X_test_scaled = preprocessor.scale_numerical_features(X_test)
    print("   [OK] Features scaled")
    
    # 2. Train model with hyperparameter tuning
    print("\n[2/5] Training model with hyperparameter tuning...")
    model = CreditCardModel()
    
    # Store feature names
    feature_names = list(X_train_scaled.columns)
    
    # Tune hyperparameters
    best_params = model.tune_hyperparameters(
        X_train_scaled.values,
        y_train.values,
        cv=5
    )
    print(f"   [OK] Best hyperparameters found:")
    for param, value in best_params.items():
        print(f"     - {param}: {value}")
    
    # Store feature names in model
    model.feature_names = feature_names
    
    # 3. Evaluate model
    print("\n[3/5] Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled.values)
    y_test_pred = model.predict(X_test_scaled.values)
    
    # Calculate metrics for training set
    train_metrics = calculate_metrics(y_train.values, y_train_pred)
    print("   Training Set Metrics:")
    print(f"     - Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"     - Precision: {train_metrics['precision']:.4f}")
    print(f"     - Recall:    {train_metrics['recall']:.4f}")
    print(f"     - F1-Score:  {train_metrics['f1_score']:.4f}")
    
    # Calculate metrics for test set
    test_metrics = calculate_metrics(y_test.values, y_test_pred)
    print("\n   Test Set Metrics:")
    print(f"     - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"     - Precision: {test_metrics['precision']:.4f}")
    print(f"     - Recall:    {test_metrics['recall']:.4f}")
    print(f"     - F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(y_test.values, y_test_pred)
    print("\n   Confusion Matrix:")
    print(f"     {cm}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    print("\n   Top 5 Most Important Features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        print(f"     {i}. {feature}: {importance:.4f}")
    
    # 4. Save model
    print("\n[4/5] Saving trained model...")
    model_path = Path("models/credit_card_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"   [OK] Model saved to {model_path}")
    
    # 5. Save visualizations
    print("\n[5/5] Generating and saving visualizations...")
    
    # Save confusion matrix plot
    cm_fig = plot_confusion_matrix(cm)
    cm_path = Path("models/confusion_matrix.png")
    cm_fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Confusion matrix saved to {cm_path}")
    
    # Save feature importance plot
    fi_fig = plot_feature_importance(feature_names, np.array([feature_importance[f] for f in feature_names]))
    fi_path = Path("models/feature_importance.png")
    fi_fig.savefig(fi_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Feature importance saved to {fi_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
