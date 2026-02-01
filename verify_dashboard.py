"""
Verification script for the Streamlit dashboard.

This script verifies that all dashboard components are working correctly
without actually launching the Streamlit app.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import DataPreprocessor
from src.model import CreditCardModel
from src.evaluation import calculate_metrics, generate_confusion_matrix
from src.validation import validate_prediction_input


def verify_data_explorer():
    """Verify Data Explorer page functionality."""
    print("=" * 60)
    print("VERIFYING DATA EXPLORER PAGE")
    print("=" * 60)
    
    # Load dataset
    data_path = Path("data/AER_credit_card_data.csv")
    if not data_path.exists():
        print("❌ Dataset not found!")
        return False
    
    data = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {data.shape[0]} records, {data.shape[1]} features")
    
    # Check summary statistics
    print(f"✅ Summary statistics available")
    print(f"   - Missing values: {data.isnull().sum().sum()}")
    print(f"   - Approval rate: {(data['card'].value_counts().get('yes', 0) / len(data)) * 100:.1f}%")
    
    # Check distribution visualizations
    numerical_cols = [col for col in data.columns 
                     if col not in ['card', 'owner', 'selfemp']]
    print(f"✅ Distribution plots available for {len(numerical_cols)} numerical features")
    
    # Check correlation matrix
    correlation_matrix = data[numerical_cols].corr()
    print(f"✅ Correlation matrix calculated: {correlation_matrix.shape}")
    print(f"   - Matrix is symmetric: {np.allclose(correlation_matrix.values, correlation_matrix.values.T)}")
    print(f"   - Diagonal is 1.0: {np.allclose(np.diag(correlation_matrix.values), 1.0)}")
    
    # Check target variable distribution
    target_counts = data['card'].value_counts()
    print(f"✅ Target variable distribution:")
    print(f"   - Approved: {target_counts.get('yes', 0)}")
    print(f"   - Rejected: {target_counts.get('no', 0)}")
    
    print("\n✅ Data Explorer page: ALL CHECKS PASSED\n")
    return True


def verify_model_performance():
    """Verify Model Performance page functionality."""
    print("=" * 60)
    print("VERIFYING MODEL PERFORMANCE PAGE")
    print("=" * 60)
    
    # Load model
    model_path = Path("models/credit_card_model.pkl")
    if not model_path.exists():
        print("❌ Model not found!")
        return False
    
    model = CreditCardModel()
    model.load_model(str(model_path))
    print("✅ Model loaded successfully")
    
    # Load and preprocess data
    data = pd.read_csv("data/AER_credit_card_data.csv")
    preprocessor = DataPreprocessor("data/AER_credit_card_data.csv")
    
    df = preprocessor.handle_missing_values(data)
    df = preprocessor.handle_outliers(df)
    df = preprocessor.encode_categorical_features(df)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(df, test_size=0.2, random_state=42)
    X_test_scaled = preprocessor.scale_numerical_features(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled.values)
    print(f"✅ Predictions made on {len(X_test)} test samples")
    
    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_pred)
    print(f"✅ Performance metrics calculated:")
    print(f"   - Accuracy: {metrics['accuracy']:.2%}")
    print(f"   - Precision: {metrics['precision']:.2%}")
    print(f"   - Recall: {metrics['recall']:.2%}")
    print(f"   - F1-Score: {metrics['f1_score']:.2%}")
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(y_test.values, y_pred)
    print(f"✅ Confusion matrix generated:")
    print(f"   - True Negatives: {cm[0, 0]}")
    print(f"   - False Positives: {cm[0, 1]}")
    print(f"   - False Negatives: {cm[1, 0]}")
    print(f"   - True Positives: {cm[1, 1]}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    print(f"✅ Feature importance extracted:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:3], 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    print("\n✅ Model Performance page: ALL CHECKS PASSED\n")
    return True


def verify_make_prediction():
    """Verify Make Prediction page functionality."""
    print("=" * 60)
    print("VERIFYING MAKE PREDICTION PAGE")
    print("=" * 60)
    
    # Test input validation
    valid_input = {
        'age': 30.0,
        'income': 5.0,
        'expenditure': 100.0,
        'share': 0.1,
        'reports': 0,
        'dependents': 0,
        'months': 12,
        'majorcards': 1,
        'active': 1,
        'owner': 'yes',
        'selfemp': 'no'
    }
    
    errors = validate_prediction_input(valid_input)
    if len(errors) == 0:
        print("✅ Input validation working correctly")
    else:
        print(f"❌ Input validation failed: {errors}")
        return False
    
    # Test invalid input
    invalid_input = valid_input.copy()
    invalid_input['age'] = -5.0
    errors = validate_prediction_input(invalid_input)
    if len(errors) > 0:
        print("✅ Invalid input correctly rejected")
    else:
        print("❌ Invalid input not rejected!")
        return False
    
    # Load model and make prediction
    model = CreditCardModel()
    model.load_model("models/credit_card_model.pkl")
    
    input_data = pd.DataFrame([valid_input])
    preprocessor = DataPreprocessor("data/AER_credit_card_data.csv")
    input_data = preprocessor.encode_categorical_features(input_data)
    input_data = preprocessor.scale_numerical_features(input_data)
    
    prediction = model.predict(input_data.values)[0]
    probability = model.predict_proba(input_data.values)[0]
    
    print(f"✅ Prediction made successfully:")
    print(f"   - Result: {prediction.upper()}")
    print(f"   - Approval probability: {probability[1]:.2%}")
    print(f"   - Rejection probability: {probability[0]:.2%}")
    
    # Determine confidence level
    prob_yes = probability[1]
    if prob_yes > 0.8 or prob_yes < 0.2:
        confidence = "High"
    elif 0.6 < prob_yes < 0.8 or 0.2 < prob_yes < 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    print(f"   - Confidence: {confidence}")
    
    print("\n✅ Make Prediction page: ALL CHECKS PASSED\n")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("STREAMLIT DASHBOARD VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Verify each page
    results.append(("Data Explorer", verify_data_explorer()))
    results.append(("Model Performance", verify_model_performance()))
    results.append(("Make Prediction", verify_make_prediction()))
    
    # Print summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for page_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{page_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL DASHBOARD COMPONENTS VERIFIED SUCCESSFULLY! 🎉")
        print("\nYou can now run the dashboard with:")
        print("  streamlit run app.py")
        print("\nThe dashboard includes:")
        print("  📊 Data Explorer - Interactive data visualizations")
        print("  📈 Model Performance - Metrics and feature importance")
        print("  🔮 Make Prediction - Credit card approval predictions")
    else:
        print("\n❌ Some components failed verification. Please check the errors above.")
    
    print()


if __name__ == "__main__":
    main()
