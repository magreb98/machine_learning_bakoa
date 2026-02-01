"""
Input validation module for Credit Card Approval Prediction System.

This module handles validation of user inputs for predictions.
"""

from typing import Dict, List


def validate_prediction_input(input_dict: Dict) -> List[str]:
    """
    Validate prediction input fields.
    
    Args:
        input_dict: Dictionary containing all input fields
        
    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    
    # Check for required fields
    required_fields = ['age', 'owner', 'selfemp', 'dependents', 'months', 
                      'income', 'expenditure', 'share', 'reports', 'majorcards', 'active']
    
    for field in required_fields:
        if field not in input_dict or input_dict[field] is None:
            errors.append(f"Field '{field}' is required")
    
    # Validate data types and ranges
    if 'age' in input_dict and input_dict['age'] is not None:
        if not isinstance(input_dict['age'], (int, float)) or input_dict['age'] < 18 or input_dict['age'] > 100:
            errors.append("Age must be between 18 and 100")
    
    if 'income' in input_dict and input_dict['income'] is not None:
        if not isinstance(input_dict['income'], (int, float)) or input_dict['income'] < 0:
            errors.append("Income must be a non-negative number")
    
    if 'expenditure' in input_dict and input_dict['expenditure'] is not None:
        if not isinstance(input_dict['expenditure'], (int, float)) or input_dict['expenditure'] < 0:
            errors.append("Expenditure must be a non-negative number")
    
    if 'share' in input_dict and input_dict['share'] is not None:
        if not isinstance(input_dict['share'], (int, float)) or input_dict['share'] < 0 or input_dict['share'] > 1:
            errors.append("Share must be between 0 and 1")
    
    if 'reports' in input_dict and input_dict['reports'] is not None:
        if not isinstance(input_dict['reports'], int) or input_dict['reports'] < 0:
            errors.append("Reports must be a non-negative integer")
    
    if 'dependents' in input_dict and input_dict['dependents'] is not None:
        if not isinstance(input_dict['dependents'], int) or input_dict['dependents'] < 0:
            errors.append("Dependents must be a non-negative integer")
    
    if 'months' in input_dict and input_dict['months'] is not None:
        if not isinstance(input_dict['months'], int) or input_dict['months'] < 0:
            errors.append("Months must be a non-negative integer")
    
    if 'majorcards' in input_dict and input_dict['majorcards'] is not None:
        if not isinstance(input_dict['majorcards'], int) or input_dict['majorcards'] < 0:
            errors.append("Major cards must be a non-negative integer")
    
    if 'active' in input_dict and input_dict['active'] is not None:
        if not isinstance(input_dict['active'], int) or input_dict['active'] < 0:
            errors.append("Active accounts must be a non-negative integer")
    
    if 'owner' in input_dict and input_dict['owner'] is not None:
        if input_dict['owner'] not in ['yes', 'no']:
            errors.append("Owner must be 'yes' or 'no'")
    
    if 'selfemp' in input_dict and input_dict['selfemp'] is not None:
        if input_dict['selfemp'] not in ['yes', 'no']:
            errors.append("Self-employed must be 'yes' or 'no'")
    
    return errors
