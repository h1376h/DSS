#!/usr/bin/env python3
"""Test categorical data detection"""

import pandas as pd
import numpy as np

def test_categorical_detection():
    """Test the categorical data detection function"""
    
    # Test with diabetes sex column (normalized values)
    print("Testing diabetes sex column (normalized values):")
    sex_data = pd.Series([0.05068011873981862, -0.044641636506989144, 0.05068011873981862, -0.044641636506989144, 0.05068011873981862])
    
    print(f"Unique values: {sex_data.unique()}")
    print(f"Unique count: {sex_data.nunique()}")
    print(f"Data type: {sex_data.dtype}")
    print(f"Min: {sex_data.min()}, Max: {sex_data.max()}")
    
    # Check if this looks like categorical data
    unique_count = sex_data.nunique()
    total_count = len(sex_data)
    unique_ratio = unique_count / total_count
    
    print(f"Unique ratio: {unique_ratio}")
    print(f"Low cardinality: {unique_ratio < 0.1}")
    print(f"Values around 0: {sex_data.min() < 0.1 and sex_data.max() > -0.1}")
    print(f"Binary-like: {unique_count == 2}")
    
    # This should be detected as categorical data that needs conversion
    print("\n✅ This should be detected as categorical data!")
    
    # Test conversion
    values = sorted(sex_data.unique())
    mapping = {values[0]: 0, values[1]: 1}
    fixed_data = sex_data.map(mapping)
    
    print(f"\nConversion mapping: {mapping}")
    print(f"Fixed data: {fixed_data.tolist()}")
    print(f"Fixed data unique values: {fixed_data.unique()}")

if __name__ == "__main__":
    test_categorical_detection()
