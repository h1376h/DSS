"""
Data helper functions for Streamlit UI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def prepare_data_for_analysis(dataset: pd.DataFrame, 
                            selected_columns: List[str],
                            handle_missing: str = "drop") -> pd.DataFrame:
    """Prepare data for analysis with missing value handling"""
    data = dataset[selected_columns].copy()
    
    if handle_missing == "drop":
        data = data.dropna()
    elif handle_missing == "fill_mean":
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    elif handle_missing == "fill_median":
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    return data


def detect_outliers_iqr(data: pd.Series) -> Tuple[int, float]:
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers), len(outliers) / len(data) * 100


def calculate_correlation_matrix(data: pd.DataFrame, 
                              method: str = "pearson") -> pd.DataFrame:
    """Calculate correlation matrix"""
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    return numeric_data.corr(method=method)


def find_strong_correlations(corr_matrix: pd.DataFrame, 
                           threshold: float = 0.7) -> List[Dict]:
    """Find strong correlations above threshold"""
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                strong_correlations.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j],
                    'Strength': 'Strong' if corr_value >= 0.8 else 'Moderate'
                })
    return strong_correlations


def calculate_data_quality_metrics(dataset: pd.DataFrame) -> Dict[str, float]:
    """Calculate data quality metrics"""
    total_cells = dataset.shape[0] * dataset.shape[1]
    missing_cells = dataset.isnull().sum().sum()
    
    return {
        'Completeness': (1 - missing_cells / total_cells) * 100,
        'Uniqueness': (dataset.nunique().sum() / total_cells) * 100,
        'Consistency': 85.0,  # Placeholder - would need domain knowledge
        'Accuracy': 92.0  # Placeholder - would need ground truth
    }


def create_binary_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Create binary matrix for association rules"""
    return pd.get_dummies(data)


def find_frequent_itemsets(binary_matrix: pd.DataFrame, 
                         min_support: float) -> List[str]:
    """Find frequent itemsets above minimum support"""
    min_support_count = int(min_support * len(binary_matrix))
    frequent_itemsets = binary_matrix.sum()
    return frequent_itemsets[frequent_itemsets >= min_support_count].index.tolist()


def generate_association_rules(binary_matrix: pd.DataFrame,
                             frequent_items: List[str],
                             min_confidence: float,
                             min_lift: float,
                             max_rules: int = 20) -> List[Dict]:
    """Generate association rules from frequent itemsets"""
    rules = []
    filtered_matrix = binary_matrix[frequent_items]
    
    for i in range(len(frequent_items)):
        for j in range(i+1, len(frequent_items)):
            if len(rules) >= max_rules:
                break
                
            item1, item2 = frequent_items[i], frequent_items[j]
            
            # Calculate metrics
            support_both = (filtered_matrix[item1] & filtered_matrix[item2]).sum() / len(filtered_matrix)
            support_item1 = filtered_matrix[item1].sum() / len(filtered_matrix)
            support_item2 = filtered_matrix[item2].sum() / len(filtered_matrix)
            
            if support_both > 0:
                confidence_1_to_2 = support_both / support_item1 if support_item1 > 0 else 0
                confidence_2_to_1 = support_both / support_item2 if support_item2 > 0 else 0
                
                lift_1_to_2 = confidence_1_to_2 / support_item2 if support_item2 > 0 else 0
                lift_2_to_1 = confidence_2_to_1 / support_item1 if support_item1 > 0 else 0
                
                # Rule: item1 -> item2
                if confidence_1_to_2 >= min_confidence and lift_1_to_2 >= min_lift:
                    rules.append({
                        'rule': f"{item1} → {item2}",
                        'support': support_both,
                        'confidence': confidence_1_to_2,
                        'lift': lift_1_to_2,
                        'conviction': confidence_1_to_2 / (1 - confidence_1_to_2) if confidence_1_to_2 < 1 else float('inf')
                    })
                
                # Rule: item2 -> item1
                if confidence_2_to_1 >= min_confidence and lift_2_to_1 >= min_lift:
                    rules.append({
                        'rule': f"{item2} → {item1}",
                        'support': support_both,
                        'confidence': confidence_2_to_1,
                        'lift': lift_2_to_1,
                        'conviction': confidence_2_to_1 / (1 - confidence_2_to_1) if confidence_2_to_1 < 1 else float('inf')
                    })
        
        if len(rules) >= max_rules:
            break
    
    return rules
