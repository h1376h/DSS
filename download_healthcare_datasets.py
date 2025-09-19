#!/usr/bin/env python3
"""
Healthcare DSS Dataset Downloader
Downloads healthcare datasets from Python libraries and saves them as CSV files
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_linnerud, load_breast_cancer, load_wine
import seaborn as sns
import os

def create_datasets_directory():
    """Create datasets directory if it doesn't exist"""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"Created directory: {datasets_dir}")
    return datasets_dir

def download_diabetes_dataset(datasets_dir):
    """Download and save diabetes dataset from scikit-learn"""
    print("Downloading Diabetes dataset...")
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    
    # Create DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Add description
    df.attrs['description'] = diabetes.DESCR
    
    # Save to CSV
    filepath = os.path.join(datasets_dir, "diabetes_dataset.csv")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Diabetes dataset saved to: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {list(diabetes.feature_names)}")
    print(f"  Target: Disease progression (quantitative measure)")
    print()
    
    return df

def download_breast_cancer_dataset(datasets_dir):
    """Download and save breast cancer dataset from scikit-learn"""
    print("Downloading Breast Cancer dataset...")
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    df['target_name'] = [cancer.target_names[i] for i in cancer.target]
    
    # Add description
    df.attrs['description'] = cancer.DESCR
    
    # Save to CSV
    filepath = os.path.join(datasets_dir, "breast_cancer_dataset.csv")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Breast Cancer dataset saved to: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len(cancer.feature_names)} features (mean, se, worst values)")
    print(f"  Target: {cancer.target_names}")
    print()
    
    return df

def download_heart_disease_dataset(datasets_dir):
    """Download and save heart disease dataset from seaborn"""
    print("Downloading Heart Disease dataset...")
    
    try:
        # Load heart disease dataset from seaborn
        df = sns.load_dataset('heart')
        
        # Save to CSV
        filepath = os.path.join(datasets_dir, "heart_disease_dataset.csv")
        df.to_csv(filepath, index=False)
        
        print(f"✓ Heart Disease dataset saved to: {filepath}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()
        
        return df
        
    except Exception as e:
        print(f"✗ Could not load heart disease dataset: {e}")
        print("  Note: This dataset might not be available in your seaborn version")
        print()
        return None

def download_wine_dataset(datasets_dir):
    """Download and save wine dataset from scikit-learn (for medical analysis)"""
    print("Downloading Wine dataset...")
    
    # Load wine dataset
    wine = load_wine()
    
    # Create DataFrame
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['target_name'] = [wine.target_names[i] for i in wine.target]
    
    # Add description
    df.attrs['description'] = wine.DESCR
    
    # Save to CSV
    filepath = os.path.join(datasets_dir, "wine_dataset.csv")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Wine dataset saved to: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {len(wine.feature_names)} chemical properties")
    print(f"  Target: {wine.target_names}")
    print()
    
    return df

def download_linnerud_dataset(datasets_dir):
    """Download and save Linnerud dataset from scikit-learn"""
    print("Downloading Linnerud dataset...")
    
    # Load Linnerud dataset
    linnerud = load_linnerud()
    
    # Create DataFrame
    df = pd.DataFrame(linnerud.data, columns=linnerud.feature_names)
    
    # Add target variables
    target_df = pd.DataFrame(linnerud.target, columns=linnerud.target_names)
    df = pd.concat([df, target_df], axis=1)
    
    # Add description
    df.attrs['description'] = linnerud.DESCR
    
    # Save to CSV
    filepath = os.path.join(datasets_dir, "linnerud_dataset.csv")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Linnerud dataset saved to: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {linnerud.feature_names}")
    print(f"  Targets: {linnerud.target_names}")
    print()
    
    return df

def create_dataset_summary(datasets_dir, datasets):
    """Create a summary of all downloaded datasets"""
    print("Creating dataset summary...")
    
    summary = {
        'Dataset': [],
        'Shape': [],
        'Features': [],
        'Target': [],
        'Description': []
    }
    
    for name, df in datasets.items():
        if df is not None:
            summary['Dataset'].append(name)
            summary['Shape'].append(f"{df.shape[0]} rows, {df.shape[1]} columns")
            summary['Features'].append(df.shape[1] - 1)  # Exclude target column
            summary['Target'].append("Yes" if 'target' in df.columns else "Multiple targets")
            summary['Description'].append("Healthcare/Medical dataset")
    
    summary_df = pd.DataFrame(summary)
    
    # Save summary
    filepath = os.path.join(datasets_dir, "dataset_summary.csv")
    summary_df.to_csv(filepath, index=False)
    
    print(f"✓ Dataset summary saved to: {filepath}")
    print()
    
    return summary_df

def main():
    """Main function to download all healthcare datasets"""
    print("=" * 60)
    print("HEALTHCARE DSS DATASET DOWNLOADER")
    print("=" * 60)
    print()
    
    # Create datasets directory
    datasets_dir = create_datasets_directory()
    
    # Download datasets
    datasets = {}
    
    # 1. Diabetes dataset (perfect for DSS - disease progression prediction)
    datasets['diabetes'] = download_diabetes_dataset(datasets_dir)
    
    # 2. Breast Cancer dataset (excellent for medical diagnosis DSS)
    datasets['breast_cancer'] = download_breast_cancer_dataset(datasets_dir)
    
    # 3. Heart Disease dataset (if available)
    datasets['heart_disease'] = download_heart_disease_dataset(datasets_dir)
    
    # 4. Wine dataset (useful for medical analysis - chemical properties)
    datasets['wine'] = download_wine_dataset(datasets_dir)
    
    # 5. Linnerud dataset (physiological measurements)
    datasets['linnerud'] = download_linnerud_dataset(datasets_dir)
    
    # Create summary
    summary_df = create_dataset_summary(datasets_dir, datasets)
    
    # Print final summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total datasets downloaded: {len([d for d in datasets.values() if d is not None])}")
    print(f"Datasets directory: {datasets_dir}")
    print()
    print("Available datasets:")
    for name, df in datasets.items():
        if df is not None:
            print(f"  ✓ {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  ✗ {name}: Not available")
    print()
    print("These datasets are perfect for your Healthcare DSS project!")
    print("They can be used for:")
    print("  - Disease prediction and diagnosis")
    print("  - Patient outcome analysis")
    print("  - Resource allocation optimization")
    print("  - Capacity planning models")
    print("  - Clinical decision support systems")

if __name__ == "__main__":
    main()
