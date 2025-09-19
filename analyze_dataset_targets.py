#!/usr/bin/env python3
"""
Dataset Target Analysis Script
Analyzes all datasets to identify target variables and their characteristics
for smarter DSS functionalities
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class DatasetTargetAnalyzer:
    """Analyzes datasets to identify target variables and their characteristics"""
    
    def __init__(self, datasets_dir: str = "datasets/raw"):
        self.datasets_dir = datasets_dir
        self.target_analysis = {}
        
    def analyze_dataset(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single dataset to identify target variables"""
        try:
            df = pd.read_csv(filepath)
            filename = os.path.basename(filepath)
            
            analysis = {
                'filename': filename,
                'shape': df.shape,
                'columns': list(df.columns),
                'target_variables': [],
                'feature_variables': [],
                'data_types': {},
                'target_characteristics': {},
                'dataset_type': 'unknown',
                'ml_task_type': 'unknown'
            }
            
            # Analyze data types
            for col in df.columns:
                analysis['data_types'][col] = str(df[col].dtype)
            
            # Identify potential target variables based on common patterns
            target_candidates = self._identify_target_candidates(df)
            
            for candidate in target_candidates:
                target_info = self._analyze_target_variable(df, candidate)
                analysis['target_variables'].append(target_info)
            
            # Identify feature variables (non-targets)
            target_cols = [t['column'] for t in analysis['target_variables']]
            analysis['feature_variables'] = [col for col in df.columns if col not in target_cols]
            
            # Determine dataset type and ML task type
            analysis['dataset_type'] = self._determine_dataset_type(df, analysis['target_variables'])
            analysis['ml_task_type'] = self._determine_ml_task_type(analysis['target_variables'])
            
            return analysis
            
        except Exception as e:
            return {
                'filename': os.path.basename(filepath),
                'error': str(e),
                'target_variables': [],
                'feature_variables': [],
                'dataset_type': 'error',
                'ml_task_type': 'error'
            }
    
    def _identify_target_candidates(self, df: pd.DataFrame) -> List[str]:
        """Identify potential target variable candidates"""
        candidates = []
        
        # Common target variable names
        target_keywords = [
            'target', 'label', 'outcome', 'result', 'prediction', 'class',
            'diagnosis', 'success', 'failure', 'risk', 'score', 'rating',
            'effectiveness', 'satisfaction', 'performance', 'quality',
            'progression', 'survival', 'mortality', 'readmission',
            'complication', 'adherence', 'capacity_value', 'performance_rating'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Direct keyword match
            if any(keyword in col_lower for keyword in target_keywords):
                candidates.append(col)
                continue
            
            # Check if column contains categorical/binary data (potential classification target)
            if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10:
                candidates.append(col)
                continue
            
            # Check if column is numeric and could be regression target
            if pd.api.types.is_numeric_dtype(df[col]):
                # Skip obvious feature columns
                if any(skip_word in col_lower for skip_word in ['id', 'date', 'time', 'age', 'count', 'volume']):
                    continue
                candidates.append(col)
        
        return candidates
    
    def _analyze_target_variable(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze characteristics of a target variable"""
        target_info = {
            'column': column,
            'data_type': str(df[column].dtype),
            'unique_values': df[column].nunique(),
            'missing_values': df[column].isnull().sum(),
            'value_counts': df[column].value_counts().head(10).to_dict(),
            'statistics': {},
            'target_type': 'unknown',
            'is_primary_target': False
        }
        
        # Calculate statistics for numeric targets
        if pd.api.types.is_numeric_dtype(df[column]):
            target_info['statistics'] = {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'median': float(df[column].median())
            }
        
        # Determine target type
        if df[column].nunique() == 2:
            target_info['target_type'] = 'binary_classification'
        elif df[column].nunique() <= 10 and df[column].dtype in ['object', 'category']:
            target_info['target_type'] = 'multiclass_classification'
        elif pd.api.types.is_numeric_dtype(df[column]):
            target_info['target_type'] = 'regression'
        else:
            target_info['target_type'] = 'other'
        
        # Determine if this is likely the primary target
        target_info['is_primary_target'] = self._is_primary_target(df, column)
        
        return target_info
    
    def _is_primary_target(self, df: pd.DataFrame, column: str) -> bool:
        """Determine if this column is likely the primary target variable"""
        col_lower = column.lower()
        
        # High priority indicators
        if any(keyword in col_lower for keyword in ['target', 'label', 'outcome', 'diagnosis']):
            return True
        
        # Medium priority indicators
        if any(keyword in col_lower for keyword in ['success', 'risk', 'score', 'rating', 'effectiveness']):
            return True
        
        # Check if it's the only categorical/binary column
        categorical_cols = [col for col in df.columns if df[col].nunique() <= 10 or df[col].dtype in ['object', 'category']]
        if len(categorical_cols) == 1 and column in categorical_cols:
            return True
        
        return False
    
    def _determine_dataset_type(self, df: pd.DataFrame, targets: List[Dict]) -> str:
        """Determine the type of dataset based on its characteristics"""
        if not targets:
            return 'unsupervised'
        
        # Check for healthcare-specific indicators
        healthcare_keywords = [
            'patient', 'medical', 'clinical', 'diagnosis', 'treatment',
            'hospital', 'health', 'disease', 'cancer', 'diabetes',
            'medication', 'surgery', 'cardiology', 'oncology'
        ]
        
        all_columns = ' '.join(df.columns).lower()
        if any(keyword in all_columns for keyword in healthcare_keywords):
            return 'healthcare'
        
        # Check for financial/business indicators
        business_keywords = ['revenue', 'cost', 'expense', 'financial', 'performance', 'staff']
        if any(keyword in all_columns for keyword in business_keywords):
            return 'business'
        
        return 'general'
    
    def _determine_ml_task_type(self, targets: List[Dict]) -> str:
        """Determine the ML task type based on target variables"""
        if not targets:
            return 'unsupervised'
        
        primary_targets = [t for t in targets if t.get('is_primary_target', False)]
        if not primary_targets:
            primary_targets = targets[:1]  # Use first target if no primary identified
        
        target_types = [t['target_type'] for t in primary_targets]
        
        if 'binary_classification' in target_types:
            return 'binary_classification'
        elif 'multiclass_classification' in target_types:
            return 'multiclass_classification'
        elif 'regression' in target_types:
            return 'regression'
        else:
            return 'mixed'
    
    def analyze_all_datasets(self) -> Dict[str, Any]:
        """Analyze all datasets in the directory"""
        if not os.path.exists(self.datasets_dir):
            print(f"Directory {self.datasets_dir} does not exist!")
            return {}
        
        csv_files = [f for f in os.listdir(self.datasets_dir) if f.endswith('.csv')]
        
        print(f"Found {len(csv_files)} CSV files to analyze:")
        for file in csv_files:
            print(f"  - {file}")
        print()
        
        for filename in csv_files:
            filepath = os.path.join(self.datasets_dir, filename)
            print(f"Analyzing {filename}...")
            
            analysis = self.analyze_dataset(filepath)
            self.target_analysis[filename] = analysis
            
            if 'error' in analysis:
                print(f"  ✗ Error: {analysis['error']}")
            else:
                print(f"  ✓ Shape: {analysis['shape']}")
                print(f"  ✓ Target variables: {len(analysis['target_variables'])}")
                print(f"  ✓ Dataset type: {analysis['dataset_type']}")
                print(f"  ✓ ML task type: {analysis['ml_task_type']}")
                
                for target in analysis['target_variables']:
                    if target.get('is_primary_target', False):
                        print(f"    → Primary target: {target['column']} ({target['target_type']})")
            print()
        
        return self.target_analysis
    
    def generate_target_config(self) -> Dict[str, Any]:
        """Generate configuration for target variables"""
        config = {
            'dataset_targets': {},
            'summary': {
                'total_datasets': len(self.target_analysis),
                'healthcare_datasets': 0,
                'business_datasets': 0,
                'general_datasets': 0,
                'classification_datasets': 0,
                'regression_datasets': 0,
                'mixed_datasets': 0
            },
            'recommendations': []
        }
        
        for filename, analysis in self.target_analysis.items():
            if 'error' in analysis:
                continue
            
            dataset_config = {
                'filename': filename,
                'dataset_type': analysis['dataset_type'],
                'ml_task_type': analysis['ml_task_type'],
                'shape': analysis['shape'],
                'primary_targets': [],
                'secondary_targets': [],
                'feature_variables': analysis['feature_variables'],
                'recommended_models': [],
                'preprocessing_needed': []
            }
            
            # Categorize targets
            for target in analysis['target_variables']:
                target_config = {
                    'column': target['column'],
                    'target_type': target['target_type'],
                    'data_type': target['data_type'],
                    'unique_values': target['unique_values'],
                    'missing_values': target['missing_values']
                }
                
                if target.get('is_primary_target', False):
                    dataset_config['primary_targets'].append(target_config)
                else:
                    dataset_config['secondary_targets'].append(target_config)
            
            # Generate recommendations
            self._generate_recommendations(dataset_config, analysis)
            
            config['dataset_targets'][filename] = dataset_config
            
            # Update summary
            config['summary'][f'{analysis["dataset_type"]}_datasets'] += 1
            
            # Handle different ML task types
            ml_task_type = analysis["ml_task_type"]
            if ml_task_type == 'binary_classification':
                config['summary']['classification_datasets'] += 1
            elif ml_task_type == 'multiclass_classification':
                config['summary']['classification_datasets'] += 1
            elif ml_task_type == 'regression':
                config['summary']['regression_datasets'] += 1
            elif ml_task_type == 'mixed':
                config['summary']['mixed_datasets'] += 1
        
        # Generate global recommendations
        self._generate_global_recommendations(config)
        
        return config
    
    def _generate_recommendations(self, dataset_config: Dict, analysis: Dict):
        """Generate recommendations for a specific dataset"""
        recommendations = []
        
        # Model recommendations based on task type
        if analysis['ml_task_type'] == 'binary_classification':
            dataset_config['recommended_models'] = [
                'LogisticRegression', 'RandomForestClassifier', 'XGBClassifier',
                'SVM', 'NaiveBayes'
            ]
        elif analysis['ml_task_type'] == 'multiclass_classification':
            dataset_config['recommended_models'] = [
                'RandomForestClassifier', 'XGBClassifier', 'SVM',
                'LogisticRegression', 'KNeighborsClassifier'
            ]
        elif analysis['ml_task_type'] == 'regression':
            dataset_config['recommended_models'] = [
                'RandomForestRegressor', 'XGBRegressor', 'LinearRegression',
                'SVR', 'Ridge', 'Lasso'
            ]
        
        # Preprocessing recommendations
        preprocessing = []
        for target in analysis['target_variables']:
            if target['missing_values'] > 0:
                preprocessing.append(f"Handle missing values in {target['column']}")
            
            if target['target_type'] == 'binary_classification':
                preprocessing.append(f"Ensure binary encoding for {target['column']}")
            elif target['target_type'] == 'multiclass_classification':
                preprocessing.append(f"Apply label encoding for {target['column']}")
        
        dataset_config['preprocessing_needed'] = preprocessing
    
    def _generate_global_recommendations(self, config: Dict):
        """Generate global recommendations"""
        recommendations = []
        
        # Dataset type recommendations
        if config['summary']['healthcare_datasets'] > 0:
            recommendations.append({
                'type': 'healthcare',
                'message': 'Healthcare datasets detected. Consider implementing clinical decision support features.',
                'suggestions': [
                    'Implement risk stratification models',
                    'Add clinical outcome prediction',
                    'Create patient monitoring dashboards',
                    'Implement treatment recommendation systems'
                ]
            })
        
        if config['summary']['business_datasets'] > 0:
            recommendations.append({
                'type': 'business',
                'message': 'Business datasets detected. Consider implementing operational analytics.',
                'suggestions': [
                    'Implement performance monitoring',
                    'Add cost optimization models',
                    'Create staff efficiency analytics',
                    'Implement resource allocation optimization'
                ]
            })
        
        # Task type recommendations
        if config['summary']['classification_datasets'] > 0:
            recommendations.append({
                'type': 'classification',
                'message': 'Classification datasets available. Implement predictive classification features.',
                'suggestions': [
                    'Add model evaluation metrics (accuracy, precision, recall, F1)',
                    'Implement confusion matrix visualization',
                    'Create prediction confidence scoring',
                    'Add feature importance analysis'
                ]
            })
        
        if config['summary']['regression_datasets'] > 0:
            recommendations.append({
                'type': 'regression',
                'message': 'Regression datasets available. Implement predictive regression features.',
                'suggestions': [
                    'Add model evaluation metrics (RMSE, MAE, R²)',
                    'Implement residual analysis',
                    'Create prediction interval estimation',
                    'Add feature importance analysis'
                ]
            })
        
        config['recommendations'] = recommendations
    
    def save_analysis(self, output_file: str = "dataset_target_analysis.json"):
        """Save the analysis results to a JSON file"""
        config = self.generate_target_config()
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Analysis saved to {output_file}")
        return config

def main():
    """Main function to run the analysis"""
    print("=" * 80)
    print("DATASET TARGET ANALYSIS FOR HEALTHCARE DSS")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = DatasetTargetAnalyzer()
    
    # Analyze all datasets
    analysis_results = analyzer.analyze_all_datasets()
    
    if not analysis_results:
        print("No datasets found to analyze!")
        return
    
    # Generate and save configuration
    config = analyzer.save_analysis()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total datasets analyzed: {config['summary']['total_datasets']}")
    print(f"Healthcare datasets: {config['summary']['healthcare_datasets']}")
    print(f"Business datasets: {config['summary']['business_datasets']}")
    print(f"General datasets: {config['summary']['general_datasets']}")
    print()
    print(f"Classification datasets: {config['summary']['classification_datasets']}")
    print(f"Regression datasets: {config['summary']['regression_datasets']}")
    print(f"Mixed datasets: {config['summary']['mixed_datasets']}")
    print()
    
    # Print recommendations
    if config['recommendations']:
        print("RECOMMENDATIONS:")
        for rec in config['recommendations']:
            print(f"\n{rec['type'].upper()}: {rec['message']}")
            for suggestion in rec['suggestions']:
                print(f"  • {suggestion}")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check 'dataset_target_analysis.json' for detailed results.")
    print("=" * 80)

if __name__ == "__main__":
    main()
