"""
Association Rules Mining Module for Healthcare DSS

This module implements association rules mining to discover patterns and relationships
in healthcare data, such as drug interactions, symptom combinations, and treatment protocols.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AssociationRulesMiner:
    """
    Association Rules Mining for Healthcare Data
    
    Discovers frequent patterns and association rules in healthcare datasets
    to identify relationships between symptoms, treatments, medications, etc.
    """
    
    def __init__(self, data_manager):
        """
        Initialize Association Rules Miner
        
        Args:
            data_manager: DataManager instance with loaded datasets (includes all dataset functionality)
        """
        self.data_manager = data_manager
        self.frequent_itemsets = None
        self.association_rules = None
        self.transaction_data = None
    
    def _get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get dataset from the consolidated DataManager"""
        if hasattr(self.data_manager, 'datasets') and dataset_name in self.data_manager.datasets:
            return self.data_manager.datasets[dataset_name]
        else:
            logger.warning(f"Dataset '{dataset_name}' not found in DataManager")
            return pd.DataFrame()
        
    def prepare_transaction_data(self, dataset_name: str, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Prepare transaction data for association rules mining
        
        Args:
            dataset_name: Name of the dataset to analyze
            categorical_columns: List of categorical columns to use for transactions
            
        Returns:
            DataFrame with transaction data
        """
        if dataset_name not in self.data_manager.datasets and (not self.dataset_manager or dataset_name not in self.dataset_manager.datasets):
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self._get_dataset(dataset_name).copy()
        
        # Create transactions from categorical data
        transactions = []
        
        for _, row in df.iterrows():
            transaction = []
            for col in categorical_columns:
                if col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        # Create item as column_name=value
                        item = f"{col}={value}"
                        transaction.append(item)
            if transaction:  # Only add non-empty transactions
                transactions.append(transaction)
        
        # Convert to transaction format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        self.transaction_data = df_transactions
        logger.info(f"Prepared {len(transactions)} transactions with {len(te.columns_)} items")
        
        return df_transactions
    
    def mine_frequent_itemsets(self, min_support: float = 0.1, max_len: int = 3) -> pd.DataFrame:
        """
        Mine frequent itemsets using Apriori algorithm
        
        Args:
            min_support: Minimum support threshold
            max_len: Maximum length of itemsets
            
        Returns:
            DataFrame with frequent itemsets
        """
        if self.transaction_data is None:
            raise ValueError("Transaction data not prepared. Call prepare_transaction_data() first.")
        
        logger.info(f"Mining frequent itemsets with min_support={min_support}, max_len={max_len}")
        
        # Mine frequent itemsets
        self.frequent_itemsets = apriori(
            self.transaction_data, 
            min_support=min_support, 
            use_colnames=True, 
            max_len=max_len
        )
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        return self.frequent_itemsets
    
    def generate_association_rules(self, metric: str = "confidence", min_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets
        
        Args:
            metric: Metric to use for rule generation (confidence, lift, conviction)
            min_threshold: Minimum threshold for the metric
            
        Returns:
            DataFrame with association rules
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            raise ValueError("No frequent itemsets found. Call mine_frequent_itemsets() first.")
        
        logger.info(f"Generating association rules with {metric} >= {min_threshold}")
        
        # Validate frequent itemsets to avoid division by zero
        if len(self.frequent_itemsets) < 2:
            logger.warning("Not enough frequent itemsets to generate meaningful association rules")
            self.association_rules = pd.DataFrame()
            return self.association_rules
        
        # Check for sufficient support values to avoid division by zero
        min_support = self.frequent_itemsets['support'].min()
        if min_support <= 0:
            logger.warning("Zero support values detected, adjusting threshold")
            min_threshold = max(min_threshold, 0.01)  # Ensure minimum threshold
        
        # Generate association rules with safer parameters
        try:
            # Suppress numpy warnings for division by zero
            import numpy as np
            with np.errstate(divide='ignore', invalid='ignore'):
                self.association_rules = association_rules(
                    self.frequent_itemsets, 
                    metric=metric, 
                    min_threshold=min_threshold,
                    support_only=False
                )
        except Exception as e:
            logger.warning(f"Error generating association rules: {e}")
            # Fallback to basic rules with higher threshold
            with np.errstate(divide='ignore', invalid='ignore'):
                self.association_rules = association_rules(
                    self.frequent_itemsets, 
                    metric="support", 
                    min_threshold=0.1,
                    support_only=True
                )
        
        if len(self.association_rules) > 0:
            # Sort by confidence and support
            self.association_rules = self.association_rules.sort_values(
                ['confidence', 'support'], ascending=[False, False]
            )
            logger.info(f"Generated {len(self.association_rules)} association rules")
        else:
            logger.warning("No association rules found with the given threshold")
        
        return self.association_rules
    
    def analyze_healthcare_patterns(self, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze healthcare patterns using association rules mining
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing healthcare patterns in {dataset_name}")
        
        # Get the dataset
        df = self._get_dataset(dataset_name).copy()
        
        # Create categorical features dynamically for any dataset
        categorical_columns = []
        
        # Process numeric columns by creating categories
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and len(df[col].dropna()) > 0:
                try:
                    # Create 3-4 categories based on quartiles
                    if df[col].nunique() > 10:  # Only categorize if enough unique values
                        col_categories = pd.qcut(df[col], q=4, duplicates='drop', labels=False)
                        if len(col_categories.dropna()) > 0:
                            df[f'{col}_category'] = col_categories
                            categorical_columns.append(f'{col}_category')
                except Exception as e:
                    logger.warning(f"Could not categorize column {col}: {str(e)}")
        
        # Process categorical/object columns
        object_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in object_columns:
            if col in df.columns and df[col].nunique() <= 20:  # Only use columns with reasonable number of categories
                categorical_columns.append(col)
        
        # If no categorical columns were created, create some basic ones
        if not categorical_columns:
            logger.warning(f"No suitable categorical columns found for {dataset_name}, creating basic categories")
            
            # Try to create categories from the first few numeric columns
            for i, col in enumerate(numeric_columns[:3]):
                try:
                    if df[col].nunique() > 5:
                        df[f'{col}_level'] = pd.qcut(df[col], q=3, duplicates='drop', labels=['Low', 'Medium', 'High'])
                        categorical_columns.append(f'{col}_level')
                except Exception as e:
                    logger.warning(f"Could not create basic category for {col}: {str(e)}")
        
        if not categorical_columns:
            raise ValueError(f"Cannot create categorical features for dataset: {dataset_name}. Dataset may not be suitable for association rules mining.")
        
        logger.info(f"Created {len(categorical_columns)} categorical features: {categorical_columns}")
        
        # Temporarily store the modified dataset
        original_dataset = self._get_dataset(dataset_name)
        
        # Store the modified dataset in the appropriate manager
        if hasattr(self.data_manager, 'datasets') and dataset_name in self.data_manager.datasets:
            self.data_manager.datasets[dataset_name] = df
        
        try:
            # Prepare transaction data
            self.prepare_transaction_data(dataset_name, categorical_columns)
            
            # Mine frequent itemsets
            frequent_itemsets = self.mine_frequent_itemsets(min_support=0.1, max_len=3)
            
            # Generate association rules
            rules = self.generate_association_rules(metric="confidence", min_threshold=0.6)
            
            # Analyze results
            analysis_results = {
                'dataset': dataset_name,
                'total_transactions': len(self.transaction_data),
                'total_items': len(self.transaction_data.columns),
                'frequent_itemsets_count': len(frequent_itemsets),
                'association_rules_count': len(rules),
                'top_itemsets': frequent_itemsets.head(10).to_dict('records') if len(frequent_itemsets) > 0 else [],
                'top_rules': rules.head(10).to_dict('records') if len(rules) > 0 else []
            }
            
        finally:
            # Restore original dataset
            if hasattr(self.data_manager, 'datasets') and dataset_name in self.data_manager.datasets:
                self.data_manager.datasets[dataset_name] = original_dataset
        
        return analysis_results
    
    def get_insights(self) -> List[str]:
        """
        Generate insights from discovered association rules
        
        Returns:
            List of insight strings
        """
        if self.association_rules is None or len(self.association_rules) == 0:
            return ["No association rules found to generate insights from."]
        
        insights = []
        
        # Get top rules
        top_rules = self.association_rules.head(5)
        
        for _, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            confidence = rule['confidence']
            support = rule['support']
            lift = rule['lift']
            
            insight = f"When {antecedents} occurs, {consequents} is likely to occur with {confidence:.1%} confidence (support: {support:.1%}, lift: {lift:.2f})"
            insights.append(insight)
        
        return insights
    
    def create_visualization(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of association rules
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.association_rules is None or len(self.association_rules) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No association rules to visualize', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Association Rules Visualization')
            return fig
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Support vs Confidence
        ax1.scatter(self.association_rules['support'], self.association_rules['confidence'], 
                   c=self.association_rules['lift'], cmap='viridis', alpha=0.6)
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Support vs Confidence (colored by Lift)')
        
        # Plot 2: Top rules by confidence
        top_rules = self.association_rules.head(10)
        rule_labels = [f"{list(rule['antecedents'])[0]} → {list(rule['consequents'])[0]}" 
                      for _, rule in top_rules.iterrows()]
        ax2.barh(range(len(rule_labels)), top_rules['confidence'])
        ax2.set_yticks(range(len(rule_labels)))
        ax2.set_yticklabels(rule_labels, fontsize=8)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Top Rules by Confidence')
        
        # Plot 3: Lift distribution
        ax3.hist(self.association_rules['lift'], bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Lift')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Lift Values')
        
        # Plot 4: Support distribution
        ax4.hist(self.association_rules['support'], bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Support')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Support Values')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """
        Generate comprehensive report of association rules analysis
        
        Returns:
            Formatted report string
        """
        if self.association_rules is None:
            return "No association rules analysis performed."
        
        report = []
        report.append("=" * 60)
        report.append("ASSOCIATION RULES MINING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 20)
        report.append(f"Total Transactions: {len(self.transaction_data) if self.transaction_data is not None else 0}")
        report.append(f"Total Items: {len(self.transaction_data.columns) if self.transaction_data is not None else 0}")
        report.append(f"Frequent Itemsets: {len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0}")
        report.append(f"Association Rules: {len(self.association_rules)}")
        report.append("")
        
        # Top frequent itemsets
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            report.append("TOP FREQUENT ITEMSETS")
            report.append("-" * 25)
            top_itemsets = self.frequent_itemsets.head(10)
            for _, itemset in top_itemsets.iterrows():
                items = ', '.join(list(itemset['itemsets']))
                support = itemset['support']
                report.append(f"  {items}: {support:.3f} support")
            report.append("")
        
        # Top association rules
        if len(self.association_rules) > 0:
            report.append("TOP ASSOCIATION RULES")
            report.append("-" * 25)
            top_rules = self.association_rules.head(10)
            for _, rule in top_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                confidence = rule['confidence']
                support = rule['support']
                lift = rule['lift']
                report.append(f"  {antecedents} → {consequents}")
                report.append(f"    Confidence: {confidence:.3f}, Support: {support:.3f}, Lift: {lift:.3f}")
                report.append("")
        
        # Insights
        insights = self.get_insights()
        report.append("KEY INSIGHTS")
        report.append("-" * 15)
        for i, insight in enumerate(insights, 1):
            report.append(f"{i}. {insight}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
