"""
Intelligent Binning System for Healthcare DSS
============================================

This module provides intelligent binning strategies for converting continuous
target variables into discrete classes for classification tasks. It automatically
detects when binning is needed and provides optimal binning strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class IntelligentBinningSystem:
    """Intelligent binning system for continuous to discrete conversion"""
    
    def __init__(self):
        self.binning_strategies = {
            'quantile': 'Quantile-based binning (equal frequency)',
            'uniform': 'Uniform binning (equal width)',
            'kmeans': 'K-means clustering binning',
            'jenks': 'Jenks natural breaks binning',
            'custom': 'Custom threshold binning'
        }
    
    def detect_binning_need(self, y: np.ndarray, task_type: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if binning is needed for the target variable
        
        Args:
            y: Target variable array
            task_type: Intended task type ('classification' or 'regression')
            
        Returns:
            Tuple of (needs_binning, analysis_details)
        """
        # Handle empty data
        if len(y) == 0:
            raise ValueError("Cannot analyze empty data")
        
        unique_values = len(np.unique(y))
        total_samples = len(y)
        
        analysis = {
            'unique_values': unique_values,
            'total_samples': total_samples,
            'unique_ratio': unique_values / total_samples if total_samples > 0 else 0.0,
            'is_numeric': pd.api.types.is_numeric_dtype(y),
            'data_type': str(y.dtype),
            'min_value': float(np.min(y)),
            'max_value': float(np.max(y)),
            'range': float(np.max(y) - np.min(y)),
            'std': float(np.std(y)),
            'mean': float(np.mean(y))
        }
        
        # Determine if binning is needed
        needs_binning = False
        reasons = []
        
        if task_type == 'classification':
            unique_values = analysis['unique_values']
            unique_ratio = analysis['unique_ratio']
            
            # For discrete data with very low cardinality, no binning needed
            if unique_values <= 5 and unique_ratio <= 0.2:
                needs_binning = False
                reasons.append(f"Discrete data with low cardinality ({unique_values} unique values)")
            
            # Check if we have too many unique values for classification
            elif unique_values > 20:
                needs_binning = True
                reasons.append(f"Too many unique values ({unique_values}) for classification")
            
            # Check if unique ratio is very high (continuous data)
            elif unique_ratio > 0.5:
                needs_binning = True
                reasons.append(f"Very high unique ratio ({unique_ratio:.1%}) suggests continuous data")
            
            # Check if values are numeric and have high cardinality
            elif analysis['is_numeric'] and unique_ratio > 0.3:
                needs_binning = True
                reasons.append("Numeric data with high cardinality")
            
            else:
                needs_binning = False
                reasons.append(f"Appropriate discrete data ({unique_values} unique values)")
        
        analysis['needs_binning'] = needs_binning
        analysis['reasons'] = reasons
        
        return needs_binning, analysis
    
    def suggest_optimal_bins(self, y: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest optimal number of bins based on data characteristics
        
        Args:
            y: Target variable array
            analysis: Analysis details from detect_binning_need
            
        Returns:
            Dictionary with binning suggestions
        """
        suggestions = {
            'optimal_bins': 3,
            'min_bins': 2,
            'max_bins': 10,
            'recommended_strategy': 'quantile',
            'reasoning': []
        }
        
        unique_values = analysis['unique_values']
        total_samples = analysis['total_samples']
        
        # Calculate optimal bins using various methods
        # Method 1: Square root rule
        sqrt_bins = max(2, int(np.sqrt(total_samples)))
        
        # Method 2: Sturges' rule
        sturges_bins = max(2, int(1 + np.log2(total_samples)))
        
        # Method 3: Rice rule
        rice_bins = max(2, int(2 * (total_samples ** (1/3))))
        
        # Method 4: Based on unique values
        unique_based_bins = min(10, max(2, unique_values // 10))
        
        # Choose optimal bins (use median of methods)
        methods = [sqrt_bins, sturges_bins, rice_bins, unique_based_bins]
        optimal_bins = int(np.median(methods))
        
        # Ensure reasonable bounds
        optimal_bins = max(2, min(10, optimal_bins))
        
        suggestions['optimal_bins'] = optimal_bins
        suggestions['min_bins'] = 2
        suggestions['max_bins'] = min(10, unique_values // 2)
        suggestions['reasoning'] = [
            f"Square root rule suggests {sqrt_bins} bins",
            f"Sturges' rule suggests {sturges_bins} bins", 
            f"Rice rule suggests {rice_bins} bins",
            f"Unique values suggest {unique_based_bins} bins",
            f"Selected optimal: {optimal_bins} bins"
        ]
        
        # Recommend strategy based on data distribution
        if analysis['std'] / analysis['mean'] < 0.5:  # Low coefficient of variation
            suggestions['recommended_strategy'] = 'uniform'
            suggestions['reasoning'].append("Low variation suggests uniform binning")
        else:
            suggestions['recommended_strategy'] = 'quantile'
            suggestions['reasoning'].append("High variation suggests quantile binning")
        
        return suggestions
    
    def apply_binning(self, y: np.ndarray, strategy: str, n_bins: int, 
                     custom_thresholds: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply binning strategy to convert continuous values to discrete classes
        
        Args:
            y: Target variable array
            strategy: Binning strategy ('quantile', 'uniform', 'kmeans', 'jenks', 'custom')
            n_bins: Number of bins
            custom_thresholds: Custom thresholds for 'custom' strategy
            
        Returns:
            Tuple of (binned_values, binning_info)
        """
        # Validate inputs
        if strategy not in ['quantile', 'uniform', 'kmeans', 'jenks', 'custom']:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        if n_bins < 2:
            raise ValueError(f"Number of bins must be at least 2, got {n_bins}")
        
        if len(y) == 0:
            raise ValueError("Cannot bin empty data")
        
        # Handle infinite and NaN values
        if np.any(np.isinf(y)) or np.any(np.isnan(y)):
            logger.warning("Data contains infinite or NaN values, filtering them out")
            y = y[np.isfinite(y)]
            if len(y) == 0:
                raise ValueError("No finite values remaining after filtering")
        
        binning_info = {
            'strategy': strategy,
            'n_bins': n_bins,
            'original_unique': len(np.unique(y)),
            'binned_unique': 0,
            'bin_edges': [],
            'bin_counts': [],
            'bin_labels': []
        }
        
        try:
            if strategy == 'quantile':
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
                binning_info['bin_edges'] = discretizer.bin_edges_[0].tolist()
                
            elif strategy == 'uniform':
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
                y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
                binning_info['bin_edges'] = discretizer.bin_edges_[0].tolist()
                
            elif strategy == 'kmeans':
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
                binning_info['bin_edges'] = discretizer.bin_edges_[0].tolist()
                
            elif strategy == 'jenks':
                # Implement Jenks natural breaks
                y_binned = self._jenks_binning(y, n_bins)
                binning_info['bin_edges'] = self._get_jenks_breaks(y, n_bins)
                
            elif strategy == 'custom' and custom_thresholds:
                y_binned = self._custom_binning(y, custom_thresholds)
                binning_info['bin_edges'] = custom_thresholds
                
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
            
            # Ensure integer labels
            y_binned = y_binned.astype(int)
            
            # Calculate bin statistics
            binning_info['binned_unique'] = len(np.unique(y_binned))
            binning_info['bin_counts'] = np.bincount(y_binned).tolist()
            binning_info['bin_labels'] = [f"Class {i}" for i in range(len(np.unique(y_binned)))]
            
            # Validate binning
            min_class_count = min(binning_info['bin_counts'])
            if min_class_count < 2:
                logger.warning(f"Binning resulted in class with only {min_class_count} samples")
                # Try with fewer bins, but avoid infinite recursion
                if n_bins > 2 and len(binning_info['bin_counts']) > 2:
                    return self.apply_binning(y, strategy, n_bins - 1, custom_thresholds)
                else:
                    # If we can't reduce further, use the current result
                    logger.warning(f"Using binning with {min_class_count} samples in smallest class")
            
            return y_binned, binning_info
            
        except Exception as e:
            logger.error(f"Binning failed: {e}")
            # Fallback to simple quantile binning
            return self.apply_binning(y, 'quantile', max(2, n_bins - 1), custom_thresholds)
    
    def _jenks_binning(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Apply Jenks natural breaks binning"""
        sorted_y = np.sort(y)
        breaks = self._get_jenks_breaks(sorted_y, n_bins)
        
        y_binned = np.zeros_like(y, dtype=int)
        for i, value in enumerate(y):
            for j, break_point in enumerate(breaks[1:]):
                if value <= break_point:
                    y_binned[i] = j
                    break
            else:
                y_binned[i] = len(breaks) - 2
        
        return y_binned
    
    def _get_jenks_breaks(self, y: np.ndarray, n_bins: int) -> List[float]:
        """Calculate Jenks natural breaks"""
        if n_bins >= len(y):
            return sorted(y)
        
        # Simplified Jenks implementation
        # For production, consider using a more sophisticated implementation
        breaks = [y[0]]
        for i in range(1, n_bins):
            idx = int(i * len(y) / n_bins)
            breaks.append(y[idx])
        breaks.append(y[-1])
        
        return sorted(list(set(breaks)))
    
    def _custom_binning(self, y: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """Apply custom threshold binning"""
        y_binned = np.zeros_like(y, dtype=int)
        
        for i, value in enumerate(y):
            for j, threshold in enumerate(thresholds):
                if value <= threshold:
                    y_binned[i] = j
                    break
            else:
                y_binned[i] = len(thresholds)
        
        return y_binned
    
    def get_user_override_options(self, y: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user override options for advanced binning control
        
        Args:
            y: Target variable array
            analysis: Analysis details from detect_binning_need
            
        Returns:
            Dictionary with user override options
        """
        override_options = {
            'custom_thresholds': [],
            'custom_bin_labels': [],
            'force_binning': False,
            'disable_binning': False,
            'custom_strategy': None,
            'advanced_settings': {}
        }
        
        # Generate custom threshold suggestions based on data distribution
        if analysis['is_numeric']:
            min_val = analysis['min_value']
            max_val = analysis['max_value']
            std_val = analysis['std']
            mean_val = analysis['mean']
            
            # Suggest thresholds based on statistical properties
            percentile_thresholds = [25, 50, 75]  # Quartiles
            sigma_thresholds = [mean_val - std_val, mean_val, mean_val + std_val]  # Mean ± 1σ
            
            override_options['custom_thresholds'] = {
                'percentile_based': [np.percentile(y, p) for p in percentile_thresholds],
                'sigma_based': sigma_thresholds,
                'equal_width': np.linspace(min_val, max_val, 4)[1:-1].tolist(),  # 3 equal-width bins
                'manual': []
            }
            
            # Generate custom bin labels
            override_options['custom_bin_labels'] = {
                'numeric_ranges': [f"{min_val:.1f}-{max_val:.1f}"],
                'severity_levels': ['Low', 'Medium', 'High'],
                'risk_levels': ['Low Risk', 'Moderate Risk', 'High Risk'],
                'performance_levels': ['Poor', 'Average', 'Good', 'Excellent'],
                'manual': []
            }
        
        return override_options
    
    def apply_user_override_binning(self, y: np.ndarray, override_config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply binning with user override configuration
        
        Args:
            y: Target variable array
            override_config: User override configuration
            
        Returns:
            Tuple of (binned_values, binning_info)
        """
        binning_info = {
            'strategy': 'user_override',
            'user_configured': True,
            'original_unique': len(np.unique(y)),
            'binned_unique': 0,
            'bin_edges': [],
            'bin_counts': [],
            'bin_labels': [],
            'override_config': override_config
        }
        
        try:
            # Handle custom thresholds
            if override_config.get('custom_thresholds'):
                thresholds = override_config['custom_thresholds']
                y_binned = self._custom_binning(y, thresholds)
                binning_info['bin_edges'] = thresholds
                binning_info['bin_labels'] = override_config.get('custom_bin_labels', [f"Class {i}" for i in range(len(thresholds) + 1)])
            
            # Handle custom strategy override
            elif override_config.get('custom_strategy') and override_config.get('n_bins'):
                strategy = override_config['custom_strategy']
                n_bins = override_config['n_bins']
                y_binned, standard_info = self.apply_binning(y, strategy, n_bins)
                binning_info.update(standard_info)
                binning_info['strategy'] = f"user_override_{strategy}"
            
            # Handle force binning with optimal settings
            elif override_config.get('force_binning'):
                needs_binning, analysis = self.detect_binning_need(y, 'classification')
                suggestions = self.suggest_optimal_bins(y, analysis)
                strategy = suggestions['recommended_strategy']
                n_bins = suggestions['optimal_bins']
                y_binned, standard_info = self.apply_binning(y, strategy, n_bins)
                binning_info.update(standard_info)
                binning_info['strategy'] = f"forced_{strategy}"
            
            # Handle disable binning
            elif override_config.get('disable_binning'):
                # Return original data with warning
                binning_info['strategy'] = 'disabled'
                binning_info['warning'] = 'Binning disabled by user - may cause classification issues'
                return y, binning_info
            
            else:
                raise ValueError("Invalid override configuration")
            
            # Ensure integer labels
            y_binned = y_binned.astype(int)
            
            # Calculate bin statistics
            binning_info['binned_unique'] = len(np.unique(y_binned))
            binning_info['bin_counts'] = np.bincount(y_binned).tolist()
            
            # Validate binning
            min_class_count = min(binning_info['bin_counts'])
            if min_class_count < 2:
                binning_info['warning'] = f"Class with only {min_class_count} samples - may cause CV issues"
            
            return y_binned, binning_info
            
        except Exception as e:
            logger.error(f"User override binning failed: {e}")
            # Fallback to standard binning
            return self.apply_binning(y, 'quantile', 3)
    
    def get_binning_preview(self, y: np.ndarray, strategy: str, n_bins: int) -> Dict[str, Any]:
        """
        Get a preview of binning results without applying it
        
        Args:
            y: Target variable array
            strategy: Binning strategy
            n_bins: Number of bins
            
        Returns:
            Dictionary with binning preview information
        """
        try:
            y_binned, binning_info = self.apply_binning(y, strategy, n_bins)
            
            preview = {
                'strategy': strategy,
                'n_bins': n_bins,
                'bin_counts': binning_info['bin_counts'],
                'bin_labels': binning_info['bin_labels'],
                'bin_edges': binning_info['bin_edges'],
                'min_class_count': min(binning_info['bin_counts']),
                'max_class_count': max(binning_info['bin_counts']),
                'class_balance': min(binning_info['bin_counts']) / max(binning_info['bin_counts']),
                'success': True
            }
            
            return preview
            
        except Exception as e:
            return {
                'strategy': strategy,
                'n_bins': n_bins,
                'error': str(e),
                'success': False
            }


# Global instance
intelligent_binning = IntelligentBinningSystem()
