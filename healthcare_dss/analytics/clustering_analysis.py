"""
Clustering Analysis Module for Healthcare DSS

This module implements clustering algorithms to discover natural groupings
of patients based on their characteristics, enabling patient segmentation,
disease subtype identification, and treatment group analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

logger = logging.getLogger(__name__)

class ClusteringAnalyzer:
    """
    Clustering Analysis for Healthcare Data
    
    Implements various clustering algorithms to discover patient groups,
    disease subtypes, and treatment patterns in healthcare datasets.
    """
    
    def __init__(self, data_manager):
        """
        Initialize Clustering Analyzer
        
        Args:
            data_manager: DataManager instance with loaded datasets
        """
        self.data_manager = data_manager
        self.cluster_models = {}
        self.cluster_results = {}
        self.scaler = StandardScaler()
        
    def prepare_data_for_clustering(self, dataset_name: str, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for clustering analysis
        
        Args:
            dataset_name: Name of the dataset to analyze
            features: List of features to use for clustering (if None, use all numeric features)
            
        Returns:
            Tuple of (scaled_features, original_features)
        """
        if dataset_name not in self.data_manager.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.data_manager.datasets[dataset_name].copy()
        
        # Select features for clustering
        if features is None:
            # Use all numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # If no numeric features, try to convert object columns to numeric
            if not numeric_features:
                logger.info(f"No numeric features found in {dataset_name}, attempting to convert object columns to numeric")
                # Try to convert object columns that might contain numeric data
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Try to convert to numeric, replacing '..' and other non-numeric values with NaN
                        try:
                            converted = pd.to_numeric(df[col], errors='coerce')
                            if not converted.isna().all():  # If at least some values converted successfully
                                df[col] = converted
                                numeric_features.append(col)
                        except:
                            continue
            
            # Remove target columns if they exist
            target_columns = ['target', 'target_name']
            features = [col for col in numeric_features if col not in target_columns]
        
        # Check if we have any features to cluster on
        if not features:
            raise ValueError(f"No suitable numeric features found for clustering in dataset {dataset_name}. "
                           f"Dataset has {len(df.columns)} columns: {list(df.columns)}")
        
        # Extract feature data
        feature_data = df[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        # Check if we have any valid data after handling missing values
        if feature_data.empty or feature_data.isna().all().all():
            raise ValueError(f"No valid data remaining for clustering in dataset {dataset_name}")
        
        # Scale features
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(feature_data),
            columns=feature_data.columns,
            index=feature_data.index
        )
        
        logger.info(f"Prepared {len(scaled_data)} samples with {len(features)} features for clustering")
        
        return scaled_data, feature_data
    
    def find_optimal_clusters(self, data: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            data: Scaled feature data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with optimal cluster analysis results
        """
        logger.info(f"Finding optimal number of clusters (max: {max_clusters})")
        
        # Test different numbers of clusters
        cluster_range = range(2, min(max_clusters + 1, len(data) // 2))
        
        # Store metrics
        metrics = {
            'k_values': [],
            'inertia': [],
            'silhouette_scores': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': []
        }
        
        for k in cluster_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
            davies_bouldin = davies_bouldin_score(data, cluster_labels)
            
            metrics['k_values'].append(k)
            metrics['inertia'].append(inertia)
            metrics['silhouette_scores'].append(silhouette)
            metrics['calinski_harabasz_scores'].append(calinski_harabasz)
            metrics['davies_bouldin_scores'].append(davies_bouldin)
        
        # Find optimal k based on silhouette score (higher is better)
        optimal_k = cluster_range[np.argmax(metrics['silhouette_scores'])]
        
        results = {
            'optimal_k': optimal_k,
            'metrics': metrics,
            'best_silhouette_score': max(metrics['silhouette_scores']),
            'recommendation': f"Optimal number of clusters: {optimal_k} (silhouette score: {max(metrics['silhouette_scores']):.3f})"
        }
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return results
    
    def perform_kmeans_clustering(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """
        Perform K-means clustering
        
        Args:
            data: Scaled feature data
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        silhouette = silhouette_score(data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
        davies_bouldin = davies_bouldin_score(data, cluster_labels)
        
        # Store results
        results = {
            'algorithm': 'K-means',
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'model': kmeans
        }
        
        self.cluster_models['kmeans'] = kmeans
        self.cluster_results['kmeans'] = results
        
        logger.info(f"K-means clustering completed. Silhouette score: {silhouette:.3f}")
        return results
    
    def perform_dbscan_clustering(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering
        
        Args:
            data: Scaled feature data
            eps: Maximum distance between samples
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)
        
        # Calculate number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        results = {
            'algorithm': 'DBSCAN',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_labels': cluster_labels,
            'eps': eps,
            'min_samples': min_samples,
            'model': dbscan
        }
        
        # Calculate metrics if we have more than 1 cluster
        if n_clusters > 1:
            # Remove noise points for metric calculation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                results['silhouette_score'] = silhouette_score(data[non_noise_mask], cluster_labels[non_noise_mask])
                results['calinski_harabasz_score'] = calinski_harabasz_score(data[non_noise_mask], cluster_labels[non_noise_mask])
                results['davies_bouldin_score'] = davies_bouldin_score(data[non_noise_mask], cluster_labels[non_noise_mask])
        
        self.cluster_models['dbscan'] = dbscan
        self.cluster_results['dbscan'] = results
        
        logger.info(f"DBSCAN clustering completed. Found {n_clusters} clusters and {n_noise} noise points")
        return results
    
    def perform_hierarchical_clustering(self, data: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """
        Perform Hierarchical clustering
        
        Args:
            data: Scaled feature data
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing Hierarchical clustering with {n_clusters} clusters")
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(data)
        
        # Calculate metrics
        silhouette = silhouette_score(data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
        davies_bouldin = davies_bouldin_score(data, cluster_labels)
        
        # Store results
        results = {
            'algorithm': 'Hierarchical',
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'model': hierarchical
        }
        
        self.cluster_models['hierarchical'] = hierarchical
        self.cluster_results['hierarchical'] = results
        
        logger.info(f"Hierarchical clustering completed. Silhouette score: {silhouette:.3f}")
        return results
    
    def analyze_patient_segments(self, dataset_name: str, cluster_labels: np.ndarray, original_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patient segments based on clustering results
        
        Args:
            dataset_name: Name of the dataset
            cluster_labels: Cluster labels for each patient
            original_data: Original feature data
            
        Returns:
            Dictionary with segment analysis results
        """
        logger.info(f"Analyzing patient segments for {dataset_name}")
        
        # Add cluster labels to data
        analysis_data = original_data.copy()
        analysis_data['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = analysis_data.groupby('cluster').agg({
            col: ['mean', 'std', 'count'] for col in original_data.columns
        }).round(3)
        
        # Calculate cluster sizes
        cluster_sizes = analysis_data['cluster'].value_counts().sort_index()
        
        # Calculate percentage of each cluster
        cluster_percentages = (cluster_sizes / len(analysis_data) * 100).round(2)
        
        # Create cluster profiles
        cluster_profiles = {}
        for cluster_id in sorted(analysis_data['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = analysis_data[analysis_data['cluster'] == cluster_id]
            profile = {
                'size': len(cluster_data),
                'percentage': cluster_percentages[cluster_id],
                'characteristics': {}
            }
            
            # Calculate mean values for each feature
            for col in original_data.columns:
                profile['characteristics'][col] = {
                    'mean': cluster_data[col].mean(),
                    'std': cluster_data[col].std(),
                    'min': cluster_data[col].min(),
                    'max': cluster_data[col].max()
                }
            
            cluster_profiles[f'Cluster_{cluster_id}'] = profile
        
        results = {
            'dataset': dataset_name,
            'total_patients': len(analysis_data),
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_percentages': cluster_percentages.to_dict(),
            'cluster_profiles': cluster_profiles,
            'cluster_statistics': cluster_stats
        }
        
        return results
    
    def create_visualization(self, data: pd.DataFrame, cluster_labels: np.ndarray, 
                           algorithm: str = "Clustering", figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create visualization of clustering results
        
        Args:
            data: Feature data
            cluster_labels: Cluster labels
            algorithm: Name of the clustering algorithm
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: 2D scatter plot of clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points
                ax1.scatter(data_2d[cluster_labels == label, 0], 
                           data_2d[cluster_labels == label, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax1.scatter(data_2d[cluster_labels == label, 0], 
                           data_2d[cluster_labels == label, 1], 
                           c=[color], label=f'Cluster {label}', alpha=0.7)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title(f'{algorithm} Results (2D Projection)')
        ax1.legend()
        
        # Plot 2: Cluster size distribution
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_sizes.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Cluster Size Distribution')
        ax2.tick_params(axis='x', rotation=0)
        
        # Plot 3: Feature importance (PCA components)
        feature_importance = np.abs(pca.components_).mean(axis=0)
        feature_names = data.columns
        sorted_idx = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
        
        ax3.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        ax3.set_yticks(range(len(sorted_idx)))
        ax3.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax3.set_xlabel('Feature Importance')
        ax3.set_title('Top 10 Most Important Features')
        
        # Plot 4: Silhouette analysis (if applicable)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            from sklearn.metrics import silhouette_samples
            silhouette_vals = silhouette_samples(data, cluster_labels)
            
            y_lower = 10
            for i, label in enumerate(unique_labels):
                cluster_silhouette_vals = silhouette_vals[cluster_labels == label]
                cluster_silhouette_vals.sort()
                
                size_cluster_i = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = colors[i] if i < len(colors) else 'gray'
                ax4.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                                 facecolor=color, edgecolor=color, alpha=0.7)
                
                ax4.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                y_lower = y_upper + 10
            
            ax4.axvline(x=0, color="black", linestyle="--")
            ax4.set_xlabel("Silhouette Coefficient")
            ax4.set_ylabel("Cluster")
            ax4.set_title("Silhouette Analysis")
        else:
            ax4.text(0.5, 0.5, 'Silhouette analysis\nnot applicable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Silhouette Analysis")
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, dataset_name: str) -> str:
        """
        Generate comprehensive clustering analysis report
        
        Args:
            dataset_name: Name of the analyzed dataset
            
        Returns:
            Formatted report string
        """
        if not self.cluster_results:
            return "No clustering analysis performed."
        
        report = []
        report.append("=" * 60)
        report.append("CLUSTERING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary for each algorithm
        for algo_name, results in self.cluster_results.items():
            report.append(f"{results['algorithm'].upper()} CLUSTERING RESULTS")
            report.append("-" * 30)
            report.append(f"Number of clusters: {results['n_clusters']}")
            
            if 'silhouette_score' in results:
                report.append(f"Silhouette score: {results['silhouette_score']:.3f}")
            if 'calinski_harabasz_score' in results:
                report.append(f"Calinski-Harabasz score: {results['calinski_harabasz_score']:.3f}")
            if 'davies_bouldin_score' in results:
                report.append(f"Davies-Bouldin score: {results['davies_bouldin_score']:.3f}")
            if 'n_noise' in results:
                report.append(f"Noise points: {results['n_noise']}")
            
            report.append("")
        
        # Cluster analysis
        if 'kmeans' in self.cluster_results:
            kmeans_results = self.cluster_results['kmeans']
            cluster_labels = kmeans_results['cluster_labels']
            
            # Analyze segments
            _, original_data = self.prepare_data_for_clustering(dataset_name)
            segment_analysis = self.analyze_patient_segments(dataset_name, cluster_labels, original_data)
            
            report.append("PATIENT SEGMENT ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total patients: {segment_analysis['total_patients']}")
            report.append(f"Number of segments: {segment_analysis['n_clusters']}")
            report.append("")
            
            report.append("Segment Sizes:")
            for cluster_id, size in segment_analysis['cluster_sizes'].items():
                percentage = segment_analysis['cluster_percentages'][cluster_id]
                report.append(f"  Cluster {cluster_id}: {size} patients ({percentage}%)")
            report.append("")
            
            # Segment characteristics
            report.append("Segment Characteristics:")
            for cluster_name, profile in segment_analysis['cluster_profiles'].items():
                report.append(f"  {cluster_name}:")
                report.append(f"    Size: {profile['size']} patients ({profile['percentage']}%)")
                report.append("    Key characteristics:")
                for feature, stats in profile['characteristics'].items():
                    report.append(f"      {feature}: {stats['mean']:.3f} ± {stats['std']:.3f}")
                report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
