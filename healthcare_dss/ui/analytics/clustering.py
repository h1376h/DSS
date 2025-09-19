"""
Clustering Analysis Module
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_dataset_info, 
    get_available_datasets,
    display_error_message,
    display_success_message,
    display_warning_message,
    get_numeric_columns,
    create_analysis_summary,
    safe_dataframe_display
)
from healthcare_dss.ui.utils.visualization import create_cluster_visualization
from healthcare_dss.utils.debug_manager import debug_manager


def show_clustering_analysis():
    """Show clustering analysis interface"""
    st.header("Clustering Analysis")
    st.markdown("**Discover patient segments and patterns using unsupervised learning**")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Clustering Analysis Debug", expanded=False):
            debug_data = debug_manager.get_page_debug_data("Clustering Analysis", {
                "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                "Available Datasets": len(st.session_state.data_manager.datasets) if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager else 0,
                "System Initialized": check_system_initialization()
            })
            
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                st.markdown("---")
                st.subheader("Available Datasets for Clustering")
                datasets = get_dataset_names()
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        numeric_cols = get_numeric_columns(df)
                        st.write(f"**{dataset}**: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Numeric columns: {len(numeric_cols)} ({numeric_cols})")
    
    if not check_system_initialization():
        st.error("DSS system not initialized. Please refresh the page or restart the application.")
        return
    
    try:
        datasets = get_available_datasets()
        
        if not datasets:
            display_warning_message("No datasets available. Please load datasets first.")
            return
        
        # Dataset selection
        st.subheader("1. Select Dataset")
        selected_dataset = st.selectbox("Choose a dataset:", list(datasets.keys()), key="cluster_dataset")
        
        if selected_dataset:
            dataset = datasets[selected_dataset]
            display_dataset_info(dataset, selected_dataset)
            
            # Feature selection
            st.subheader("2. Select Features for Clustering")
            numeric_cols = get_numeric_columns(dataset)
            
            if not numeric_cols:
                display_warning_message("No numeric columns found. Clustering requires numeric features.")
                return
            
            # Feature selection
            selected_features = st.multiselect(
                "Select features for clustering:",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                key="cluster_features"
            )
            
            if not selected_features:
                st.error("Please select at least one feature for clustering.")
                return
            
            # Clustering parameters
            st.subheader("3. Clustering Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_clusters = st.number_input("Number of clusters", 2, 20, 3, key="cluster_n_clusters")
            with col2:
                algorithm = st.selectbox("Clustering Algorithm", ["kmeans", "hierarchical", "dbscan"], key="cluster_algorithm")
            with col3:
                random_state = st.number_input("Random State", 0, 1000, 42, key="cluster_random_state")
            
            # Additional parameters for specific algorithms
            if algorithm == "dbscan":
                eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, key="cluster_eps")
                min_samples = st.number_input("Minimum samples", 2, 20, 5, key="cluster_min_samples")
            
            # Run clustering
            if st.button("🔍 Perform Clustering", type="primary"):
                with st.spinner("Performing clustering analysis..."):
                    try:
                        # Prepare data
                        clustering_data = dataset[selected_features].copy()
                        clustering_data = clustering_data.dropna()
                        
                        if len(clustering_data) == 0:
                            st.error("No data available after removing missing values.")
                            return
                        
                        # Standardize features
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(clustering_data)
                        
                        # Perform clustering
                        if algorithm == "kmeans":
                            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
                            cluster_labels = clusterer.fit_predict(scaled_data)
                            
                        elif algorithm == "hierarchical":
                            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                            cluster_labels = clusterer.fit_predict(scaled_data)
                            
                        elif algorithm == "dbscan":
                            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                            cluster_labels = clusterer.fit_predict(scaled_data)
                        
                        # Add cluster labels to data
                        clustering_data['Cluster'] = cluster_labels
                        
                        # Display results
                        display_success_message("Clustering completed successfully!")
                        
                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        cluster_stats = clustering_data.groupby('Cluster').agg({
                            **{col: ['count', 'mean', 'std'] for col in selected_features}
                        }).round(3)
                        
                        safe_dataframe_display(cluster_stats)
                        
                        # Cluster distribution
                        st.subheader("Cluster Distribution")
                        cluster_counts = clustering_data['Cluster'].value_counts().sort_index()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(cluster_counts)
                        with col2:
                            st.write("**Cluster Sizes:**")
                            for cluster_id, count in cluster_counts.items():
                                percentage = (count / len(clustering_data)) * 100
                                st.write(f"Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
                        
                        # Feature importance per cluster
                        st.subheader("Cluster Characteristics")
                        for cluster_id in sorted(cluster_labels):
                            if cluster_id == -1:  # Noise points in DBSCAN
                                continue
                                
                            cluster_data = clustering_data[clustering_data['Cluster'] == cluster_id]
                            st.write(f"**Cluster {cluster_id}:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                for feature in selected_features[:3]:  # Show top 3 features
                                    mean_val = cluster_data[feature].mean()
                                    st.write(f"- {feature}: {mean_val:.2f}")
                            with col2:
                                for feature in selected_features[3:6]:  # Show next 3 features
                                    if feature in cluster_data.columns:
                                        mean_val = cluster_data[feature].mean()
                                        st.write(f"- {feature}: {mean_val:.2f}")
                        
                        # Visualization (if we have 2D data)
                        if len(selected_features) >= 2:
                            create_cluster_visualization(
                                clustering_data, cluster_labels, 
                                selected_features[0], selected_features[1], algorithm
                            )
                        
                        # Summary
                        create_analysis_summary(len(clustering_data), len(selected_features), f"{algorithm.title()} Clustering")
                        
                    except Exception as e:
                        display_error_message(e, "in clustering analysis")
    
    except Exception as e:
        display_error_message(e, "accessing datasets")
