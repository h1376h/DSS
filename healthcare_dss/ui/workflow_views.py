"""
Workflow Views Module
=====================

Contains workflow and analysis views for the Healthcare DSS:
- CRISP-DM Workflow
- Classification Evaluation
- Association Rules
- Clustering Analysis
- Time Series Analysis
- Prescriptive Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import logging
from healthcare_dss.ui.utils.common import (
    check_system_initialization,
    display_error_message,
    display_success_message,
    display_warning_message,
    safe_dataframe_display,
    get_dataset_names,
    get_dataset_from_managers
)

logger = logging.getLogger(__name__)


def show_crisp_dm_workflow():
    """Show CRISP-DM workflow interface"""
    st.header("CRISP-DM Workflow")
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 CRISP-DM Workflow Debug", expanded=True):
            st.write("**CRISP-DM Workflow Debug:**")
            st.write(f"- Function called: show_crisp_dm_workflow()")
            st.write(f"- Session state initialized: {st.session_state.get('initialized', False)}")
            st.write(f"- Has crisp_dm_workflow: {hasattr(st.session_state, 'crisp_dm_workflow')}")
            st.write(f"- Has data_manager: {hasattr(st.session_state, 'data_manager')}")
            st.write(f"- Debug mode: {st.session_state.get('debug_mode', False)}")
            
            if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager is not None:
                st.write(f"- Available datasets: {list(st.session_state.data_manager.datasets.keys())}")
            else:
                st.write("- Data Manager not available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not check_system_initialization():
            st.error("System not initialized. Please refresh the page.")
            return
        
        datasets = get_dataset_names()
        if not datasets:
            st.warning("No datasets available. Please load datasets first.")
            return
        
        dataset_name = st.selectbox(
            "Select Dataset",
            datasets,
            key="crisp_dataset"
        )
        
        if dataset_name:
            df = get_dataset_from_managers(dataset_name)
            if df is not None:
                # Import smart target manager
                try:
                    from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
                    smart_manager = SmartDatasetTargetManager()
                    
                    # Get smart target recommendations
                    smart_targets = smart_manager.get_dataset_targets(dataset_name)
                    
                    if smart_targets:
                        st.info("Smart Target Recommendations Available!")
                        
                        # Show smart targets
                        st.write("**Recommended Targets:**")
                        for i, target in enumerate(smart_targets):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{target['column']}** ({target['target_type']})")
                                st.caption(target.get('business_meaning', 'Target variable'))
                            with col2:
                                if st.button(f"Select", key=f"crisp_select_{i}"):
                                    st.session_state.crisp_target = target['column']
                                    st.rerun()
                        
                        st.markdown("---")
                        
                except Exception as e:
                    st.warning(f"Could not load smart recommendations: {e}")
                
                # Standard target selection
                target_column = st.selectbox(
                    "Target Column",
                    df.columns.tolist(),
                    key="crisp_target"
                )
            else:
                st.error(f"Dataset '{dataset_name}' not found in any data manager.")
        else:
            target_column = st.selectbox(
                "Target Column",
                ["target", "outcome", "diagnosis"],
                key="crisp_target"
            )
    
    with col2:
        business_objective = st.text_area(
            "Business Objective",
            "Predict patient outcomes for clinical decision support",
            height=100
        )
    
    if st.button("Run CRISP-DM Workflow"):
        with st.spinner("Running CRISP-DM workflow..."):
            try:
                if hasattr(st.session_state, 'crisp_dm_workflow') and st.session_state.crisp_dm_workflow:
                    # Validate dataset and target column before running workflow
                    df = get_dataset_from_managers(dataset_name)
                    if df is None:
                        st.error(f"Dataset '{dataset_name}' not found in any data manager.")
                        return
                    
                    if target_column not in df.columns:
                        st.error(f"Target column '{target_column}' not found in dataset '{dataset_name}'.")
                        return
                    
                    # Check if target column has sufficient data
                    target_values = df[target_column]
                    unique_values = target_values.nunique()
                    min_count = target_values.value_counts().min()
                    
                    if unique_values < 2:
                        st.error(f"Target column '{target_column}' has only {unique_values} unique value(s). Need at least 2 for classification.")
                        return
                    
                    if min_count < 2:
                        st.error(f"Target column '{target_column}' has classes with insufficient data (minimum count: {min_count}). Need at least 2 samples per class.")
                        return
                    
                    if len(df) < 10:
                        st.error(f"Dataset '{dataset_name}' has only {len(df)} samples. Need at least 10 samples for meaningful analysis.")
                        return
                    
                    workflow_results = st.session_state.crisp_dm_workflow.execute_full_workflow(
                        dataset_name=dataset_name,
                        target_column=target_column,
                        business_objective=business_objective
                    )
                    
                    st.success("CRISP-DM workflow completed successfully!")
                    
                    # Display results
                    st.subheader("Workflow Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Model", workflow_results['evaluation']['best_model'])
                    with col2:
                        st.metric("Best Accuracy", f"{workflow_results['evaluation']['best_accuracy']:.3f}")
                    with col3:
                        st.metric("Models Evaluated", workflow_results['evaluation']['evaluation_summary']['models_evaluated'])
                    
                    # Generate report
                    if st.button("Generate Workflow Report"):
                        report = st.session_state.crisp_dm_workflow.generate_workflow_report()
                        st.text_area("CRISP-DM Report", report, height=400)
                else:
                    st.warning("CRISP-DM workflow component not available. Using mock results.")
                    
                    # Mock results for demonstration
                    st.success("CRISP-DM workflow completed successfully!")
                    
                    st.subheader("Workflow Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Model", "Random Forest")
                    with col2:
                        st.metric("Best Accuracy", "0.956")
                    with col3:
                        st.metric("Models Evaluated", "5")
                    
                    # Mock report
                    if st.button("Generate Workflow Report"):
                        mock_report = f"""
CRISP-DM Workflow Report
========================

Dataset: {dataset_name}
Target Column: {target_column}
Business Objective: {business_objective}

Phase 1: Business Understanding
- Objective: {business_objective}
- Success Criteria: Model accuracy > 90%

Phase 2: Data Understanding
- Dataset: {dataset_name}
- Records: 1,000+
- Features: 10+

Phase 3: Data Preparation
- Missing values handled
- Feature scaling applied
- Train/test split: 80/20

Phase 4: Modeling
- Models tested: Random Forest, SVM, Neural Network, KNN, Logistic Regression
- Best performing: Random Forest
- Cross-validation: 5-fold

Phase 5: Evaluation
- Accuracy: 95.6%
- Precision: 94.2%
- Recall: 96.8%
- F1-Score: 95.5%

Phase 6: Deployment
- Model ready for production
- Performance monitoring enabled
                        """
                        st.text_area("CRISP-DM Report", mock_report, height=400)
                
            except ValueError as e:
                st.error(f"Data validation error: {str(e)}")
            except Exception as e:
                st.error(f"Error running CRISP-DM workflow: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.exception(e)


def show_classification_evaluation():
    """Show classification evaluation interface with cross-validation"""
    st.header("Classification Evaluation & Cross-Validation")
    
    # Import required modules
    try:
        from healthcare_dss.core.data_management import DataManager
        from healthcare_dss.analytics.model_evaluation import ModelEvaluationEngine
        from healthcare_dss.analytics.model_training import ModelTrainingEngine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import pandas as pd
        import numpy as np
        
        # Initialize components
        data_manager = DataManager()
        evaluation_engine = ModelEvaluationEngine(data_manager)
        training_engine = ModelTrainingEngine()
        
        st.success("✅ All components loaded successfully!")
        
        # Dataset selection
        st.subheader("Dataset Selection")
        available_datasets = list(data_manager.datasets.keys())
        classification_datasets = [d for d in available_datasets if d in ['breast_cancer', 'wine', 'medication_effectiveness']]
        
        if not classification_datasets:
            st.warning("No classification datasets available. Please ensure datasets are loaded.")
            return
            
        selected_dataset = st.selectbox("Select Dataset", classification_datasets)
        
        if selected_dataset:
            # Get dataset info
            dataset_df = data_manager.datasets[selected_dataset]
            st.info(f"Dataset: {selected_dataset} | Shape: {dataset_df.shape}")
            
            # Target column selection
            numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select Target Column", numeric_cols)
                
                if target_col:
                    # Prepare data
                    features = dataset_df.drop(columns=[target_col])
                    numeric_features = features.select_dtypes(include=[np.number]).columns
                    features = features[numeric_features]
                    target = dataset_df[target_col]
                    
                    st.info(f"Features: {len(numeric_features)} | Samples: {len(features)}")
                    
                    # Model selection
                    st.subheader("Model Selection")
                    model_options = {
                        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
                        'SVM': SVC(random_state=42, probability=True)
                    }
                    
                    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
                    selected_model = model_options[selected_model_name]
                    
                    # Cross-validation options
                    st.subheader("Cross-Validation Options")
                    cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
                    
                    # Run evaluation buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Run Cross-Validation", type="primary"):
                            with st.spinner("Running cross-validation..."):
                                try:
                                    # Perform cross-validation
                                    cv_results = evaluation_engine.cross_validate_model(
                                        model=selected_model,
                                        X=features.head(100),  # Use subset for demo
                                        y=target.head(100),
                                        cv=cv_folds,
                                        task_type='classification'
                                    )
                                    
                                    if 'error' not in cv_results:
                                        st.success("✅ Cross-validation completed!")
                                        
                                        # Display results
                                        st.subheader("Cross-Validation Results")
                                        
                                        # Overall metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Overall Score", f"{cv_results['overall_score']:.3f}")
                                        with col2:
                                            st.metric("Stability Score", f"{cv_results['stability_score']:.3f}")
                                        with col3:
                                            st.metric("CV Folds", cv_results['cv_folds'])
                                        with col4:
                                            st.metric("Task Type", cv_results['task_type'])
                                        
                                        # Detailed CV results
                                        st.subheader("Detailed CV Metrics")
                                        cv_metrics = cv_results['cv_results']
                                        
                                        for metric, results in cv_metrics.items():
                                            if 'error' not in results:
                                                st.write(f"**{metric.replace('_', ' ').title()}:**")
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Mean", f"{results['mean']:.3f}")
                                                with col2:
                                                    st.metric("Std", f"{results['std']:.3f}")
                                                with col3:
                                                    st.metric("Min-Max", f"{results['min']:.3f} - {results['max']:.3f}")
                                        
                                        # Performance assessment
                                        st.subheader("Performance Assessment")
                                        perf = cv_results['model_performance']
                                        stability = cv_results['stability_assessment']
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Model Performance:**")
                                            for level, status in perf.items():
                                                emoji = "✅" if status else "❌"
                                                st.write(f"{emoji} {level.title()}")
                                        
                                        with col2:
                                            st.write("**Stability Assessment:**")
                                            for level, status in stability.items():
                                                emoji = "✅" if status else "❌"
                                                st.write(f"{emoji} {level.replace('_', ' ').title()}")
                                    
                                    else:
                                        st.error(f"Cross-validation failed: {cv_results['error']}")
                                        
                                except Exception as e:
                                    st.error(f"Error during cross-validation: {str(e)}")
                    
                    with col2:
                        if st.button("Run Model Training & Evaluation"):
                            with st.spinner("Training model and running evaluation..."):
                                try:
                                    # Train model
                                    training_result = training_engine.train_model(
                                        features=features.head(100),
                                        target=target.head(100),
                                        model_name='random_forest',
                                        task_type='classification'
                                    )
                                    
                                    if 'model_key' in training_result:
                                        st.success("✅ Model training completed!")
                                        
                                        # Run evaluation
                                        model = training_result['model']
                                        y_pred = model.predict(features.head(50))
                                        
                                        eval_results = evaluation_engine.evaluate_model_performance(
                                            model=model,
                                            X_test=features.head(50),
                                            y_test=target.head(50),
                                            y_pred=y_pred,
                                            task_type='classification'
                                        )
                                        
                                        st.subheader("Model Evaluation Results")
                                        
                                        # Display basic metrics
                                        if 'basic_metrics' in eval_results:
                                            basic_metrics = eval_results['basic_metrics']
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Accuracy", f"{basic_metrics.get('accuracy', 0):.3f}")
                                            with col2:
                                                st.metric("Precision", f"{basic_metrics.get('precision', 0):.3f}")
                                            with col3:
                                                st.metric("Recall", f"{basic_metrics.get('recall', 0):.3f}")
                                            with col4:
                                                st.metric("F1-Score", f"{basic_metrics.get('f1_score', 0):.3f}")
                                        
                                        # Display detailed metrics
                                        if 'detailed_metrics' in eval_results:
                                            detailed = eval_results['detailed_metrics']
                                            if detailed.get('confusion_matrix'):
                                                st.subheader("Confusion Matrix")
                                                cm = np.array(detailed['confusion_matrix']['matrix'])
                                                fig = px.imshow(cm, 
                                                              text_auto=True, 
                                                              aspect="auto",
                                                              title="Confusion Matrix")
                                                st.plotly_chart(fig, width="stretch")
                                    
                                    else:
                                        st.error("Model training failed")
                                        
                                except Exception as e:
                                    st.error(f"Error during training/evaluation: {str(e)}")
        
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.info("Please ensure all required modules are properly installed.")
    except Exception as e:
        st.error(f"Error initializing evaluation interface: {e}")
        st.info("This feature requires proper dataset and model setup.")
    
    # Fallback to mock evaluation if there are issues
    st.subheader("Mock Evaluation (Fallback)")
    if st.button("Run Mock Evaluation"):
        with st.spinner("Running evaluation..."):
            st.success("Evaluation completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", "0.956")
            with col2:
                st.metric("Precision", "0.942")
            with col3:
                st.metric("Recall", "0.968")
            with col4:
                st.metric("F1-Score", "0.955")
            
            # Confusion matrix visualization
            confusion_matrix = np.array([[85, 3], [2, 90]])
            fig = px.imshow(confusion_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, width="stretch")


def show_association_rules():
    """Show association rules mining interface"""
    st.header("Association Rules Mining")
    
    # Import the utility functions
    from healthcare_dss.ui.utils.common import get_dataset_names, get_dataset_from_managers
    
    # Get all available datasets from both managers
    datasets = get_dataset_names()
    if not datasets:
        st.warning("No datasets available. Please load datasets first.")
        return
    
    dataset_name = st.selectbox(
        "Select Dataset",
        datasets,
        key="association_dataset"
    )
    
    if st.button("Mine Association Rules"):
        with st.spinner("Mining association rules..."):
            try:
                if hasattr(st.session_state, 'association_rules_miner') and st.session_state.association_rules_miner:
                    results = st.session_state.association_rules_miner.analyze_healthcare_patterns(dataset_name)
                    
                    st.success("Association rules mining completed!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Transactions", results['total_transactions'])
                    with col2:
                        st.metric("Items", results['total_items'])
                    with col3:
                        st.metric("Frequent Itemsets", results['frequent_itemsets_count'])
                    with col4:
                        st.metric("Association Rules", results['association_rules_count'])
                    
                    # Generate insights
                    insights = st.session_state.association_rules_miner.get_insights()
                    st.subheader("Key Insights")
                    for i, insight in enumerate(insights[:5], 1):
                        st.write(f"{i}. {insight}")
                else:
                    st.warning("Association rules miner not available. Using mock results.")
                    
                    # Mock results
                    st.success("Association rules mining completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Transactions", "1,000")
                    with col2:
                        st.metric("Items", "25")
                    with col3:
                        st.metric("Frequent Itemsets", "150")
                    with col4:
                        st.metric("Association Rules", "45")
                    
                    # Mock insights
                    st.subheader("Key Insights")
                    mock_insights = [
                        "High BMI is associated with diabetes risk",
                        "Family history increases diabetes probability",
                        "Age > 45 correlates with hypertension",
                        "Smoking is linked to cardiovascular disease",
                        "Regular exercise reduces diabetes risk"
                    ]
                    for i, insight in enumerate(mock_insights, 1):
                        st.write(f"{i}. {insight}")
                
            except Exception as e:
                st.error(f"Error mining association rules: {e}")


def show_clustering_analysis():
    """Show clustering analysis interface"""
    st.header("Clustering Analysis")
    
    # Check system initialization
    if not check_system_initialization():
        st.error("System not initialized. Please refresh the page.")
        return
    
    # Get available datasets
    available_datasets = get_dataset_names()
    
    if not available_datasets:
        st.warning("No datasets available. Please load datasets first.")
        return
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Select Dataset",
        available_datasets,
        key="clustering_dataset",
        help="Choose a dataset for clustering analysis"
    )
    
    # Show dataset info
    if dataset_name:
        df = get_dataset_from_managers(dataset_name)
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=['number']).columns)
                st.metric("Numeric Features", numeric_cols)
        
            # Show dataset preview
            with st.expander("Dataset Preview"):
                safe_dataframe_display(df, max_rows=10)
        else:
            st.error(f"Dataset '{dataset_name}' not found in any data manager.")
    
    if st.button("Run Clustering Analysis"):
        with st.spinner("Running clustering analysis..."):
            try:
                if hasattr(st.session_state, 'clustering_analyzer') and st.session_state.clustering_analyzer:
                    # Prepare data
                    scaled_data, original_data = st.session_state.clustering_analyzer.prepare_data_for_clustering(dataset_name)
                    
                    # Find optimal clusters
                    optimal_results = st.session_state.clustering_analyzer.find_optimal_clusters(scaled_data)
                    optimal_k = optimal_results['optimal_k']
                    
                    # Perform clustering
                    kmeans_results = st.session_state.clustering_analyzer.perform_kmeans_clustering(scaled_data, optimal_k)
                    
                    st.success("Clustering analysis completed!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Optimal Clusters", optimal_k)
                    with col2:
                        st.metric("Silhouette Score", f"{kmeans_results['silhouette_score']:.3f}")
                    with col3:
                        st.metric("Calinski-Harabasz", f"{kmeans_results['calinski_harabasz_score']:.3f}")
                    
                    # Analyze segments
                    segment_analysis = st.session_state.clustering_analyzer.analyze_patient_segments(
                        dataset_name, kmeans_results['cluster_labels'], original_data
                    )
                    
                    st.subheader("Patient Segments")
                    for cluster_id, size in segment_analysis['cluster_sizes'].items():
                        percentage = segment_analysis['cluster_percentages'][cluster_id]
                        st.write(f"**Cluster {cluster_id}**: {size} patients ({percentage}%)")
                else:
                    st.warning("Clustering analyzer not available. Using mock results.")
                    
                    # Mock results
                    st.success("Clustering analysis completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Optimal Clusters", "3")
                    with col2:
                        st.metric("Silhouette Score", "0.756")
                    with col3:
                        st.metric("Calinski-Harabasz", "245.3")
                    
                    st.subheader("Patient Segments")
                    mock_segments = {
                        "Cluster 0": {"size": 350, "percentage": 35.0},
                        "Cluster 1": {"size": 400, "percentage": 40.0},
                        "Cluster 2": {"size": 250, "percentage": 25.0}
                    }
                    
                    for cluster_id, data in mock_segments.items():
                        st.write(f"**{cluster_id}**: {data['size']} patients ({data['percentage']}%)")
                
            except Exception as e:
                st.error(f"Error running clustering analysis: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.code(f"Exception: {type(e).__name__}")
                    st.code(f"Details: {str(e)}")


def show_time_series_analysis():
    """Show time series analysis interface"""
    st.header("Time Series Analysis")
    
    dataset_name = st.selectbox(
        "Select Dataset",
        ["healthcare_expenditure"] if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager and 'healthcare_expenditure' in st.session_state.data_manager.datasets else ["healthcare_expenditure"],
        key="timeseries_dataset"
    )
    
    if st.button("Run Time Series Analysis"):
        with st.spinner("Running time series analysis..."):
            try:
                if hasattr(st.session_state, 'time_series_analyzer') and st.session_state.time_series_analyzer:
                    # Prepare time series data
                    st.session_state.time_series_analyzer.prepare_time_series_data(dataset_name)
                    
                    # Analyze patterns
                    pattern_results = st.session_state.time_series_analyzer.analyze_temporal_patterns(dataset_name)
                    
                    st.success("Time series analysis completed!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Data Points", pattern_results['total_data_points'])
                    with col2:
                        st.metric("Countries", pattern_results['countries_analyzed'])
                    with col3:
                        st.metric("Trend", pattern_results['trend_analysis']['direction'])
                    with col4:
                        st.metric("Volatility", pattern_results['volatility_analysis']['volatility_level'])
                    
                    # Detect anomalies
                    anomaly_results = st.session_state.time_series_analyzer.detect_anomalies(dataset_name)
                    st.subheader("Anomaly Detection")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", anomaly_results['total_anomalies'])
                    with col2:
                        st.metric("Anomaly Rate", f"{anomaly_results['anomaly_rate']:.1f}%")
                else:
                    st.warning("Time series analyzer not available. Using mock results.")
                    
                    # Mock results
                    st.success("Time series analysis completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Data Points", "1,200")
                    with col2:
                        st.metric("Countries", "25")
                    with col3:
                        st.metric("Trend", "Increasing")
                    with col4:
                        st.metric("Volatility", "Medium")
                    
                    st.subheader("Anomaly Detection")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", "12")
                    with col2:
                        st.metric("Anomaly Rate", "1.0%")
                
            except Exception as e:
                st.error(f"Error running time series analysis: {e}")


def show_prescriptive_analytics():
    """Show prescriptive analytics interface"""
    st.header("Prescriptive Analytics")
    
    st.subheader("Resource Allocation Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input("Total Budget", min_value=100000, max_value=10000000, value=1000000)
        min_allocation = st.number_input("Min Allocation", min_value=1000, max_value=100000, value=10000)
    
    with col2:
        max_allocation = st.number_input("Max Allocation", min_value=100000, max_value=1000000, value=200000)
        objective_type = st.selectbox("Objective", ["maximize_benefit", "minimize_cost"])
    
    if st.button("Optimize Resource Allocation"):
        with st.spinner("Optimizing resource allocation..."):
            try:
                if hasattr(st.session_state, 'prescriptive_analyzer') and st.session_state.prescriptive_analyzer:
                    constraints = {
                        'total_budget': total_budget,
                        'min_allocation': min_allocation,
                        'max_allocation': max_allocation
                    }
                    
                    results = st.session_state.prescriptive_analyzer.optimize_resource_allocation(
                        'healthcare_expenditure', constraints, objective_type
                    )
                    
                    if results['success']:
                        st.success("Resource allocation optimization completed!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Cost", f"${results['total_cost']:,.2f}")
                        with col2:
                            st.metric("Objective Value", f"{results['objective_value']:,.2f}")
                        
                        # Show allocations
                        st.subheader("Optimal Allocations")
                        for alloc in results['allocations'][:10]:  # Show top 10
                            st.write(f"**{alloc['country']}**: ${alloc['optimal_allocation']:,.2f}")
                    else:
                        st.error(f"Optimization failed: {results.get('message', 'Unknown error')}")
                else:
                    st.warning("Prescriptive analyzer not available. Using mock results.")
                    
                    # Mock results
                    st.success("Resource allocation optimization completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Cost", f"${total_budget:,.2f}")
                    with col2:
                        st.metric("Objective Value", f"{total_budget * 1.2:,.2f}")
                    
                    st.subheader("Optimal Allocations")
                    mock_allocations = [
                        {"country": "United States", "optimal_allocation": 200000},
                        {"country": "Germany", "optimal_allocation": 150000},
                        {"country": "Japan", "optimal_allocation": 120000},
                        {"country": "United Kingdom", "optimal_allocation": 100000},
                        {"country": "Canada", "optimal_allocation": 80000}
                    ]
                    
                    for alloc in mock_allocations:
                        st.write(f"**{alloc['country']}**: ${alloc['optimal_allocation']:,.2f}")
                
            except Exception as e:
                st.error(f"Error optimizing resource allocation: {e}")