"""
CRISP-DM Workflow Module
Implements the complete CRISP-DM methodology for data mining projects
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime
from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.ui.utils.common import (
    check_system_initialization, 
    display_error_message, 
    display_success_message, 
    display_warning_message,
    get_dataset_names,
    get_dataset_from_managers,
    safe_dataframe_display
)

logger = logging.getLogger(__name__)

class CRISPDMWorkflow:
    """CRISP-DM Workflow Implementation"""
    
    def __init__(self):
        self.current_phase = "Business Understanding"
        self.workflow_data = {
            "Business Understanding": {
                "status": "in_progress",
                "progress": 0,
                "tasks": [
                    "Define business objectives",
                    "Assess situation",
                    "Determine data mining goals",
                    "Produce project plan"
                ],
                "completed_tasks": [],
                "artifacts": {}
            },
            "Data Understanding": {
                "status": "pending",
                "progress": 0,
                "tasks": [
                    "Collect initial data",
                    "Describe data",
                    "Explore data",
                    "Verify data quality"
                ],
                "completed_tasks": [],
                "artifacts": {}
            },
            "Data Preparation": {
                "status": "pending",
                "progress": 0,
                "tasks": [
                    "Select data",
                    "Clean data",
                    "Construct data",
                    "Integrate data",
                    "Format data"
                ],
                "completed_tasks": [],
                "artifacts": {}
            },
            "Modeling": {
                "status": "pending",
                "progress": 0,
                "tasks": [
                    "Select modeling technique",
                    "Generate test design",
                    "Build model",
                    "Assess model"
                ],
                "completed_tasks": [],
                "artifacts": {}
            },
            "Evaluation": {
                "status": "pending",
                "progress": 0,
                "tasks": [
                    "Evaluate results",
                    "Review process",
                    "Determine next steps"
                ],
                "completed_tasks": [],
                "artifacts": {}
            },
            "Deployment": {
                "status": "pending",
                "progress": 0,
                "tasks": [
                    "Plan deployment",
                    "Plan monitoring and maintenance",
                    "Produce final report",
                    "Review project"
                ],
                "completed_tasks": [],
                "artifacts": {}
            }
        }
    
    def render_workflow(self):
        """Render the complete CRISP-DM workflow interface"""
        st.header("CRISP-DM Workflow")
        st.markdown("**Cross-Industry Standard Process for Data Mining**")
        
        # Debug logging
        debug_manager.log_debug("CRISP-DM Workflow rendered", "SYSTEM", {
            "current_phase": self.current_phase,
            "total_phases": len(self.workflow_data),
            "completed_phases": sum(1 for phase in self.workflow_data.values() if phase["status"] == "completed")
        })
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 CRISP-DM Workflow Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("CRISP-DM Workflow", {
                    "Current Phase": self.current_phase,
                    "Total Phases": len(self.workflow_data),
                    "Completed Phases": sum(1 for phase in self.workflow_data.values() if phase["status"] == "completed"),
                    "Workflow Data Keys": list(self.workflow_data.keys()),
                    "Has Data Manager": hasattr(st.session_state, 'data_manager'),
                    "Available Datasets": len(get_dataset_names())
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
                
                # Additional workflow-specific debug info
                st.markdown("---")
                st.subheader("Workflow Status")
                
                for phase_name, phase_data in self.workflow_data.items():
                    status_icon = "✅" if phase_data["status"] == "completed" else "⏳" if phase_data["status"] == "in_progress" else "⭕"
                    st.write(f"{status_icon} **{phase_name}**: {phase_data['status']}")
                
                if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
                    st.markdown("---")
                    st.subheader("Available Datasets")
                datasets = get_dataset_names()
                for dataset in datasets:
                    df = get_dataset_from_managers(dataset)
                    if df is not None:
                        st.write(f"- {dataset}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Workflow overview
        self._render_workflow_overview()
        
        # Phase selection
        selected_phase = st.selectbox(
            "Select Phase",
            list(self.workflow_data.keys()),
            index=list(self.workflow_data.keys()).index(self.current_phase)
        )
        
        # Phase details
        self._render_phase_details(selected_phase)
        
        # Navigation
        self._render_navigation()
    
    def _render_workflow_overview(self):
        """Render workflow overview"""
        st.subheader("Workflow Overview")
        
        # Create phase progress visualization
        phases = list(self.workflow_data.keys())
        progress_values = [self.workflow_data[phase]["progress"] for phase in phases]
        
        fig = go.Figure(data=go.Bar(
            x=phases,
            y=progress_values,
            marker_color=['#ff6b6b' if phase == self.current_phase else '#4ecdc4' for phase in phases]
        ))
        
        fig.update_layout(
            title="CRISP-DM Phase Progress",
            xaxis_title="Phase",
            yaxis_title="Progress (%)",
            height=400
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Phase status table
        status_data = []
        for phase, data in self.workflow_data.items():
            status_data.append({
                "Phase": phase,
                "Status": data["status"].replace("_", " ").title(),
                "Progress": f"{data['progress']}%",
                "Completed Tasks": f"{len(data['completed_tasks'])}/{len(data['tasks'])}"
            })
        
        safe_dataframe_display(pd.DataFrame(status_data), max_rows=20)
    
    def _render_phase_details(self, phase: str):
        """Render detailed phase information"""
        st.subheader(f"{phase} Phase")
        
        phase_data = self.workflow_data[phase]
        
        # Phase status and progress
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", phase_data["status"].replace("_", " ").title())
        with col2:
            st.metric("Progress", f"{phase_data['progress']}%")
        with col3:
            st.metric("Tasks Completed", f"{len(phase_data['completed_tasks'])}/{len(phase_data['tasks'])}")
        
        # Tasks
        st.markdown("### Tasks")
        for i, task in enumerate(phase_data["tasks"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                if task in phase_data["completed_tasks"]:
                    st.success(f"✅ {task}")
                else:
                    st.info(f"⏳ {task}")
            with col2:
                if task not in phase_data["completed_tasks"]:
                    if st.button(f"Complete", key=f"complete_{phase}_{i}"):
                        self._complete_task(phase, task)
        
        # Phase-specific content
        if phase == "Business Understanding":
            self._render_business_understanding()
        elif phase == "Data Understanding":
            self._render_data_understanding()
        elif phase == "Data Preparation":
            self._render_data_preparation()
        elif phase == "Modeling":
            self._render_modeling()
        elif phase == "Evaluation":
            self._render_evaluation()
        elif phase == "Deployment":
            self._render_deployment()
    
    def _render_business_understanding(self):
        """Render Business Understanding phase"""
        st.markdown("### Business Understanding Tools")
        
        # Business objectives
        with st.expander("Define Business Objectives"):
            business_objective = st.text_area(
                "Business Objective",
                value="Improve patient outcomes through predictive analytics",
                help="Describe the main business problem or opportunity"
            )
            
            success_criteria = st.text_area(
                "Success Criteria",
                value="Reduce readmission rates by 15% within 6 months",
                help="Define measurable success criteria"
            )
            
            if st.button("Save Business Objective"):
                self.workflow_data["Business Understanding"]["artifacts"]["business_objective"] = business_objective
                self.workflow_data["Business Understanding"]["artifacts"]["success_criteria"] = success_criteria
                st.success("Business objective saved!")
        
        # Situation assessment
        with st.expander("Assess Situation"):
            current_situation = st.text_area(
                "Current Situation",
                value="High readmission rates affecting patient satisfaction and costs",
                help="Describe the current situation and context"
            )
            
            constraints = st.text_area(
                "Constraints",
                value="Limited budget, regulatory compliance requirements, data privacy concerns",
                help="List any constraints or limitations"
            )
            
            if st.button("Save Situation Assessment"):
                self.workflow_data["Business Understanding"]["artifacts"]["current_situation"] = current_situation
                self.workflow_data["Business Understanding"]["artifacts"]["constraints"] = constraints
                st.success("Situation assessment saved!")
        
        # Data mining goals
        with st.expander("Determine Data Mining Goals"):
            dm_goal = st.selectbox(
                "Data Mining Goal",
                ["Classification", "Regression", "Clustering", "Association Rules", "Time Series Analysis"]
            )
            
            target_variable = st.text_input(
                "Target Variable",
                value="readmission_risk",
                help="The variable you want to predict or analyze"
            )
            
            if st.button("Save Data Mining Goals"):
                self.workflow_data["Business Understanding"]["artifacts"]["dm_goal"] = dm_goal
                self.workflow_data["Business Understanding"]["artifacts"]["target_variable"] = target_variable
                st.success("Data mining goals saved!")
    
    def _render_data_understanding(self):
        """Render Data Understanding phase"""
        st.markdown("### Data Understanding Tools")
        
        # Data collection
        with st.expander("Collect Initial Data"):
            data_sources = st.multiselect(
                "Data Sources",
                ["Electronic Health Records", "Patient Surveys", "Financial Data", "Operational Data", "External Data"],
                default=["Electronic Health Records", "Patient Surveys"]
            )
            
            data_format = st.selectbox(
                "Data Format",
                ["CSV", "Excel", "Database", "API", "JSON"]
            )
            
            if st.button("Save Data Collection Plan"):
                self.workflow_data["Data Understanding"]["artifacts"]["data_sources"] = data_sources
                self.workflow_data["Data Understanding"]["artifacts"]["data_format"] = data_format
                st.success("Data collection plan saved!")
        
        # Data description
        with st.expander("Describe Data"):
            if not check_system_initialization():
                st.error("System not initialized. Please refresh the page.")
                return
            
            datasets = get_dataset_names()
            if not datasets:
                st.warning("No datasets available. Please load datasets first.")
                return
            
            selected_dataset = st.selectbox("Select Dataset", datasets)
            
            if selected_dataset:
                df = get_dataset_from_managers(selected_dataset)
                if df is not None:
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset Info:**")
                        st.write(f"Shape: {df.shape}")
                        st.write(f"Columns: {list(df.columns)}")
                    
                    with col2:
                        st.write("**Data Types:**")
                        st.write(df.dtypes)
                    
                    if st.button("Save Data Description"):
                        self.workflow_data["Data Understanding"]["artifacts"]["dataset_info"] = {
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "dtypes": df.dtypes.to_dict()
                        }
                        st.success("Data description saved!")
            else:
                st.warning("No data manager available")
        
        # Data exploration
        with st.expander("Explore Data"):
            if st.session_state.get('data_manager'):
                datasets = get_dataset_names()
                selected_dataset = st.selectbox("Select Dataset for Exploration", datasets, key="explore_dataset")
                
                if selected_dataset:
                    df = get_dataset_from_managers(selected_dataset)
                    if df is not None:
                        # Basic statistics
                        st.write("**Basic Statistics:**")
                        safe_dataframe_display(df.describe(), max_rows=20)
                        
                        # Missing values
                        st.write("**Missing Values:**")
                        missing_data = df.isnull().sum()
                        missing_df = missing_data[missing_data > 0].to_frame('Missing Count')
                        if not missing_df.empty:
                            safe_dataframe_display(missing_df, max_rows=20)
                        else:
                            st.write("No missing values found.")
                        
                        # Correlation matrix
                        if len(df.select_dtypes(include=[np.number]).columns) > 1:
                            st.write("**Correlation Matrix:**")
                            corr_matrix = df.select_dtypes(include=[np.number]).corr()
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                            st.plotly_chart(fig, width="stretch")
            else:
                st.warning("No data manager available")
    
    def _render_data_preparation(self):
        """Render Data Preparation phase"""
        st.markdown("### Data Preparation Tools")
        
        # Data selection
        with st.expander("Select Data"):
            if not check_system_initialization():
                st.error("System not initialized. Please refresh the page.")
                return
            
            datasets = get_dataset_names()
            if not datasets:
                st.warning("No datasets available. Please load datasets first.")
                return
            
            selected_dataset = st.selectbox("Select Dataset", datasets, key="prep_dataset")
            
            if selected_dataset:
                df = get_dataset_from_managers(selected_dataset)
                if df is not None:
                    
                    # Column selection
                    selected_columns = st.multiselect(
                        "Select Columns",
                        df.columns,
                        default=df.columns[:5]  # Default to first 5 columns
                    )
                    
                    # Row filtering
                    filter_option = st.selectbox(
                        "Row Filtering",
                        ["All rows", "First N rows", "Random sample", "Custom filter"]
                    )
                    
                    if filter_option == "First N rows":
                        n_rows = st.number_input("Number of rows", min_value=1, max_value=len(df), value=1000)
                        filtered_df = df[selected_columns].head(n_rows)
                    elif filter_option == "Random sample":
                        sample_size = st.number_input("Sample size", min_value=1, max_value=len(df), value=1000)
                        filtered_df = df[selected_columns].sample(n=sample_size)
                    else:
                        filtered_df = df[selected_columns]
                    
                    st.write(f"Selected data shape: {filtered_df.shape}")
                    
                    if st.button("Save Selected Data"):
                        self.workflow_data["Data Preparation"]["artifacts"]["selected_data"] = {
                            "columns": selected_columns,
                            "shape": filtered_df.shape,
                            "filter_option": filter_option
                        }
                        st.success("Data selection saved!")
            else:
                st.warning("No data manager available")
        
        # Data cleaning
        with st.expander("Clean Data"):
            cleaning_options = st.multiselect(
                "Cleaning Operations",
                ["Handle missing values", "Remove duplicates", "Handle outliers", "Standardize formats"]
            )
            
            if "Handle missing values" in cleaning_options:
                missing_strategy = st.selectbox(
                    "Missing Value Strategy",
                    ["Drop rows", "Drop columns", "Fill with mean", "Fill with median", "Fill with mode"]
                )
            
            if "Handle outliers" in cleaning_options:
                outlier_method = st.selectbox(
                    "Outlier Detection Method",
                    ["IQR method", "Z-score method", "Isolation Forest"]
                )
            
            if st.button("Apply Cleaning Operations"):
                st.info("Data cleaning operations would be applied here")
                self.workflow_data["Data Preparation"]["artifacts"]["cleaning_operations"] = cleaning_options
                st.success("Cleaning operations saved!")
        
        # Data construction
        with st.expander("Construct Data"):
            st.write("**Feature Engineering:**")
            
            # Create new features
            new_features = st.text_area(
                "New Features to Create",
                value="age_group = categorize_age(age)\nrisk_score = calculate_risk(features)",
                help="Define new features to create"
            )
            
            # Data transformation
            transformation_options = st.multiselect(
                "Transformations",
                ["Normalization", "Standardization", "Log transformation", "One-hot encoding"]
            )
            
            if st.button("Save Data Construction Plan"):
                self.workflow_data["Data Preparation"]["artifacts"]["new_features"] = new_features
                self.workflow_data["Data Preparation"]["artifacts"]["transformations"] = transformation_options
                st.success("Data construction plan saved!")
    
    def _render_modeling(self):
        """Render Modeling phase"""
        st.markdown("### Modeling Tools")
        
        # Model selection
        with st.expander("Select Modeling Technique"):
            model_type = st.selectbox(
                "Model Type",
                ["Classification", "Regression", "Clustering", "Association Rules"]
            )
            
            if model_type == "Classification":
                algorithms = st.multiselect(
                    "Algorithms",
                    ["Random Forest", "Logistic Regression", "SVM", "Neural Network", "Decision Tree"]
                )
            elif model_type == "Regression":
                algorithms = st.multiselect(
                    "Algorithms",
                    ["Linear Regression", "Random Forest", "SVM", "Neural Network", "XGBoost"]
                )
            elif model_type == "Clustering":
                algorithms = st.multiselect(
                    "Algorithms",
                    ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"]
                )
            else:  # Association Rules
                algorithms = st.multiselect(
                    "Algorithms",
                    ["Apriori", "FP-Growth", "Eclat"]
                )
            
            if st.button("Save Model Selection"):
                self.workflow_data["Modeling"]["artifacts"]["model_type"] = model_type
                self.workflow_data["Modeling"]["artifacts"]["algorithms"] = algorithms
                st.success("Model selection saved!")
        
        # Test design
        with st.expander("Generate Test Design"):
            test_method = st.selectbox(
                "Test Method",
                ["Train/Test Split", "Cross-Validation", "Time Series Split"]
            )
            
            if test_method == "Train/Test Split":
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            elif test_method == "Cross-Validation":
                cv_folds = st.number_input("CV Folds", 3, 10, 5)
            
            validation_strategy = st.selectbox(
                "Validation Strategy",
                ["Hold-out", "Cross-validation", "Bootstrap"]
            )
            
            if st.button("Save Test Design"):
                self.workflow_data["Modeling"]["artifacts"]["test_method"] = test_method
                self.workflow_data["Modeling"]["artifacts"]["validation_strategy"] = validation_strategy
                st.success("Test design saved!")
        
        # Model building
        with st.expander("Build Model"):
            if st.button("Start Model Training"):
                st.info("Model training would start here")
                # This would integrate with the existing model training functionality
                st.success("Model training initiated!")
    
    def _render_evaluation(self):
        """Render Evaluation phase"""
        st.markdown("### Evaluation Tools")
        
        # Results evaluation
        with st.expander("Evaluate Results"):
            evaluation_metrics = st.multiselect(
                "Evaluation Metrics",
                ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "RMSE", "MAE"]
            )
            
            business_criteria = st.text_area(
                "Business Criteria",
                value="Model should achieve >85% accuracy and reduce false positives by 20%",
                help="Define business-specific evaluation criteria"
            )
            
            if st.button("Save Evaluation Criteria"):
                self.workflow_data["Evaluation"]["artifacts"]["metrics"] = evaluation_metrics
                self.workflow_data["Evaluation"]["artifacts"]["business_criteria"] = business_criteria
                st.success("Evaluation criteria saved!")
        
        # Process review
        with st.expander("Review Process"):
            process_issues = st.text_area(
                "Process Issues",
                value="Data quality issues identified in preparation phase",
                help="Document any issues encountered during the process"
            )
            
            lessons_learned = st.text_area(
                "Lessons Learned",
                value="Feature engineering significantly improved model performance",
                help="Document key learnings from the project"
            )
            
            if st.button("Save Process Review"):
                self.workflow_data["Evaluation"]["artifacts"]["process_issues"] = process_issues
                self.workflow_data["Evaluation"]["artifacts"]["lessons_learned"] = lessons_learned
                st.success("Process review saved!")
    
    def _render_deployment(self):
        """Render Deployment phase"""
        st.markdown("### Deployment Tools")
        
        # Deployment plan
        with st.expander("Plan Deployment"):
            deployment_type = st.selectbox(
                "Deployment Type",
                ["Batch Processing", "Real-time", "API Service", "Dashboard Integration"]
            )
            
            deployment_environment = st.selectbox(
                "Environment",
                ["Development", "Staging", "Production"]
            )
            
            rollback_plan = st.text_area(
                "Rollback Plan",
                value="Maintain previous model version for quick rollback if issues arise",
                help="Define rollback strategy"
            )
            
            if st.button("Save Deployment Plan"):
                self.workflow_data["Deployment"]["artifacts"]["deployment_type"] = deployment_type
                self.workflow_data["Deployment"]["artifacts"]["environment"] = deployment_environment
                self.workflow_data["Deployment"]["artifacts"]["rollback_plan"] = rollback_plan
                st.success("Deployment plan saved!")
        
        # Monitoring plan
        with st.expander("Plan Monitoring and Maintenance"):
            monitoring_metrics = st.multiselect(
                "Monitoring Metrics",
                ["Model Performance", "Data Drift", "Prediction Latency", "Error Rates"]
            )
            
            maintenance_schedule = st.selectbox(
                "Maintenance Schedule",
                ["Daily", "Weekly", "Monthly", "Quarterly"]
            )
            
            if st.button("Save Monitoring Plan"):
                self.workflow_data["Deployment"]["artifacts"]["monitoring_metrics"] = monitoring_metrics
                self.workflow_data["Deployment"]["artifacts"]["maintenance_schedule"] = maintenance_schedule
                st.success("Monitoring plan saved!")
    
    def _render_navigation(self):
        """Render navigation controls"""
        st.markdown("### Navigation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Previous Phase"):
                phases = list(self.workflow_data.keys())
                current_index = phases.index(self.current_phase)
                if current_index > 0:
                    self.current_phase = phases[current_index - 1]
                    st.rerun()
        
        with col2:
            if st.button("Next Phase"):
                phases = list(self.workflow_data.keys())
                current_index = phases.index(self.current_phase)
                if current_index < len(phases) - 1:
                    self.current_phase = phases[current_index + 1]
                    st.rerun()
        
        with col3:
            if st.button("Reset Workflow"):
                self.__init__()
                st.rerun()
    
    def _complete_task(self, phase: str, task: str):
        """Complete a task in a phase"""
        if task not in self.workflow_data[phase]["completed_tasks"]:
            self.workflow_data[phase]["completed_tasks"].append(task)
            
            # Update progress
            total_tasks = len(self.workflow_data[phase]["tasks"])
            completed_tasks = len(self.workflow_data[phase]["completed_tasks"])
            self.workflow_data[phase]["progress"] = int((completed_tasks / total_tasks) * 100)
            
            # Update status
            if self.workflow_data[phase]["progress"] == 100:
                self.workflow_data[phase]["status"] = "completed"
            else:
                self.workflow_data[phase]["status"] = "in_progress"
            
            # Debug logging
            debug_manager.log_debug(f"Task completed: {task} in {phase}", "SYSTEM", {
                "phase": phase,
                "task": task,
                "progress": self.workflow_data[phase]["progress"],
                "status": self.workflow_data[phase]["status"]
            })
            
            st.success(f"Task '{task}' completed!")
            st.rerun()


def show_crisp_dm_workflow():
    """Show CRISP-DM workflow interface"""
    if 'crisp_dm_workflow' not in st.session_state:
        st.session_state.crisp_dm_workflow = CRISPDMWorkflow()
    
    workflow = st.session_state.crisp_dm_workflow
    workflow.render_workflow()
