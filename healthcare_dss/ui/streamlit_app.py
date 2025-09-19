"""
Healthcare Decision Support System - Streamlit Dashboard
Main application file with modular architecture
"""

import os
import sys
import warnings

# CRITICAL: Suppress warnings BEFORE any other imports
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

# Fix SSL certificate issues for dataset downloads
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress all deprecation warnings at the system level
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Parsing dates involving a day of month without a year.*")
warnings.filterwarnings("ignore", message=".*ipykernel.comm.Comm.*")
warnings.filterwarnings("ignore", message=".*Jupyter is migrating its paths.*")
warnings.filterwarnings("ignore", message=".*The `ipykernel.comm.Comm` class has been deprecated.*")
warnings.filterwarnings("ignore", message=".*datetime.strptime.*")
warnings.filterwarnings("ignore", module="ipykernel.*")
warnings.filterwarnings("ignore", module="dash.*")
warnings.filterwarnings("ignore", module="jupyter.*")

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the main HealthcareDSS class
try:
    from main import HealthcareDSS
except ImportError:
    # Fallback: import components directly
    from healthcare_dss import DataManager, ModelManager, KnowledgeManager
    from healthcare_dss.core.preprocessing_engine import PreprocessingEngine
    
    class HealthcareDSS:
        """Fallback HealthcareDSS class"""
        def __init__(self):
            self.data_manager = DataManager(data_dir="datasets/raw")
            self.model_manager = ModelManager(self.data_manager)
            self.knowledge_manager = KnowledgeManager(self.data_manager, self.model_manager)

# Import UI components
from healthcare_dss.ui.utils.common import check_system_initialization
from healthcare_dss.ui.analytics import (
    show_association_rules,
    show_clustering_analysis,
    show_prescriptive_analytics,
    show_analytics_dashboard,
    show_analytics_overview,
    show_advanced_analytics,
    show_time_series_analysis,
    show_optimization_models,
    show_simulation_capabilities,
    show_ensemble_modeling,
    show_statistical_analysis,
    show_data_visualization,
    show_machine_learning_pipeline
)
from healthcare_dss.ui.dashboards import (
    show_data_management,
    show_model_management
)
from healthcare_dss.ui.dashboards.model_training import show_model_training
from healthcare_dss.ui.dashboards.clinical_dashboard import (
    show_clinical_dashboard,
    show_patient_flow_management,
    show_quality_safety_monitoring,
    show_resource_allocation_guidance,
    show_strategic_planning,
    show_performance_management,
    show_clinical_analytics,
    show_outcome_analysis,
    show_risk_assessment,
    show_compliance_monitoring
)
from healthcare_dss.ui.dashboards.executive_dashboard import (
    show_executive_dashboard,
    show_regulatory_compliance,
    show_resource_planning,
    show_kpi_dashboard,
    show_financial_overview,
    show_operational_analytics,
    show_risk_management,
    show_stakeholder_reports
)
from healthcare_dss.ui.dashboards.financial_dashboard import (
    show_financial_dashboard,
    show_cost_analysis,
    show_budget_tracking,
    show_revenue_cycle,
    show_investment_analysis,
    show_financial_reporting,
    show_cost_optimization,
    show_revenue_forecasting,
    show_financial_risk_analysis,
    show_budget_planning
)
from healthcare_dss.ui.dashboards.department_dashboard import (
    show_department_dashboard,
    show_staff_scheduling,
    show_resource_allocation,
    show_department_performance,
    show_budget_management,
    show_quality_metrics
)
from healthcare_dss.ui.dashboards.clinical_staff_dashboard import (
    show_clinical_staff_dashboard,
    show_patient_care_tools,
    show_clinical_decision_support,
    show_medication_management,
    show_documentation_tools,
    show_workflow_management
)
from healthcare_dss.ui.dashboards.clinical_staff_functions import (
    show_patient_assessment,
    show_treatment_recommendations,
    show_clinical_guidelines
)
from healthcare_dss.ui.dashboards.data_analyst_dashboard import (
    show_data_analyst_dashboard,
    show_data_exploration,
    show_machine_learning_tools
)
from healthcare_dss.ui.workflow_views import (
    show_crisp_dm_workflow, show_classification_evaluation, show_association_rules,
    show_clustering_analysis, show_time_series_analysis, show_prescriptive_analytics
)
from healthcare_dss.ui.analytics_views import (
    show_knowledge_management, show_kpi_dashboard
)
from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.config.dashboard_config import config_manager
from healthcare_dss.ui.dynamic_dashboard_manager import dashboard_manager

# Global debug mode configuration
DEBUG_MODE = os.getenv('DSS_DEBUG_MODE', 'false').lower() == 'true'

# Page configuration
st.set_page_config(
    page_title="Healthcare Decision Support System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_dss_system():
    """Initialize the Healthcare DSS system"""
    if not st.session_state.get('initialized', False):
        with st.spinner("Initializing Healthcare Decision Support System..."):
            try:
                # Initialize the DSS system
                dss = HealthcareDSS()
                
                # Store components in session state
                st.session_state.data_manager = dss.data_manager
                st.session_state.model_manager = dss.model_manager
                st.session_state.knowledge_manager = dss.knowledge_manager
                
                # DatasetManager functionality is now integrated into DataManager
                st.session_state.dataset_manager = dss.data_manager
                st.session_state.preprocessing_engine = dss.model_manager.preprocessing_engine
                
                # Initialize workflow components
                st.session_state.crisp_dm_workflow = dss.crisp_dm_workflow
                st.session_state.classification_evaluator = dss.classification_evaluator
                st.session_state.association_rules_miner = dss.association_rules_miner
                st.session_state.clustering_analyzer = dss.clustering_analyzer
                st.session_state.time_series_analyzer = dss.time_series_analyzer
                st.session_state.prescriptive_analyzer = dss.prescriptive_analyzer
                
                st.session_state.initialized = True
                
                # Debug logging
                debug_manager.log_debug("DSS System initialized successfully", "SYSTEM", {
                    "components": ["data_manager", "model_manager", "knowledge_manager", "preprocessing_engine"],
                    "initialization_time": datetime.now().isoformat()
                })
                
                st.success("Healthcare DSS system initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize DSS system: {str(e)}")
                debug_manager.render_error_debug(e, "DSS System Initialization")
                st.session_state.initialized = False


def create_sidebar():
    """Create the sidebar with navigation and debug panel"""
    # Get system configuration
    system_config = config_manager.get_system_config()
    
    st.sidebar.markdown("### Main Sections")
    
    # Role-based dashboard selection using configuration
    available_roles = config_manager.get_all_roles()
    user_role = st.sidebar.selectbox(
        "Select Your Role",
        available_roles,
        key="user_role_selector"
    )
    
    # Store user role in session state
    st.session_state.user_role = user_role
    
    # Get role-specific pages from configuration
    role_config = config_manager.get_role_config(user_role)
    page_options = role_config.pages
    
    # Enhanced role-based page options (from old implementation)
    if user_role == "Clinical Leadership":
        page_options = [
            "Clinical Dashboard",
            "Patient Flow Management",
            "Quality & Safety Monitoring",
            "Resource Allocation Guidance",
            "Data Management",
            "Model Management"
        ]
    elif user_role == "Administrative Executive":
        page_options = [
            "Executive Dashboard",
            "Strategic Planning",
            "Performance Management",
            "Regulatory Compliance",
            "Resource Planning",
            "KPI Dashboard"
        ]
    elif user_role == "Financial Manager":
        page_options = [
            "Financial Dashboard",
            "Cost Analysis",
            "Budget Tracking",
            "Revenue Cycle",
            "Investment Analysis",
            "Prescriptive Analytics"
        ]
    elif user_role == "Department Manager":
        page_options = [
            "Department Dashboard",
            "Staff Scheduling",
            "Resource Utilization",
            "Patient Satisfaction",
            "Quality Metrics",
            "Time Series Analysis"
        ]
    elif user_role == "Clinical Staff":
        page_options = [
            "Clinical Staff Dashboard",
            "Clinical Decision Support",
            "Patient Assessment",
            "Treatment Recommendations",
            "Clinical Guidelines",
            "Knowledge Management",
            "Classification Evaluation",
            "Model Training"
        ]
    else:  # Data Analyst
        page_options = [
            "Analytics Dashboard",
            "Data Management",
            "Model Management",
            "Model Training",
            "CRISP-DM Workflow",
            "Association Rules",
            "Clustering Analysis"
        ]
    
    # Handle selected page from quick actions
    if 'selected_page' in st.session_state and st.session_state.selected_page in page_options:
        default_index = page_options.index(st.session_state.selected_page)
        # Clear the selected page after using it
        del st.session_state.selected_page
    else:
        default_index = 0
    
    page = st.sidebar.selectbox("Select Page", page_options, index=default_index)
    
    # System status section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    if st.session_state.get('initialized', False):
        st.sidebar.success("System Ready")
        
        # Quick stats in sidebar
        if st.session_state.data_manager:
            # Calculate unique datasets and records
            all_datasets = {}
            all_datasets.update(st.session_state.data_manager.datasets)
            if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
                all_datasets.update(st.session_state.dataset_manager.datasets)
            
            total_datasets = len(all_datasets)
            total_records = sum(len(df) for df in all_datasets.values())
            
            st.sidebar.metric("Datasets", total_datasets)
            st.sidebar.metric("Records", f"{total_records:,}")
    else:
        st.sidebar.warning("System Initializing...")

    # Debug mode toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Settings")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=system_config['debug_mode'], help="Show detailed debugging information")
    
    # Store debug mode in session state
    st.session_state.debug_mode = debug_mode
    
    # Enhanced Debug Panel (better implementation)
    if debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Debug Panel")
        
        # System Status
        if st.session_state.get('initialized', False):
            # Debug actions
            if st.sidebar.button("Show System Logs", width="stretch"):
                st.session_state.show_system_logs = True
                st.rerun()
            
            if st.sidebar.button("Performance Metrics", width="stretch"):
                st.session_state.show_performance_metrics = True
                st.rerun()
        else:
            st.sidebar.error("System Not Ready")
    
    # Help section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Help")
    
    with st.sidebar.expander("Getting Started"):
        st.markdown("""
        **Welcome to the Healthcare DSS**
        
        1. Select your role from the dropdown
        2. Choose a page from the navigation
        3. Use the debug mode for technical details
        4. Refresh the system if needed
        """)
    
    with st.sidebar.expander("System Info"):
        st.markdown(f"""
        **Version:** 2.0.0  
        **Mode:** Dashboard  
        **Debug:** {'Enabled' if debug_mode else 'Disabled'}  
        **Role:** {user_role}
        **Theme:** {system_config['theme']}
        **Language:** {system_config['language']}
        """)
    
    return page


def show_placeholder_page(page_name: str):
    """Show placeholder page for unimplemented functionality"""
    st.header(page_name)
    st.markdown(f"**{page_name} functionality is being developed**")
    
    st.info(f"The {page_name} page is currently under development. This functionality will be available in a future update.")
    
    # Show some sample content based on page type
    if "Dashboard" in page_name:
        st.subheader("Sample Dashboard Content")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Metric 1", "100")
        with col2:
            st.metric("Sample Metric 2", "85%")
        with col3:
            st.metric("Sample Metric 3", "42")
    
    elif "Analysis" in page_name or "Analytics" in page_name:
        st.subheader("Sample Analysis")
        st.write("This page will contain advanced analytics and data analysis tools.")
        
    elif "Management" in page_name:
        st.subheader("Sample Management Tools")
        st.write("This page will contain management and administrative tools.")


def main():
    """Main application function"""
    # Initialize the DSS system
    initialize_dss_system()
    
    # Create sidebar and get selected page
    page = create_sidebar()
    
    # Handle debug panel actions
    if st.session_state.get('show_system_logs', False):
        st.session_state.show_system_logs = False  # Clear the flag
        show_system_logs()
        return
    
    if st.session_state.get('show_performance_metrics', False):
        st.session_state.show_performance_metrics = False  # Clear the flag
        show_performance_metrics()
        return
    
    # Page routing - Clinical Leadership
    if page == "Clinical Dashboard":
        show_clinical_dashboard()
    elif page == "Patient Flow Management":
        show_patient_flow_management()
    elif page == "Quality & Safety Monitoring":
        show_quality_safety_monitoring()
    elif page == "Resource Allocation Guidance":
        show_resource_allocation_guidance()
    elif page == "Strategic Planning":
        show_strategic_planning()
    elif page == "Performance Management":
        show_performance_management()
    elif page == "Clinical Analytics":
        show_clinical_analytics()
    elif page == "Outcome Analysis":
        show_outcome_analysis()
    elif page == "Risk Assessment":
        show_risk_assessment()
    elif page == "Compliance Monitoring":
        show_compliance_monitoring()
    
    # Page routing - Administrative Executive
    elif page == "Executive Dashboard":
        show_executive_dashboard()
    elif page == "Regulatory Compliance":
        show_regulatory_compliance()
    elif page == "Resource Planning":
        show_resource_planning()
    elif page == "KPI Dashboard":
        show_kpi_dashboard()
    elif page == "Financial Overview":
        show_financial_overview()
    elif page == "Operational Analytics":
        show_operational_analytics()
    elif page == "Risk Management":
        show_risk_management()
    elif page == "Stakeholder Reports":
        show_stakeholder_reports()
    
    # Page routing - Financial Manager
    elif page == "Financial Dashboard":
        show_financial_dashboard()
    elif page == "Cost Analysis":
        show_cost_analysis()
    elif page == "Budget Tracking":
        show_budget_tracking()
    elif page == "Revenue Cycle":
        show_revenue_cycle()
    elif page == "Investment Analysis":
        show_investment_analysis()
    elif page == "Financial Reporting":
        show_financial_reporting()
    elif page == "Cost Optimization":
        show_cost_optimization()
    elif page == "Revenue Forecasting":
        show_revenue_forecasting()
    elif page == "Financial Risk Analysis":
        show_financial_risk_analysis()
    elif page == "Department Dashboard":
        show_department_dashboard()
    elif page == "Staff Scheduling":
        show_staff_scheduling()
    elif page == "Resource Allocation":
        show_resource_allocation()
    elif page == "Performance Management":
        show_department_performance()
    elif page == "Budget Management":
        show_budget_management()
    elif page == "Quality Metrics":
        show_quality_metrics()
    
    # Clinical Staff pages
    elif page == "Clinical Staff Dashboard":
        show_clinical_staff_dashboard()
    elif page == "Patient Care Tools":
        show_patient_care_tools()
    elif page == "Clinical Decision Support":
        from healthcare_dss.ui.dashboards.clinical_staff_functions import show_clinical_decision_support
        show_clinical_decision_support()
    elif page == "Patient Assessment":
        show_patient_assessment()
    elif page == "Treatment Recommendations":
        show_treatment_recommendations()
    elif page == "Clinical Guidelines":
        show_clinical_guidelines()
    elif page == "Medication Management":
        show_medication_management()
    elif page == "Documentation Tools":
        show_documentation_tools()
    elif page == "Workflow Management":
        show_workflow_management()
    
    # Data Analyst pages
    elif page == "Data Analyst Dashboard":
        show_data_analyst_dashboard()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Statistical Analysis":
        show_statistical_analysis()
    elif page == "Machine Learning Tools":
        show_machine_learning_tools()
    elif page == "Data Visualization":
        show_data_visualization()
    elif page == "Budget Planning":
        show_budget_planning()
    
    # Page routing - Data Management and Analytics
    elif page == "Data Management":
        show_data_management()
    elif page == "Model Management":
        show_model_management()
    elif page == "Model Training":
        show_model_training()
    elif page == "Analytics Dashboard":
        show_analytics_dashboard()
    elif page == "Association Rules":
        show_association_rules()
    elif page == "Clustering Analysis":
        show_clustering_analysis()
    elif page == "Prescriptive Analytics":
        show_prescriptive_analytics()
    elif page == "Advanced Analytics":
        show_advanced_analytics()
    elif page == "Time Series Analysis":
        show_time_series_analysis()
    elif page == "Optimization Models":
        show_optimization_models()
    elif page == "Simulation Capabilities":
        show_simulation_capabilities()
    elif page == "Ensemble Modeling":
        show_ensemble_modeling()
    
    # Page routing - Workflow Views
    elif page == "CRISP-DM Workflow":
        show_crisp_dm_workflow()
    elif page == "Classification Evaluation":
        show_classification_evaluation()
    elif page == "Knowledge Management":
        show_knowledge_management()
    elif page == "KPI Dashboard":
        show_kpi_dashboard()
    
    # Page routing - Dynamic Dashboard Manager
    elif page == "Dynamic Dashboard Manager":
        show_dynamic_dashboard_manager()
    
    else:
        # Show placeholder for unimplemented pages
        show_placeholder_page(page)


def show_system_logs():
    """Show system logs and debug information"""
    st.header("System Logs & Debug Information")
    
    # System Status
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Initialized", "Yes" if st.session_state.get('initialized', False) else "No")
    
    with col2:
        st.metric("Debug Mode", "Enabled" if st.session_state.get('debug_mode', False) else "Disabled")
    
    with col3:
        st.metric("User Role", st.session_state.get('user_role', 'Not Set'))
    
    # Data Manager Status
    st.subheader("Data Manager Status")
    if st.session_state.get('data_manager'):
        st.success("Data Manager Available")
        
        # Show datasets
        datasets = st.session_state.data_manager.datasets
        st.write(f"**Total Datasets:** {len(datasets)}")
        
        for name, df in datasets.items():
            st.write(f"**{name}**: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.error("Data Manager Not Available")
    
    # Model Manager Status
    st.subheader("Model Manager Status")
    if st.session_state.get('model_manager'):
        st.success("Model Manager Available")
    else:
        st.warning("Model Manager Not Available")
    
    # Session State Information
    st.subheader("Session State")
    session_data = {k: v for k, v in st.session_state.items() if not k.startswith('_')}
    st.json(session_data)


def show_performance_metrics():
    """Show performance metrics and system information"""
    st.header("Performance Metrics & System Information")
    
    # System Performance
    st.subheader("System Performance")
    
    # Get performance metrics from debug manager
    if hasattr(debug_manager, 'performance_metrics'):
        metrics = debug_manager.performance_metrics
        if metrics:
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    st.metric(metric_name, f"{metric_data['value']:.3f} {metric_data.get('unit', '')}")
        else:
            st.info("No performance metrics available yet")
    
    # System Information
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate uptime from debug manager start time
        uptime = time.time() - debug_manager.start_time
        st.metric("Uptime", f"{uptime:.1f}s")
        st.metric("Debug Log Entries", len(debug_manager.debug_log))
    
    with col2:
        st.metric("Query Log Entries", len(debug_manager.query_log))
        st.metric("Model Training Log Entries", len(debug_manager.model_training_log))
    
    # Recent Debug Log
    st.subheader("Recent Debug Log")
    if debug_manager.debug_log:
        for entry in list(debug_manager.debug_log)[-10:]:  # Show last 10 entries
            st.text(entry)
    else:
        st.info("No debug log entries yet")


def show_dynamic_dashboard_manager():
    """Show dynamic dashboard manager interface"""
    st.header("Dynamic Dashboard Manager")
    st.markdown("**Configure and manage dynamic dashboards without hardcoded logic**")
    
    # System capabilities
    capabilities = dashboard_manager.get_system_capabilities()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Roles", capabilities["total_roles"])
    with col2:
        st.metric("Total Dashboards", capabilities["total_dashboards"])
    with col3:
        st.metric("Total Components", capabilities["total_components"])
    with col4:
        st.metric("Current Role", st.session_state.get('user_role', 'Not Set'))
    
    # Role management
    st.subheader("Role Management")
    
    selected_role = st.selectbox(
        "Select Role to Configure",
        capabilities["available_roles"],
        key="role_selector"
    )
    
    if selected_role:
        role_config = dashboard_manager.get_role_config(selected_role)
        if role_config:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Role:** {role_config.display_name}")
                st.write(f"**Description:** {role_config.description}")
                st.write(f"**Default Dashboard:** {role_config.default_dashboard.value}")
            
            with col2:
                st.write("**Permissions:**")
                for perm in role_config.permissions:
                    st.write(f"• {perm}")
                
                st.write("**Available Dashboards:**")
                for dash_type in role_config.available_dashboards:
                    dash_config = dashboard_manager.get_dashboard_config(dash_type)
                    if dash_config:
                        st.write(f"• {dash_config.name}")
    
    # Component management
    st.subheader("Component Management")
    
    components = dashboard_manager.get_components_for_role(selected_role or "data_analyst")
    
    if components:
        st.write(f"**Available Components for {selected_role or 'data_analyst'}:**")
        
        for component in components:
            with st.expander(f"{component.name} ({component.component_type.value})"):
                st.write(f"**Description:** {component.description}")
                st.write(f"**Required Permissions:** {', '.join(component.required_permissions)}")
                st.write(f"**Data Requirements:** {', '.join(component.data_requirements)}")
                st.write(f"**Configuration:** {component.config}")
    
    # Custom dashboard creation
    st.subheader("Create Custom Dashboard")
    
    if selected_role:
        available_components = [comp.id for comp in dashboard_manager.get_components_for_role(selected_role)]
        
        if available_components:
            selected_components = st.multiselect(
                "Select Components",
                available_components,
                key="custom_components"
            )
            
            dashboard_name = st.text_input(
                "Dashboard Name",
                value=f"Custom Dashboard for {selected_role}",
                key="custom_dashboard_name"
            )
            
            if st.button("Create Custom Dashboard"):
                if selected_components and dashboard_name:
                    custom_dashboard = dashboard_manager.create_custom_dashboard(
                        selected_role, dashboard_name, selected_components
                    )
                    
                    if custom_dashboard:
                        st.success(f"Custom dashboard '{dashboard_name}' created successfully!")
                        
                        # Show dashboard configuration
                        st.subheader("Dashboard Configuration")
                        st.json({
                            "name": custom_dashboard.name,
                            "description": custom_dashboard.description,
                            "components": custom_dashboard.components,
                            "layout": custom_dashboard.layout
                        })
                    else:
                        st.error("Failed to create custom dashboard")
                else:
                    st.warning("Please select components and enter a dashboard name")
    
    # System information
    st.subheader("System Information")
    
    with st.expander("Available Roles"):
        for role_name in capabilities["available_roles"]:
            role_config = dashboard_manager.get_role_config(role_name)
            st.write(f"**{role_config.display_name}:** {role_config.description}")
    
    with st.expander("Available Dashboards"):
        for dash_type in capabilities["available_dashboards"]:
            dash_config = dashboard_manager.get_dashboard_config(dash_type)
            if dash_config:
                st.write(f"**{dash_config.name}:** {dash_config.description}")
    
    with st.expander("Available Components"):
        for comp_id in capabilities["available_components"]:
            component = dashboard_manager.components[comp_id]
            st.write(f"**{component.name}:** {component.description}")
    
    # Export configuration
    st.subheader("Configuration Management")
    
    if st.button("Export Configuration"):
        config_data = dashboard_manager.get_system_capabilities()
        st.download_button(
            label="Download Configuration",
            data=json.dumps(config_data, indent=2),
            file_name="dashboard_configuration.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()