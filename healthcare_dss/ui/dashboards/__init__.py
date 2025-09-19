"""
Dashboard Components Module
Contains role-based dashboard functions
"""

# Clinical Dashboard
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

# Executive Dashboard
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

# Financial Dashboard
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

# Department Dashboard
from healthcare_dss.ui.dashboards.department_dashboard import (
    show_department_dashboard,
    show_staff_scheduling,
    show_resource_allocation,
    show_department_performance,
    show_budget_management,
    show_quality_metrics
)

# Clinical Staff Dashboard
from healthcare_dss.ui.dashboards.clinical_staff_dashboard import (
    show_clinical_staff_dashboard,
    show_patient_care_tools,
    show_clinical_decision_support,
    show_medication_management,
    show_documentation_tools,
    show_workflow_management
)

# Clinical Staff Functions
from healthcare_dss.ui.dashboards.clinical_staff_functions import (
    show_patient_assessment,
    show_treatment_recommendations,
    show_clinical_guidelines,
    show_patient_care_tools,
    show_clinical_decision_support,
    generate_clinical_recommendations,
    get_relevant_guidelines,
    check_drug_interactions,
    calculate_cardiovascular_risk,
    get_ckd_stage
)

# Data Analyst Dashboard
from healthcare_dss.ui.dashboards.data_analyst_dashboard import (
    show_data_analyst_dashboard,
    show_data_exploration,
    show_statistical_analysis,
    show_machine_learning_tools,
    show_data_visualization
)

# Management Dashboards
from healthcare_dss.ui.dashboards.data_management import show_data_management
from healthcare_dss.ui.dashboards.model_management import show_model_management

# Model Training Dashboard
from healthcare_dss.ui.dashboards.model_training import show_model_training

# Model Training Components
from healthcare_dss.ui.dashboards.model_training_data_preprocessing import (
    prepare_data,
    detect_data_leakage,
    estimate_feature_importance_pre_training,
    detect_leakage_by_importance,
    filter_suspicious_features,
    apply_preprocessing_pipeline,
    apply_legacy_preprocessing
)
from healthcare_dss.ui.dashboards.model_training_model_creation import (
    get_model_configurations,
    create_model
)
from healthcare_dss.ui.dashboards.model_training_metrics import (
    calculate_metrics,
    get_feature_importance,
    display_training_summary
)

__all__ = [
    # Clinical Dashboard
    "show_clinical_dashboard",
    "show_patient_flow_management",
    "show_quality_safety_monitoring",
    "show_resource_allocation_guidance",
    "show_strategic_planning",
    "show_performance_management",
    "show_clinical_analytics",
    "show_outcome_analysis",
    "show_risk_assessment",
    "show_compliance_monitoring",
    
    # Executive Dashboard
    "show_executive_dashboard",
    "show_regulatory_compliance",
    "show_resource_planning",
    "show_kpi_dashboard",
    "show_financial_overview",
    "show_operational_analytics",
    "show_risk_management",
    "show_stakeholder_reports",
    
    # Financial Dashboard
    "show_financial_dashboard",
    "show_cost_analysis",
    "show_budget_tracking",
    "show_revenue_cycle",
    "show_investment_analysis",
    "show_financial_reporting",
    "show_cost_optimization",
    "show_revenue_forecasting",
    "show_financial_risk_analysis",
    "show_budget_planning",
    
    # Department Dashboard
    "show_department_dashboard",
    "show_staff_scheduling",
    "show_resource_allocation",
    "show_department_performance",
    "show_budget_management",
    "show_quality_metrics",
    
    # Clinical Staff Dashboard
    "show_clinical_staff_dashboard",
    "show_patient_care_tools",
    "show_clinical_decision_support",
    "show_medication_management",
    "show_documentation_tools",
    "show_workflow_management",
    
    # Clinical Staff Functions
    "show_patient_assessment",
    "show_treatment_recommendations",
    "show_clinical_guidelines",
    "show_patient_care_tools",
    "show_clinical_decision_support",
    "generate_clinical_recommendations",
    "get_relevant_guidelines",
    "check_drug_interactions",
    "calculate_cardiovascular_risk",
    "get_ckd_stage",
    
    # Data Analyst Dashboard
    "show_data_analyst_dashboard",
    "show_data_exploration",
    "show_statistical_analysis",
    "show_machine_learning_tools",
    "show_data_visualization",
    
    # Management Dashboards
    "show_data_management",
    "show_model_management",
    
    # Model Training Dashboard
    "show_model_training",
    
    # Model Training Components
    "prepare_data",
    "detect_data_leakage",
    "estimate_feature_importance_pre_training",
    "detect_leakage_by_importance",
    "filter_suspicious_features",
    "apply_preprocessing_pipeline",
    "apply_legacy_preprocessing",
    "get_model_configurations",
    "create_model",
    "calculate_metrics",
    "get_feature_importance",
    "display_training_summary"
]
