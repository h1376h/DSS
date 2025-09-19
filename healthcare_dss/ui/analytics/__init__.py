"""
Analytics Components Module
Contains all analytics-related functions for the Streamlit UI
"""

from healthcare_dss.ui.analytics.association_rules import show_association_rules
from healthcare_dss.ui.analytics.clustering import show_clustering_analysis
from healthcare_dss.ui.analytics.prescriptive import show_prescriptive_analytics
from healthcare_dss.ui.analytics.analytics_dashboard import show_analytics_dashboard, show_analytics_overview
from healthcare_dss.ui.analytics.advanced_analytics_module import (
    show_advanced_analytics,
    show_time_series_analysis,
    show_optimization_models,
    show_simulation_capabilities,
    show_ensemble_modeling
)
from healthcare_dss.ui.analytics.advanced_analytics import (
    show_statistical_analysis,
    show_data_visualization,
    show_machine_learning_pipeline
)
