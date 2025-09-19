"""
UI Utilities Module
Contains common utilities and helper functions for the Streamlit UI
"""

# Common utilities
from healthcare_dss.ui.utils.common import (
    check_system_initialization,
    display_dataset_info,
    get_available_datasets,
    display_error_message,
    display_success_message,
    display_warning_message,
    create_metric_columns,
    safe_dataframe_display,
    get_numeric_columns,
    get_categorical_columns,
    get_time_columns,
    validate_dataset_selection,
    create_analysis_summary
)

# Data helpers
from healthcare_dss.ui.utils.data_helpers import (
    prepare_data_for_analysis,
    detect_outliers_iqr,
    calculate_correlation_matrix,
    find_strong_correlations,
    calculate_data_quality_metrics,
    create_binary_matrix,
    find_frequent_itemsets,
    generate_association_rules
)

# Visualization utilities
from healthcare_dss.ui.utils.visualization import (
    create_histogram,
    create_scatter_plot,
    create_bar_chart,
    create_line_chart,
    create_correlation_heatmap,
    create_cluster_visualization,
    create_trend_plot,
    create_seasonal_decomposition,
    create_forecast_plot,
    create_anomaly_plot,
    create_feature_importance_plot,
    create_performance_metrics_chart
)

__all__ = [
    # Common utilities
    "check_system_initialization",
    "display_dataset_info",
    "get_available_datasets",
    "display_error_message",
    "display_success_message",
    "display_warning_message",
    "create_metric_columns",
    "safe_dataframe_display",
    "get_numeric_columns",
    "get_categorical_columns",
    "get_time_columns",
    "validate_dataset_selection",
    "create_analysis_summary",
    
    # Data helpers
    "prepare_data_for_analysis",
    "detect_outliers_iqr",
    "calculate_correlation_matrix",
    "find_strong_correlations",
    "calculate_data_quality_metrics",
    "create_binary_matrix",
    "find_frequent_itemsets",
    "generate_association_rules",
    
    # Visualization utilities
    "create_histogram",
    "create_scatter_plot",
    "create_bar_chart",
    "create_line_chart",
    "create_correlation_heatmap",
    "create_cluster_visualization",
    "create_trend_plot",
    "create_seasonal_decomposition",
    "create_forecast_plot",
    "create_anomaly_plot",
    "create_feature_importance_plot",
    "create_performance_metrics_chart"
]
