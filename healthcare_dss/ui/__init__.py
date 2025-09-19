"""
User Interface Components
=========================

This module contains user interface and dashboard components:
- Streamlit Application
- Dashboard Views
- KPI Dashboard
- Workflow Views
- User Interface Manager
"""

from healthcare_dss.ui.kpi_dashboard import KPIDashboard
from healthcare_dss.ui.user_interface import DashboardManager

__all__ = [
    "KPIDashboard",
    "DashboardManager"
]
