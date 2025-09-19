"""
Dashboard Configuration Module
Contains configuration for role-based dashboards and removes hardcoded values
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DashboardConfig:
    """Configuration for dashboard components"""
    title: str
    description: str
    metrics: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    filters: List[Dict[str, Any]]

@dataclass
class RoleConfig:
    """Configuration for user roles"""
    name: str
    description: str
    pages: List[str]
    permissions: List[str]
    default_page: str

class DashboardConfigManager:
    """Manages dashboard configuration"""
    
    def __init__(self):
        self.config_path = Path(__file__).parent
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables and config files"""
        # Default configurations
        self.roles = {
            "Clinical Leadership": RoleConfig(
                name="Clinical Leadership",
                description="Clinical leadership dashboard for patient care oversight",
                pages=[
                    "Clinical Dashboard",
                    "Patient Flow Management", 
                    "Quality & Safety Monitoring",
                    "Resource Allocation Guidance",
                    "Strategic Planning",
                    "Performance Management",
                    "Clinical Analytics",
                    "Outcome Analysis",
                    "Risk Assessment",
                    "Compliance Monitoring",
                    "Data Management",
                    "Model Management",
                    "Advanced Model Training"
                ],
                permissions=["clinical_read", "clinical_write", "admin_read"],
                default_page="Clinical Dashboard"
            ),
            "Administrative Executive": RoleConfig(
                name="Administrative Executive",
                description="Executive overview and strategic insights",
                pages=[
                    "Executive Dashboard",
                    "Strategic Planning",
                    "Performance Management", 
                    "Regulatory Compliance",
                    "Resource Planning",
                    "KPI Dashboard",
                    "Financial Overview",
                    "Operational Analytics",
                    "Risk Management",
                    "Stakeholder Reports"
                ],
                permissions=["executive_read", "executive_write", "admin_read", "admin_write"],
                default_page="Executive Dashboard"
            ),
            "Financial Manager": RoleConfig(
                name="Financial Manager",
                description="Financial management and cost analysis",
                pages=[
                    "Financial Dashboard",
                    "Cost Analysis",
                    "Budget Tracking",
                    "Revenue Cycle",
                    "Investment Analysis", 
                    "Financial Reporting",
                    "Cost Optimization",
                    "Revenue Forecasting",
                    "Financial Risk Analysis",
                    "Budget Planning"
                ],
                permissions=["financial_read", "financial_write", "admin_read"],
                default_page="Financial Dashboard"
            ),
            "Department Manager": RoleConfig(
                name="Department Manager",
                description="Department management and operational oversight",
                pages=[
                    "Department Dashboard",
                    "Staff Scheduling",
                    "Resource Utilization",
                    "Patient Satisfaction",
                    "Quality Metrics",
                    "Performance Analytics",
                    "Workforce Analytics",
                    "Capacity Planning",
                    "Efficiency Analysis",
                    "Team Performance"
                ],
                permissions=["department_read", "department_write", "admin_read"],
                default_page="Department Dashboard"
            ),
            "Clinical Staff": RoleConfig(
                name="Clinical Staff",
                description="Clinical staff tools and patient care support",
                pages=[
                    "Clinical Dashboard",
                    "Patient Care Tools",
                    "Clinical Guidelines",
                    "Quality Metrics",
                    "Training Resources",
                    "Performance Tracking",
                    "Clinical Decision Support",
                    "Patient Risk Assessment",
                    "Treatment Analytics",
                    "Outcome Tracking"
                ],
                permissions=["clinical_read", "staff_read"],
                default_page="Clinical Dashboard"
            ),
            "Data Analyst": RoleConfig(
                name="Data Analyst",
                description="Data analysis and modeling tools",
                pages=[
                    "Analytics Dashboard",
                    "Data Management",
                    "Model Management",
                    "Advanced Model Training",
                    "Descriptive Statistics",
                    "Exploratory Data Analysis",
                    "Trend Analysis",
                    "Time Series Analysis",
                    "Predictive Modeling",
                    "Train Model Interface",
                    "Model Comparison",
                    "Hyperparameter Tuning",
                    "Model Performance Analysis",
                    "CRISP-DM Workflow",
                    "Association Rules",
                    "Clustering Analysis",
                    "Prescriptive Analytics",
                    "Advanced Analytics",
                    "Data Visualization",
                    "Statistical Analysis",
                    "Machine Learning Pipeline"
                ],
                permissions=["data_read", "data_write", "model_read", "model_write"],
                default_page="Analytics Dashboard"
            )
        }
        
        # Load dashboard configurations
        self.dashboard_configs = self._load_dashboard_configs()
        
        # Load system configuration
        self.system_config = self._load_system_config()
    
    def _load_dashboard_configs(self) -> Dict[str, DashboardConfig]:
        """Load dashboard-specific configurations"""
        configs = {}
        
        # Clinical Dashboard Configuration
        configs["Clinical Dashboard"] = DashboardConfig(
            title="Clinical Dashboard",
            description="Clinical leadership dashboard for patient care oversight",
            metrics=[
                {
                    "name": "Active Patients",
                    "key": "active_patients",
                    "value": 0,
                    "format": "number",
                    "trend": "neutral",
                    "description": "Currently active patients in the system"
                },
                {
                    "name": "Average Wait Time",
                    "key": "avg_wait_time",
                    "value": 0,
                    "format": "duration",
                    "trend": "down",
                    "description": "Average patient wait time"
                },
                {
                    "name": "Quality Score",
                    "key": "quality_score",
                    "value": 0,
                    "format": "percentage",
                    "trend": "up",
                    "description": "Overall quality score"
                },
                {
                    "name": "Readmission Rate",
                    "key": "readmission_rate",
                    "value": 0,
                    "format": "percentage",
                    "trend": "down",
                    "description": "Patient readmission rate"
                }
            ],
            charts=[
                {
                    "type": "line",
                    "title": "Patient Flow Over Time",
                    "data_key": "patient_flow",
                    "x_axis": "time",
                    "y_axis": "patient_count"
                },
                {
                    "type": "bar",
                    "title": "Department Performance",
                    "data_key": "department_performance",
                    "x_axis": "department",
                    "y_axis": "score"
                },
                {
                    "type": "pie",
                    "title": "Patient Distribution",
                    "data_key": "patient_distribution",
                    "label": "category",
                    "value": "count"
                }
            ],
            filters=[
                {
                    "name": "Time Range",
                    "type": "date_range",
                    "key": "time_range",
                    "default": "last_30_days"
                },
                {
                    "name": "Department",
                    "type": "select",
                    "key": "department",
                    "options": ["All", "Emergency", "Surgery", "Cardiology", "Oncology"]
                }
            ]
        )
        
        # Executive Dashboard Configuration
        configs["Executive Dashboard"] = DashboardConfig(
            title="Executive Dashboard",
            description="Executive overview and strategic insights",
            metrics=[
                {
                    "name": "Revenue",
                    "key": "revenue",
                    "value": 0,
                    "format": "currency",
                    "trend": "up",
                    "description": "Monthly revenue"
                },
                {
                    "name": "Patient Satisfaction",
                    "key": "patient_satisfaction",
                    "value": 0,
                    "format": "percentage",
                    "trend": "up",
                    "description": "Patient satisfaction score"
                },
                {
                    "name": "Operational Efficiency",
                    "key": "operational_efficiency",
                    "value": 0,
                    "format": "percentage",
                    "trend": "up",
                    "description": "Operational efficiency metric"
                },
                {
                    "name": "Market Share",
                    "key": "market_share",
                    "value": 0,
                    "format": "percentage",
                    "trend": "up",
                    "description": "Market share percentage"
                }
            ],
            charts=[
                {
                    "type": "line",
                    "title": "Revenue Trend",
                    "data_key": "revenue_trend",
                    "x_axis": "month",
                    "y_axis": "revenue"
                },
                {
                    "type": "bar",
                    "title": "KPI Performance",
                    "data_key": "kpi_performance",
                    "x_axis": "kpi",
                    "y_axis": "value"
                }
            ],
            filters=[
                {
                    "name": "Time Period",
                    "type": "select",
                    "key": "time_period",
                    "options": ["Last Month", "Last Quarter", "Last Year", "YTD"]
                }
            ]
        )
        
        # Financial Dashboard Configuration
        configs["Financial Dashboard"] = DashboardConfig(
            title="Financial Dashboard",
            description="Financial management and cost analysis",
            metrics=[
                {
                    "name": "Monthly Revenue",
                    "key": "monthly_revenue",
                    "value": 0,
                    "format": "currency",
                    "trend": "up",
                    "description": "Monthly revenue"
                },
                {
                    "name": "Operating Costs",
                    "key": "operating_costs",
                    "value": 0,
                    "format": "currency",
                    "trend": "down",
                    "description": "Operating costs"
                },
                {
                    "name": "Profit Margin",
                    "key": "profit_margin",
                    "value": 0,
                    "format": "percentage",
                    "trend": "up",
                    "description": "Profit margin"
                },
                {
                    "name": "Cash Flow",
                    "key": "cash_flow",
                    "value": 0,
                    "format": "currency",
                    "trend": "up",
                    "description": "Cash flow"
                }
            ],
            charts=[
                {
                    "type": "line",
                    "title": "Financial Performance",
                    "data_key": "financial_performance",
                    "x_axis": "month",
                    "y_axis": "revenue"
                },
                {
                    "type": "bar",
                    "title": "Cost Breakdown",
                    "data_key": "cost_breakdown",
                    "x_axis": "category",
                    "y_axis": "amount"
                }
            ],
            filters=[
                {
                    "name": "Fiscal Year",
                    "type": "select",
                    "key": "fiscal_year",
                    "options": ["2024", "2023", "2022"]
                }
            ]
        )
        
        return configs
    
    def _load_system_config(self) -> Dict[str, Any]:
        """Load system-wide configuration"""
        return {
            "debug_mode": os.getenv('DSS_DEBUG_MODE', 'false').lower() == 'true',
            "database_path": os.getenv('DSS_DB_PATH', 'healthcare_dss.db'),
            "log_level": os.getenv('DSS_LOG_LEVEL', 'INFO'),
            "max_file_size": int(os.getenv('DSS_MAX_FILE_SIZE', '100')),  # MB
            "cache_ttl": int(os.getenv('DSS_CACHE_TTL', '3600')),  # seconds
            "refresh_interval": int(os.getenv('DSS_REFRESH_INTERVAL', '300')),  # seconds
            "theme": os.getenv('DSS_THEME', 'light'),
            "language": os.getenv('DSS_LANGUAGE', 'en'),
            "timezone": os.getenv('DSS_TIMEZONE', 'UTC')
        }
    
    def get_role_config(self, role_name: str) -> RoleConfig:
        """Get configuration for a specific role"""
        return self.roles.get(role_name, self.roles["Data Analyst"])
    
    def get_dashboard_config(self, dashboard_name: str) -> DashboardConfig:
        """Get configuration for a specific dashboard"""
        return self.dashboard_configs.get(dashboard_name, DashboardConfig(
            title=dashboard_name,
            description=f"{dashboard_name} dashboard",
            metrics=[],
            charts=[],
            filters=[]
        ))
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.system_config
    
    def get_all_roles(self) -> List[str]:
        """Get list of all available roles"""
        return list(self.roles.keys())
    
    def get_role_pages(self, role_name: str) -> List[str]:
        """Get pages available for a specific role"""
        role_config = self.get_role_config(role_name)
        return role_config.pages

# Global configuration instance
config_manager = DashboardConfigManager()
