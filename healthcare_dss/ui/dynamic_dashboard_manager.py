"""
Dynamic Dashboard Manager for Healthcare DSS
===========================================

This module provides a dynamic, configurable dashboard system that automatically
adapts to different user roles and system capabilities without hardcoded logic.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of dashboards available"""
    CLINICAL = "clinical"
    EXECUTIVE = "executive"
    FINANCIAL = "financial"
    DEPARTMENT = "department"
    CLINICAL_STAFF = "clinical_staff"
    DATA_ANALYST = "data_analyst"


class ComponentType(Enum):
    """Types of dashboard components"""
    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    FORM = "form"
    WORKFLOW = "workflow"
    ANALYSIS = "analysis"


@dataclass
class DashboardComponent:
    """Represents a dashboard component"""
    id: str
    name: str
    component_type: ComponentType
    description: str
    required_permissions: List[str]
    data_requirements: List[str]
    config: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class RoleConfig:
    """Configuration for a specific user role"""
    role_name: str
    display_name: str
    description: str
    permissions: List[str]
    default_dashboard: DashboardType
    available_dashboards: List[DashboardType]
    custom_components: List[str] = None
    
    def __post_init__(self):
        if self.custom_components is None:
            self.custom_components = []


@dataclass
class DashboardConfig:
    """Configuration for a specific dashboard"""
    dashboard_type: DashboardType
    name: str
    description: str
    components: List[str]
    layout: Dict[str, Any]
    refresh_interval: int = 30
    auto_refresh: bool = True


class DynamicDashboardManager:
    """
    Dynamic Dashboard Manager that provides configurable dashboards
    without hardcoded logic
    """
    
    def __init__(self, config_dir: str = "config/dashboards"):
        """
        Initialize the Dynamic Dashboard Manager
        
        Args:
            config_dir: Directory containing dashboard configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.roles: Dict[str, RoleConfig] = {}
        self.dashboards: Dict[DashboardType, DashboardConfig] = {}
        self.components: Dict[str, DashboardComponent] = {}
        
        # Load default configurations
        self._load_default_configurations()
        
        # Load custom configurations if they exist
        self._load_custom_configurations()
        
        logger.info(f"Dynamic Dashboard Manager initialized with {len(self.roles)} roles and {len(self.dashboards)} dashboards")
    
    def _load_default_configurations(self):
        """Load default dashboard configurations"""
        # Default roles
        default_roles = [
            RoleConfig(
                role_name="clinical_leadership",
                display_name="Clinical Leadership",
                description="Senior clinical staff and department heads",
                permissions=["view_patients", "view_quality_metrics", "manage_staff", "view_financials"],
                default_dashboard=DashboardType.CLINICAL,
                available_dashboards=[DashboardType.CLINICAL, DashboardType.DEPARTMENT, DashboardType.EXECUTIVE]
            ),
            RoleConfig(
                role_name="administrative_executive",
                display_name="Administrative Executive",
                description="Hospital executives and administrators",
                permissions=["view_all_metrics", "manage_budget", "view_financials", "manage_staff"],
                default_dashboard=DashboardType.EXECUTIVE,
                available_dashboards=[DashboardType.EXECUTIVE, DashboardType.FINANCIAL, DashboardType.CLINICAL]
            ),
            RoleConfig(
                role_name="financial_manager",
                display_name="Financial Manager",
                description="Financial planning and budget management",
                permissions=["view_financials", "manage_budget", "view_cost_analysis"],
                default_dashboard=DashboardType.FINANCIAL,
                available_dashboards=[DashboardType.FINANCIAL, DashboardType.EXECUTIVE]
            ),
            RoleConfig(
                role_name="department_manager",
                display_name="Department Manager",
                description="Department-level management",
                permissions=["view_department_metrics", "manage_staff", "view_patients"],
                default_dashboard=DashboardType.DEPARTMENT,
                available_dashboards=[DashboardType.DEPARTMENT, DashboardType.CLINICAL]
            ),
            RoleConfig(
                role_name="clinical_staff",
                display_name="Clinical Staff",
                description="Nurses, doctors, and clinical support staff",
                permissions=["view_patients", "view_clinical_guidelines", "document_care"],
                default_dashboard=DashboardType.CLINICAL_STAFF,
                available_dashboards=[DashboardType.CLINICAL_STAFF, DashboardType.CLINICAL]
            ),
            RoleConfig(
                role_name="data_analyst",
                display_name="Data Analyst",
                description="Data scientists and analysts",
                permissions=["view_all_data", "run_analytics", "create_reports", "manage_models"],
                default_dashboard=DashboardType.DATA_ANALYST,
                available_dashboards=[DashboardType.DATA_ANALYST, DashboardType.CLINICAL, DashboardType.EXECUTIVE]
            )
        ]
        
        for role in default_roles:
            self.roles[role.role_name] = role
        
        # Default components
        default_components = [
            DashboardComponent(
                id="patient_census",
                name="Patient Census",
                component_type=ComponentType.METRIC,
                description="Current patient count and occupancy",
                required_permissions=["view_patients"],
                data_requirements=["patient_data"],
                config={
                    "metric_type": "count",
                    "refresh_interval": 60,
                    "format": "number"
                }
            ),
            DashboardComponent(
                id="quality_metrics",
                name="Quality Metrics",
                component_type=ComponentType.CHART,
                description="Key quality indicators and safety metrics",
                required_permissions=["view_quality_metrics"],
                data_requirements=["quality_data", "safety_data"],
                config={
                    "chart_type": "line",
                    "metrics": ["infection_rate", "fall_rate", "readmission_rate"],
                    "time_range": "30_days"
                }
            ),
            DashboardComponent(
                id="financial_summary",
                name="Financial Summary",
                component_type=ComponentType.TABLE,
                description="Financial performance overview",
                required_permissions=["view_financials"],
                data_requirements=["financial_data"],
                config={
                    "columns": ["revenue", "expenses", "profit_margin"],
                    "time_period": "monthly"
                }
            ),
            DashboardComponent(
                id="staff_scheduling",
                name="Staff Scheduling",
                component_type=ComponentType.WORKFLOW,
                description="Staff scheduling and resource allocation",
                required_permissions=["manage_staff"],
                data_requirements=["staff_data", "schedule_data"],
                config={
                    "workflow_type": "scheduling",
                    "auto_optimize": True
                }
            ),
            DashboardComponent(
                id="predictive_analytics",
                name="Predictive Analytics",
                component_type=ComponentType.ANALYSIS,
                description="AI-powered predictions and insights",
                required_permissions=["run_analytics"],
                data_requirements=["patient_data", "model_data"],
                config={
                    "models": ["readmission_prediction", "length_of_stay", "risk_assessment"],
                    "confidence_threshold": 0.8
                }
            ),
            DashboardComponent(
                id="crisp_dm_workflow",
                name="CRISP-DM Workflow",
                component_type=ComponentType.WORKFLOW,
                description="Data mining workflow execution",
                required_permissions=["run_analytics", "manage_models"],
                data_requirements=["dataset_data"],
                config={
                    "workflow_type": "crisp_dm",
                    "phases": ["business_understanding", "data_understanding", "data_preparation", "modeling", "evaluation", "deployment"]
                }
            )
        ]
        
        for component in default_components:
            self.components[component.id] = component
        
        # Default dashboards
        default_dashboards = [
            DashboardConfig(
                dashboard_type=DashboardType.CLINICAL,
                name="Clinical Dashboard",
                description="Clinical operations and patient care metrics",
                components=["patient_census", "quality_metrics", "staff_scheduling"],
                layout={
                    "rows": 2,
                    "cols": 2,
                    "components": {
                        "patient_census": {"row": 0, "col": 0},
                        "quality_metrics": {"row": 0, "col": 1},
                        "staff_scheduling": {"row": 1, "col": 0, "colspan": 2}
                    }
                }
            ),
            DashboardConfig(
                dashboard_type=DashboardType.EXECUTIVE,
                name="Executive Dashboard",
                description="Strategic performance and organizational overview",
                components=["financial_summary", "quality_metrics", "predictive_analytics"],
                layout={
                    "rows": 2,
                    "cols": 2,
                    "components": {
                        "financial_summary": {"row": 0, "col": 0, "colspan": 2},
                        "quality_metrics": {"row": 1, "col": 0},
                        "predictive_analytics": {"row": 1, "col": 1}
                    }
                }
            ),
            DashboardConfig(
                dashboard_type=DashboardType.DATA_ANALYST,
                name="Data Analyst Dashboard",
                description="Advanced analytics and data science tools",
                components=["predictive_analytics", "crisp_dm_workflow", "quality_metrics"],
                layout={
                    "rows": 2,
                    "cols": 2,
                    "components": {
                        "crisp_dm_workflow": {"row": 0, "col": 0, "colspan": 2},
                        "predictive_analytics": {"row": 1, "col": 0},
                        "quality_metrics": {"row": 1, "col": 1}
                    }
                }
            )
        ]
        
        for dashboard in default_dashboards:
            self.dashboards[dashboard.dashboard_type] = dashboard
    
    def _load_custom_configurations(self):
        """Load custom configurations from files"""
        try:
            # Load custom roles
            roles_file = self.config_dir / "roles.yaml"
            if roles_file.exists():
                with open(roles_file, 'r') as f:
                    custom_roles = yaml.safe_load(f)
                    for role_data in custom_roles.get('roles', []):
                        role = RoleConfig(**role_data)
                        self.roles[role.role_name] = role
            
            # Load custom components
            components_file = self.config_dir / "components.yaml"
            if components_file.exists():
                with open(components_file, 'r') as f:
                    custom_components = yaml.safe_load(f)
                    for comp_data in custom_components.get('components', []):
                        comp_data['component_type'] = ComponentType(comp_data['component_type'])
                        component = DashboardComponent(**comp_data)
                        self.components[component.id] = component
            
            # Load custom dashboards
            dashboards_file = self.config_dir / "dashboards.yaml"
            if dashboards_file.exists():
                with open(dashboards_file, 'r') as f:
                    custom_dashboards = yaml.safe_load(f)
                    for dash_data in custom_dashboards.get('dashboards', []):
                        dash_data['dashboard_type'] = DashboardType(dash_data['dashboard_type'])
                        dashboard = DashboardConfig(**dash_data)
                        self.dashboards[dashboard.dashboard_type] = dashboard
                        
        except Exception as e:
            logger.warning(f"Failed to load custom configurations: {e}")
    
    def get_available_roles(self) -> List[str]:
        """Get list of available roles"""
        return list(self.roles.keys())
    
    def get_role_config(self, role_name: str) -> Optional[RoleConfig]:
        """Get configuration for a specific role"""
        return self.roles.get(role_name)
    
    def get_dashboard_config(self, dashboard_type: DashboardType) -> Optional[DashboardConfig]:
        """Get configuration for a specific dashboard"""
        return self.dashboards.get(dashboard_type)
    
    def get_components_for_role(self, role_name: str) -> List[DashboardComponent]:
        """Get components available for a specific role"""
        role = self.get_role_config(role_name)
        if not role:
            return []
        
        available_components = []
        for component in self.components.values():
            if all(perm in role.permissions for perm in component.required_permissions):
                available_components.append(component)
        
        return available_components
    
    def get_dashboards_for_role(self, role_name: str) -> List[DashboardConfig]:
        """Get dashboards available for a specific role"""
        role = self.get_role_config(role_name)
        if not role:
            return []
        
        available_dashboards = []
        for dashboard_type in role.available_dashboards:
            dashboard = self.get_dashboard_config(dashboard_type)
            if dashboard:
                available_dashboards.append(dashboard)
        
        return available_dashboards
    
    def create_custom_dashboard(self, role_name: str, dashboard_name: str, 
                              component_ids: List[str]) -> Optional[DashboardConfig]:
        """Create a custom dashboard for a role"""
        role = self.get_role_config(role_name)
        if not role:
            return None
        
        # Validate components are available for this role
        available_components = self.get_components_for_role(role_name)
        available_component_ids = [comp.id for comp in available_components]
        
        valid_components = [comp_id for comp_id in component_ids if comp_id in available_component_ids]
        
        if not valid_components:
            logger.warning(f"No valid components found for role {role_name}")
            return None
        
        # Create custom dashboard
        custom_dashboard = DashboardConfig(
            dashboard_type=DashboardType.CLINICAL,  # Default type
            name=dashboard_name,
            description=f"Custom dashboard for {role.display_name}",
            components=valid_components,
            layout={
                "rows": 2,
                "cols": 2,
                "components": {}
            }
        )
        
        # Auto-generate layout
        for i, comp_id in enumerate(valid_components):
            row = i // 2
            col = i % 2
            custom_dashboard.layout["components"][comp_id] = {"row": row, "col": col}
        
        return custom_dashboard
    
    def export_configuration(self, file_path: str):
        """Export current configuration to file"""
        config_data = {
            "roles": [asdict(role) for role in self.roles.values()],
            "components": [asdict(comp) for comp in self.components.values()],
            "dashboards": [asdict(dash) for dash in self.dashboards.values()]
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities summary"""
        return {
            "total_roles": len(self.roles),
            "total_dashboards": len(self.dashboards),
            "total_components": len(self.components),
            "available_roles": list(self.roles.keys()),
            "available_dashboards": [dash.dashboard_type.value for dash in self.dashboards.values()],
            "available_components": list(self.components.keys())
        }


# Global instance
dashboard_manager = DynamicDashboardManager()
