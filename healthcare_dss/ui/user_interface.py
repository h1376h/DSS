"""
User Interface Subsystem for Healthcare DSS
==========================================

This module implements the user interface capabilities including:
- Role-specific dashboard design and functionality
- Clinical decision support interfaces
- Executive and administrative planning tools
- Interactive visualizations and reporting
- Real-time monitoring and alerting
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from healthcare_dss.ui.utils.common import safe_dataframe_display
from datetime import datetime, timedelta
import logging
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from healthcare_dss.core.data_management import DataManager
from healthcare_dss.core.model_management import ModelManager
from healthcare_dss.core.knowledge_management import KnowledgeManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardManager:
    """
    User Interface Subsystem for Healthcare DSS
    
    Provides role-specific dashboards and interfaces for different
    healthcare stakeholders including clinicians, administrators, and executives.
    """
    
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, 
                 knowledge_manager: KnowledgeManager):
        """
        Initialize Dashboard Manager
        
        Args:
            data_manager: Instance of DataManager for data access
            model_manager: Instance of ModelManager for model access
            knowledge_manager: Instance of KnowledgeManager for knowledge access
        """
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.knowledge_manager = knowledge_manager
        
        # Dashboard configurations
        self.dashboard_configs = {
            'clinical': {
                'title': 'Clinical Decision Support Dashboard',
                'sections': ['patient_overview', 'clinical_alerts', 'treatment_recommendations', 'quality_metrics']
            },
            'administrative': {
                'title': 'Administrative Planning Dashboard',
                'sections': ['resource_utilization', 'financial_metrics', 'operational_efficiency', 'staffing_analysis']
            },
            'executive': {
                'title': 'Executive Strategic Dashboard',
                'sections': ['strategic_metrics', 'performance_indicators', 'trend_analysis', 'forecasting']
            }
        }
    
    def create_streamlit_app(self):
        """Create Streamlit web application"""
        st.set_page_config(
            page_title="Healthcare DSS",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar navigation
        st.sidebar.title("Healthcare DSS")
        user_role = st.sidebar.selectbox(
            "Select User Role",
            ["Clinical", "Administrative", "Executive", "Data Analyst"]
        )
        
        # Main content based on user role
        if user_role == "Clinical":
            self._render_clinical_dashboard()
        elif user_role == "Administrative":
            self._render_administrative_dashboard()
        elif user_role == "Executive":
            self._render_executive_dashboard()
        elif user_role == "Data Analyst":
            self._render_data_analyst_dashboard()
    
    def _render_clinical_dashboard(self):
        """Render clinical decision support dashboard"""
        st.title("🩺 Clinical Decision Support Dashboard")
        
        # Patient Overview Section
        st.header("Patient Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", "1,247", "+12")
        with col2:
            st.metric("High Risk Patients", "89", "+5")
        with col3:
            st.metric("Active Alerts", "23", "-3")
        with col4:
            st.metric("Avg. Response Time", "2.3 min", "-0.5 min")
        
        # Clinical Alerts Section
        st.header("Clinical Alerts")
        alerts_data = self._get_clinical_alerts()
        if alerts_data:
            for alert in alerts_data:
                with st.expander(f"🚨 {alert['title']} - {alert['severity']}"):
                    st.write(f"**Patient ID:** {alert['patient_id']}")
                    st.write(f"**Description:** {alert['description']}")
                    st.write(f"**Recommendation:** {alert['recommendation']}")
                    st.write(f"**Timestamp:** {alert['timestamp']}")
        
        # Treatment Recommendations Section
        st.header("Treatment Recommendations")
        
        # Patient input form
        with st.form("patient_assessment"):
            st.subheader("Patient Assessment")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=45)
                bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
                systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
                diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
            
            with col2:
                family_history = st.checkbox("Family History of Diabetes")
                smoking = st.checkbox("Current Smoker")
                exercise = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"])
                hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
            
            submitted = st.form_submit_button("Get Recommendations")
            
            if submitted:
                patient_data = {
                    'age': age,
                    'bmi': bmi,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'family_history': family_history,
                    'smoking': smoking,
                    'exercise': exercise,
                    'hba1c': hba1c
                }
                
                # Get clinical recommendations
                recommendations = self.knowledge_manager.get_clinical_recommendations(patient_data)
                
                st.subheader("Clinical Recommendations")
                for i, rec in enumerate(recommendations[:5], 1):
                    st.write(f"{i}. {rec['recommendation']}")
                    st.caption(f"Source: {rec['source']} (Evidence Level: {rec['evidence_level']})")
        
        # Quality Metrics Section
        st.header("Quality Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Patient satisfaction trend
            satisfaction_data = self._get_patient_satisfaction_data()
            fig = px.line(satisfaction_data, x='month', y='satisfaction_score', 
                         title='Patient Satisfaction Trend')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Readmission rates
            readmission_data = self._get_readmission_data()
            fig = px.bar(readmission_data, x='department', y='readmission_rate',
                        title='Readmission Rates by Department')
            st.plotly_chart(fig, width="stretch")
    
    def _render_administrative_dashboard(self):
        """Render administrative planning dashboard"""
        st.title("Administrative Planning Dashboard")
        
        # Resource Utilization Section
        st.header("Resource Utilization")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bed Occupancy", "87%", "+3%")
        with col2:
            st.metric("Staff Utilization", "92%", "+2%")
        with col3:
            st.metric("Equipment Usage", "78%", "-5%")
        
        # Financial Metrics Section
        st.header("Financial Metrics")
        
        # Healthcare expenditure analysis
        expenditure_analysis = self.data_manager.get_healthcare_expenditure_analysis()
        
        col1, col2 = st.columns(2)
        with col1:
            # Expenditure by country
            countries = list(expenditure_analysis['expenditure_trends'].keys())[:10]
            expenditures = [expenditure_analysis['expenditure_trends'][c]['avg_expenditure'] for c in countries]
            
            fig = px.bar(x=countries, y=expenditures, 
                        title='Average Healthcare Expenditure by Country')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Budget variance
            budget_data = self._get_budget_variance_data()
            fig = px.line(budget_data, x='month', y='variance_percentage',
                         title='Budget Variance Trend')
            st.plotly_chart(fig, width="stretch")
        
        # Operational Efficiency Section
        st.header("Operational Efficiency")
        
        # Model performance comparison
        if hasattr(self.model_manager, 'get_model_performance_summary'):
            model_summary = self.model_manager.get_model_performance_summary()
            if not model_summary.empty:
                st.subheader("Model Performance Summary")
                safe_dataframe_display(model_summary, width="stretch")
        
        # Staffing Analysis Section
        st.header("Staffing Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Staff distribution
            staff_data = self._get_staff_distribution_data()
            fig = px.pie(staff_data, values='count', names='department',
                        title='Staff Distribution by Department')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Overtime trends
            overtime_data = self._get_overtime_data()
            fig = px.line(overtime_data, x='week', y='overtime_hours',
                         title='Overtime Hours Trend')
            st.plotly_chart(fig, width="stretch")
    
    def _render_executive_dashboard(self):
        """Render executive strategic dashboard"""
        st.title("Executive Strategic Dashboard")
        
        # Strategic Metrics Section
        st.header("Strategic Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Patient Volume", "15,247", "+8.2%")
        with col2:
            st.metric("Revenue Growth", "12.5%", "+2.1%")
        with col3:
            st.metric("Quality Score", "94.2", "+1.8")
        with col4:
            st.metric("Market Share", "23.4%", "+0.7%")
        
        # Performance Indicators Section
        st.header("Key Performance Indicators")
        
        # KPI dashboard
        kpi_data = self._get_kpi_data()
        
        col1, col2 = st.columns(2)
        with col1:
            # KPI trend
            fig = px.line(kpi_data, x='month', y='value', color='kpi',
                         title='KPI Trends Over Time')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # KPI comparison
            current_kpis = kpi_data[kpi_data['month'] == kpi_data['month'].max()]
            fig = px.bar(current_kpis, x='kpi', y='value',
                        title='Current KPI Values')
            st.plotly_chart(fig, width="stretch")
        
        # Trend Analysis Section
        st.header("Trend Analysis")
        
        # Capacity planning
        capacity_data = self._get_capacity_planning_data()
        fig = px.area(capacity_data, x='month', y='capacity_utilization',
                     title='Capacity Utilization Forecast')
        st.plotly_chart(fig, width="stretch")
        
        # Forecasting Section
        st.header("Forecasting")
        
        col1, col2 = st.columns(2)
        with col1:
            # Patient volume forecast
            forecast_data = self._get_patient_volume_forecast()
            fig = px.line(forecast_data, x='month', y='patient_volume',
                         title='Patient Volume Forecast')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Revenue forecast
            revenue_forecast = self._get_revenue_forecast()
            fig = px.line(revenue_forecast, x='month', y='revenue',
                         title='Revenue Forecast')
            st.plotly_chart(fig, width="stretch")
    
    def _render_data_analyst_dashboard(self):
        """Render data analyst dashboard"""
        st.title("Data Analyst Dashboard")
        
        # Dataset Overview Section
        st.header("Dataset Overview")
        
        # Show available datasets
        for dataset_name in self.data_manager.datasets.keys():
            with st.expander(f"{dataset_name.upper()} Dataset"):
                dataset_info = self.data_manager.get_dataset_info(dataset_name)
                st.write(f"**Shape:** {dataset_info['shape']}")
                st.write(f"**Columns:** {len(dataset_info['columns'])}")
                st.write(f"**Memory Usage:** {dataset_info['memory_usage']:,} bytes")
                
                # Show sample data
                st.subheader("Sample Data")
                sample_df = pd.DataFrame(dataset_info['sample_data'])
                safe_dataframe_display(sample_df, width="stretch")
        
        # Data Quality Section
        st.header("Data Quality Assessment")
        
        # Quality metrics for each dataset
        for dataset_name in self.data_manager.datasets.keys():
            if dataset_name in self.data_manager.data_quality_metrics:
                quality_metrics = self.data_manager.data_quality_metrics[dataset_name]
                
                with st.expander(f"{dataset_name.upper()} Quality Metrics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Completeness", f"{quality_metrics['completeness_score']:.1f}%")
                    with col2:
                        st.metric("Duplicate Rows", f"{quality_metrics['duplicate_percentage']:.1f}%")
                    with col3:
                        total_outliers = sum([outlier['count'] for outlier in quality_metrics['outliers'].values()])
                        st.metric("Total Outliers", total_outliers)
        
        # Model Performance Section
        st.header("Model Performance")
        
        if hasattr(self.model_manager, 'get_model_performance_summary'):
            model_summary = self.model_manager.get_model_performance_summary()
            if not model_summary.empty:
                st.subheader("Model Performance Comparison")
                safe_dataframe_display(model_summary, width="stretch")
                
                # Model performance visualization
                if 'accuracy' in model_summary.columns:
                    fig = px.bar(model_summary, x='Model', y='accuracy',
                                title='Model Accuracy Comparison')
                    st.plotly_chart(fig, width="stretch")
        
        # Knowledge Base Section
        st.header("Knowledge Base")
        
        knowledge_summary = self.knowledge_manager.get_knowledge_summary()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Clinical Rules", knowledge_summary['clinical_rules'])
        with col2:
            st.metric("Guidelines", knowledge_summary['clinical_guidelines'])
        with col3:
            st.metric("Decision Trees", knowledge_summary['decision_trees'])
        with col4:
            st.metric("Active Rules", knowledge_summary['active_rules'])
    
    def create_dash_app(self):
        """Create Dash web application"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Healthcare Decision Support System", className="text-center mb-4"),
                    dbc.Tabs([
                        dbc.Tab(label="Clinical Dashboard", tab_id="clinical"),
                        dbc.Tab(label="Administrative Dashboard", tab_id="administrative"),
                        dbc.Tab(label="Executive Dashboard", tab_id="executive"),
                        dbc.Tab(label="Data Analysis", tab_id="data-analysis")
                    ], id="tabs", active_tab="clinical")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="tab-content")
                ])
            ])
        ], fluid=True)
        
        @app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            if active_tab == "clinical":
                return self._create_clinical_dash_content()
            elif active_tab == "administrative":
                return self._create_administrative_dash_content()
            elif active_tab == "executive":
                return self._create_executive_dash_content()
            elif active_tab == "data-analysis":
                return self._create_data_analysis_dash_content()
            return html.Div("Select a tab")
        
        return app
    
    def _create_clinical_dash_content(self):
        """Create clinical dashboard content for Dash"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Patient Overview"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("1,247", className="card-title"),
                                    html.P("Total Patients", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("89", className="card-title"),
                                    html.P("High Risk Patients", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("23", className="card-title"),
                                    html.P("Active Alerts", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("2.3 min", className="card-title"),
                                    html.P("Avg. Response Time", className="card-text")
                                ])
                            ])
                        ], width=3)
                    ])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Clinical Alerts"),
                    dcc.Graph(
                        figure=px.bar(
                            x=['Alert 1', 'Alert 2', 'Alert 3'],
                            y=[5, 8, 3],
                            title="Active Alerts by Type"
                        )
                    )
                ], width=6),
                dbc.Col([
                    html.H3("Quality Metrics"),
                    dcc.Graph(
                        figure=px.line(
                            x=['Jan', 'Feb', 'Mar', 'Apr'],
                            y=[85, 87, 89, 92],
                            title="Patient Satisfaction Trend"
                        )
                    )
                ], width=6)
            ])
        ])
    
    def _create_administrative_dash_content(self):
        """Create administrative dashboard content for Dash"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Resource Utilization"),
                    dcc.Graph(
                        figure=px.pie(
                            values=[87, 13],
                            names=['Occupied', 'Available'],
                            title="Bed Occupancy Rate"
                        )
                    )
                ], width=6),
                dbc.Col([
                    html.H3("Financial Metrics"),
                    dcc.Graph(
                        figure=px.bar(
                            x=['Q1', 'Q2', 'Q3', 'Q4'],
                            y=[1200000, 1350000, 1420000, 1580000],
                            title="Quarterly Revenue"
                        )
                    )
                ], width=6)
            ])
        ])
    
    def _create_executive_dash_content(self):
        """Create executive dashboard content for Dash"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Strategic Metrics"),
                    dcc.Graph(
                        figure=px.line(
                            x=['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                            y=[15.2, 15.8, 16.1, 16.5, 17.0],
                            title="Patient Volume Trend (Thousands)"
                        )
                    )
                ], width=6),
                dbc.Col([
                    html.H3("Performance Indicators"),
                    dcc.Graph(
                        figure=px.bar(
                            x=['Quality', 'Efficiency', 'Patient Satisfaction', 'Financial'],
                            y=[94.2, 87.5, 91.8, 88.3],
                            title="Current KPI Values"
                        )
                    )
                ], width=6)
            ])
        ])
    
    def _create_data_analysis_dash_content(self):
        """Create data analysis dashboard content for Dash"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Dataset Overview"),
                    html.P("Available datasets and their characteristics")
                ]),
                dbc.Col([
                    html.H3("Model Performance"),
                    html.P("Machine learning model performance metrics")
                ])
            ])
        ])
    
    # Helper methods for mock data generation
    def _get_clinical_alerts(self):
        """Get mock clinical alerts data"""
        return [
            {
                'title': 'High Blood Pressure',
                'severity': 'High',
                'patient_id': 'P001',
                'description': 'Patient has sustained high blood pressure readings',
                'recommendation': 'Consider antihypertensive medication',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            },
            {
                'title': 'Diabetes Risk',
                'severity': 'Moderate',
                'patient_id': 'P002',
                'description': 'Patient shows signs of pre-diabetes',
                'recommendation': 'Lifestyle modification and monitoring',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
        ]
    
    def _get_patient_satisfaction_data(self):
        """Get mock patient satisfaction data"""
        return pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'satisfaction_score': [85, 87, 89, 92, 94]
        })
    
    def _get_readmission_data(self):
        """Get mock readmission data"""
        return pd.DataFrame({
            'department': ['Cardiology', 'Oncology', 'Orthopedics', 'Emergency'],
            'readmission_rate': [12.5, 8.3, 15.2, 18.7]
        })
    
    def _get_budget_variance_data(self):
        """Get mock budget variance data"""
        return pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'variance_percentage': [2.1, -1.5, 3.2, -0.8, 1.9]
        })
    
    def _get_staff_distribution_data(self):
        """Get mock staff distribution data"""
        return pd.DataFrame({
            'department': ['Nursing', 'Physicians', 'Support Staff', 'Administration'],
            'count': [45, 23, 67, 12]
        })
    
    def _get_overtime_data(self):
        """Get mock overtime data"""
        return pd.DataFrame({
            'week': [f'Week {i}' for i in range(1, 13)],
            'overtime_hours': np.random.randint(100, 300, 12)
        })
    
    def _get_kpi_data(self):
        """Get mock KPI data"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        kpis = ['Quality Score', 'Efficiency', 'Patient Satisfaction', 'Financial Performance']
        
        data = []
        for month in months:
            for kpi in kpis:
                data.append({
                    'month': month,
                    'kpi': kpi,
                    'value': np.random.uniform(80, 95)
                })
        
        return pd.DataFrame(data)
    
    def _get_capacity_planning_data(self):
        """Get mock capacity planning data"""
        return pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'capacity_utilization': [78, 82, 85, 88, 91, 94]
        })
    
    def _get_patient_volume_forecast(self):
        """Get mock patient volume forecast data"""
        return pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'patient_volume': [15247, 15800, 16350, 16900, 17450, 18000]
        })
    
    def _get_revenue_forecast(self):
        """Get mock revenue forecast data"""
        return pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'revenue': [1200000, 1250000, 1300000, 1350000, 1400000, 1450000]
        })


# Example usage and testing
if __name__ == "__main__":
    from healthcare_dss.ui.data_management import DataManager
    from healthcare_dss.ui.model_management import ModelManager
    from healthcare_dss.ui.knowledge_management import KnowledgeManager
    
    # Initialize managers
    data_manager = DataManager()
    model_manager = ModelManager(data_manager)
    knowledge_manager = KnowledgeManager(data_manager, model_manager)
    
    # Create dashboard manager
    dashboard_manager = DashboardManager(data_manager, model_manager, knowledge_manager)
    
    # Run Streamlit app
    dashboard_manager.create_streamlit_app()
