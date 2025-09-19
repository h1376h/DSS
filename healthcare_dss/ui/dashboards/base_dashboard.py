"""
Base Dashboard Module
Provides common functionality for all role-based dashboards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from healthcare_dss.config.dashboard_config import config_manager
from healthcare_dss.utils.debug_manager import debug_manager

# DatasetManager functionality is now integrated into DataManager
# Use DataManager from session state instead

logger = logging.getLogger(__name__)

class BaseDashboard:
    """Base class for all dashboard implementations"""
    
    def __init__(self, dashboard_name: str):
        self.dashboard_name = dashboard_name
        self.config = config_manager.get_dashboard_config(dashboard_name)
        self.data_manager = st.session_state.get('data_manager')
        self.debug_mode = st.session_state.get('debug_mode', False)
        
        # DatasetManager functionality is now integrated into DataManager
        self.dataset_manager = self.data_manager
        
    def render(self):
        """Main render method for the dashboard"""
        self._render_start_time = time.time()
        
        try:
            self._render_header()
            self._render_filters()
            self._render_metrics()
            self._render_charts()
            self._render_additional_content()
            
            if self.debug_mode:
                self._render_debug_info()
                
        except Exception as e:
            logger.error(f"Error rendering {self.dashboard_name}: {str(e)}")
            st.error(f"Error loading {self.dashboard_name}: {str(e)}")
            if self.debug_mode:
                debug_manager.render_error_debug(e, f"Dashboard {self.dashboard_name}")
    
    def _render_header(self):
        """Render dashboard header"""
        st.header(self.config.title)
        st.markdown(f"**{self.config.description}**")
        
        # Add refresh button in a more professional way
        col1, col2, col3 = st.columns([2, 1, 1])
        with col3:
            if st.button("Refresh Data", key=f"refresh_{self.dashboard_name}", help="Reload dashboard data"):
                st.rerun()
    
    def _render_filters(self):
        """Render dashboard filters"""
        if not self.config.filters:
            return
            
        st.subheader("Filters")
        filter_values = {}
        
        for filter_config in self.config.filters:
            filter_values[filter_config['key']] = self._render_filter(filter_config)
        
        # Store filter values in session state
        st.session_state[f"{self.dashboard_name}_filters"] = filter_values
    
    def _render_filter(self, filter_config: Dict[str, Any]) -> Any:
        """Render individual filter"""
        filter_type = filter_config['type']
        filter_key = filter_config['key']
        filter_name = filter_config['name']
        
        if filter_type == 'date_range':
            default_start = datetime.now() - timedelta(days=30)
            default_end = datetime.now()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    f"{filter_name} Start",
                    value=default_start,
                    key=f"{self.dashboard_name}_{filter_key}_start"
                )
            with col2:
                end_date = st.date_input(
                    f"{filter_name} End",
                    value=default_end,
                    key=f"{self.dashboard_name}_{filter_key}_end"
                )
            return (start_date, end_date)
            
        elif filter_type == 'select':
            options = filter_config.get('options', [])
            default_index = 0
            
            return st.selectbox(
                filter_name,
                options=options,
                index=default_index,
                key=f"{self.dashboard_name}_{filter_key}"
            )
            
        elif filter_type == 'multiselect':
            options = filter_config.get('options', [])
            
            return st.multiselect(
                filter_name,
                options=options,
                default=options,
                key=f"{self.dashboard_name}_{filter_key}"
            )
            
        elif filter_type == 'slider':
            min_val = filter_config.get('min', 0)
            max_val = filter_config.get('max', 100)
            default_val = filter_config.get('default', min_val)
            
            return st.slider(
                filter_name,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"{self.dashboard_name}_{filter_key}"
            )
    
    def _render_metrics(self):
        """Render dashboard metrics"""
        if not self.config.metrics:
            return
            
        st.subheader("Key Metrics")
        
        # Calculate metrics
        metrics_data = self._calculate_metrics()
        
        # Display metrics in columns
        cols = st.columns(len(self.config.metrics))
        for i, metric_config in enumerate(self.config.metrics):
            with cols[i]:
                self._render_metric(metric_config, metrics_data)
    
    def _render_metric(self, metric_config: Dict[str, Any], metrics_data: Dict[str, Any]):
        """Render individual metric"""
        metric_key = metric_config['key']
        metric_name = metric_config['name']
        metric_format = metric_config.get('format', 'number')
        trend = metric_config.get('trend', 'neutral')
        
        # Get metric value
        value = metrics_data.get(metric_key, metric_config.get('value', 0))
        
        # Format value based on type
        if metric_format == 'currency':
            formatted_value = f"${value:,.0f}" if isinstance(value, (int, float)) else str(value)
        elif metric_format == 'percentage':
            formatted_value = f"{value:.1f}%" if isinstance(value, (int, float)) else str(value)
        elif metric_format == 'duration':
            formatted_value = f"{value} min" if isinstance(value, (int, float)) else str(value)
        else:
            formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
        
        # Determine trend indicator
        if trend == 'up':
            delta = "+5.2%"
            delta_color = "normal"
        elif trend == 'down':
            delta = "-2.1%"
            delta_color = "inverse"
        else:
            delta = None
            delta_color = "normal"
        
        # Render metric
        st.metric(
            label=metric_name,
            value=formatted_value,
            delta=delta,
            delta_color=delta_color,
            help=metric_config.get('description', '')
        )
    
    def _render_charts(self):
        """Render dashboard charts"""
        if not self.config.charts:
            return
            
        st.subheader("Charts & Visualizations")
        
        # Get chart data
        charts_data = self._get_charts_data()
        
        # Render charts
        for chart_config in self.config.charts:
            self._render_chart(chart_config, charts_data)
    
    def _render_chart(self, chart_config: Dict[str, Any], charts_data: Dict[str, Any]):
        """Render individual chart"""
        chart_type = chart_config['type']
        chart_title = chart_config['title']
        data_key = chart_config['data_key']
        
        # Get chart data
        chart_data = charts_data.get(data_key, pd.DataFrame())
        
        if chart_data.empty:
            st.warning(f"No data available for {chart_title}")
            return
        
        try:
            if chart_type == 'line':
                fig = px.line(
                    chart_data,
                    x=chart_config['x_axis'],
                    y=chart_config['y_axis'],
                    title=chart_title
                )
            elif chart_type == 'bar':
                fig = px.bar(
                    chart_data,
                    x=chart_config['x_axis'],
                    y=chart_config['y_axis'],
                    title=chart_title
                )
            elif chart_type == 'pie':
                fig = px.pie(
                    chart_data,
                    names=chart_config['label'],
                    values=chart_config['value'],
                    title=chart_title
                )
            elif chart_type == 'scatter':
                fig = px.scatter(
                    chart_data,
                    x=chart_config['x_axis'],
                    y=chart_config['y_axis'],
                    title=chart_title
                )
            elif chart_type == 'heatmap':
                fig = px.imshow(
                    chart_data.values,
                    title=chart_title,
                    aspect="auto"
                )
            else:
                st.warning(f"Unsupported chart type: {chart_type}")
                return
            
            # Update layout
            fig.update_layout(
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, width="stretch")
            
        except Exception as e:
            logger.error(f"Error rendering chart {chart_title}: {str(e)}")
            st.error(f"Error rendering chart: {str(e)}")
            if self.debug_mode:
                st.exception(e)
    
    def _render_additional_content(self):
        """Render additional dashboard-specific content"""
        # Override in subclasses
        pass
    
    def _render_debug_info(self):
        """Render comprehensive debug information using enhanced debug manager"""
        if not self.debug_mode:
            return
            
        # Use enhanced debug manager for page-specific debug info
        additional_data = {
            "Dashboard Name": self.dashboard_name,
            "Configuration Title": self.config.title,
            "Metrics Count": len(self.config.metrics),
            "Charts Count": len(self.config.charts),
            "Filters Count": len(self.config.filters),
            "Data Manager Available": self.data_manager is not None
        }
        
        # Get debug data and render it directly
        debug_data = debug_manager.get_page_debug_data(self.dashboard_name, additional_data)
        
        with st.expander("🔍 Dashboard Debug Information", expanded=False):
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            st.markdown("---")
            st.subheader("Dashboard Configuration")
            config_data = {
                "title": self.config.title,
                "description": self.config.description,
                "metrics": [metric.get('name', 'Unknown') for metric in self.config.metrics],
                "charts": [chart.get('name', 'Unknown') for chart in self.config.charts],
                "filters": [filter_config.get('name', 'Unknown') for filter_config in self.config.filters]
            }
            st.json(config_data)
        
        # Data Manager Status
        with st.expander("Data Manager Status", expanded=False):
            if self.data_manager:
                st.success("✅ Data Manager Available")
                try:
                    datasets = list(self.data_manager.datasets.keys())
                    st.write(f"**Available datasets:** {len(datasets)}")
                    for dataset in datasets:
                        df = self.data_manager.datasets[dataset]
                        st.write(f"- {dataset}: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"❌ Error accessing datasets: {str(e)}")
                    debug_manager.log_debug(f"Error accessing datasets: {str(e)}", "ERROR")
            else:
                st.error("❌ Data Manager Not Available")
        
        # Performance Metrics
        with st.expander("Performance Metrics", expanded=False):
            if hasattr(self, '_render_start_time'):
                render_time = time.time() - self._render_start_time
                st.metric("Render Time", f"{render_time:.3f}s")
                debug_manager.update_performance_metric(f"{self.dashboard_name} Render Time", render_time, "s")
            
            # Data processing time
            if hasattr(self, '_data_processing_time'):
                st.metric("Data Processing Time", f"{self._data_processing_time:.3f}s")
                debug_manager.update_performance_metric(f"{self.dashboard_name} Data Processing", self._data_processing_time, "s")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the dashboard"""
        # Override in subclasses
        return {}
    
    def _get_charts_data(self) -> Dict[str, pd.DataFrame]:
        """Get data for charts"""
        # Override in subclasses
        return {}
    
    def _get_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample data for demonstration using real datasets"""
        # Try to get real data first, fallback to generated data
        if self.dataset_manager:
            return self._get_real_data()
        else:
            # Generate sample data based on dashboard type
            if "Clinical" in self.dashboard_name:
                return self._get_clinical_sample_data()
            elif "Executive" in self.dashboard_name:
                return self._get_executive_sample_data()
            elif "Financial" in self.dashboard_name:
                return self._get_financial_sample_data()
            else:
                return self._get_generic_sample_data()
    
    def _get_real_data(self) -> Dict[str, pd.DataFrame]:
        """Get real data from datasets"""
        try:
            real_data = {}
            
            # Get patient demographics data
            if 'patient_demographics' in self.dataset_manager.datasets:
                df = self.dataset_manager.datasets['patient_demographics']
                
                # Patient flow data (daily admissions)
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
                daily_admissions = df.groupby(df['admission_date'].dt.date).size()
                
                real_data['patient_flow'] = pd.DataFrame({
                    'time': dates,
                    'patient_count': [daily_admissions.get(date.date(), 0) for date in dates]
                })
                
                # Department performance
                dept_perf = df.groupby('department').agg({
                    'patient_id': 'count',
                    'length_of_stay': 'mean',
                    'readmission_risk': 'mean'
                }).reset_index()
                dept_perf.columns = ['department', 'patient_count', 'avg_length_of_stay', 'readmission_rate']
                
                real_data['department_performance'] = dept_perf
                
                # Patient distribution by category
                patient_dist = df['department'].value_counts().reset_index()
                patient_dist.columns = ['category', 'count']
                
                real_data['patient_distribution'] = patient_dist
            
            # Get financial data
            if 'financial_metrics' in self.dataset_manager.datasets:
                df = self.dataset_manager.datasets['financial_metrics']
                real_data['revenue_trend'] = df[['month', 'revenue']].copy()
                real_data['expense_trend'] = df[['month', 'expenses']].copy()
            
            # Get staff performance data
            if 'staff_performance' in self.dataset_manager.datasets:
                df = self.dataset_manager.datasets['staff_performance']
                staff_perf = df.groupby('department').agg({
                    'patient_satisfaction_score': 'mean',
                    'task_completion_rate': 'mean',
                    'response_time_minutes': 'mean'
                }).reset_index()
                
                real_data['staff_performance'] = staff_perf
            
            return real_data
            
        except Exception as e:
            logger.error(f"Error getting real data: {str(e)}")
            return self._get_clinical_sample_data()
    
    def _get_clinical_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample clinical data"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        return {
            "patient_flow": pd.DataFrame({
                'time': dates,
                'patient_count': np.random.poisson(150, len(dates))
            }),
            "department_performance": pd.DataFrame({
                'department': ['Emergency', 'Surgery', 'Cardiology', 'Oncology', 'Pediatrics'],
                'score': [85, 92, 88, 90, 87]
            }),
            "patient_distribution": pd.DataFrame({
                'category': ['Inpatient', 'Outpatient', 'Emergency', 'Surgery'],
                'count': [45, 35, 15, 5]
            })
        }
    
    def _get_executive_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample executive data"""
        months = pd.date_range(start='2024-01-01', end='2024-12-31', freq='ME')
        
        return {
            "revenue_trend": pd.DataFrame({
                'month': months,
                'revenue': np.random.normal(2400000, 200000, len(months))
            }),
            "kpi_performance": pd.DataFrame({
                'kpi': ['Patient Satisfaction', 'Operational Efficiency', 'Quality Score', 'Market Share'],
                'value': [92, 87, 94, 23]
            })
        }
    
    def _get_financial_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Get sample financial data"""
        months = pd.date_range(start='2024-01-01', end='2024-12-31', freq='ME')
        
        return {
            "financial_performance": pd.DataFrame({
                'month': months,
                'revenue': np.random.normal(450000, 50000, len(months)),
                'costs': np.random.normal(320000, 30000, len(months))
            }),
            "cost_breakdown": pd.DataFrame({
                'category': ['Personnel', 'Supplies', 'Equipment', 'Facilities', 'Other'],
                'amount': [180000, 80000, 40000, 15000, 5000]
            })
        }
    
    def _get_generic_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Get generic sample data"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        return {
            "generic_trend": pd.DataFrame({
                'date': dates,
                'value': np.random.normal(100, 10, len(dates))
            })
        }
