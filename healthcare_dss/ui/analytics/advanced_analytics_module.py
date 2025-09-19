"""
Advanced Analytics Module
========================

Provides comprehensive advanced analytics capabilities including:
- Time Series Analysis
- Prescriptive Analytics
- Optimization Models
- Simulation Capabilities
- Ensemble Modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from healthcare_dss.utils.debug_manager import debug_manager
from healthcare_dss.ui.utils.common import safe_dataframe_display

logger = logging.getLogger(__name__)

class AdvancedAnalyticsModule:
    """Advanced Analytics Module for Healthcare DSS"""
    
    def __init__(self):
        self.data_manager = st.session_state.get('data_manager')
        
    def render_time_series_analysis(self):
        """Render Time Series Analysis Interface"""
        st.header("Time Series Analysis")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Time Series Analysis Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("Time Series Analysis", {
                    "Has Data Manager": self.data_manager is not None,
                    "Available Datasets": len(self.data_manager.datasets) if self.data_manager else 0,
                    "Dataset Names": list(self.data_manager.datasets.keys()) if self.data_manager else []
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
                
                if self.data_manager:
                    st.markdown("---")
                    st.subheader("Dataset Details")
                    for dataset_name, df in self.data_manager.datasets.items():
                        st.write(f"**{dataset_name}**: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Columns: {list(df.columns)}")
        
        if not self.data_manager:
            st.warning("No data manager available")
            return
        
        # Dataset selection
        datasets = list(self.data_manager.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        if selected_dataset:
            df = self.data_manager.datasets[selected_dataset]
            
            # Generate time series data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
            ts_data = pd.DataFrame({
                'date': dates,
                'value': np.random.normal(100, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5
            })
            
            # Time series visualization
            fig = px.line(ts_data, x='date', y='value', title='Time Series Data')
            st.plotly_chart(fig, width="stretch")
            
            # Trend analysis
            st.subheader("Trend Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Moving average
                window = st.slider("Moving Average Window", 7, 30, 14)
                ts_data['ma'] = ts_data['value'].rolling(window=window).mean()
                
                fig = px.line(ts_data, x='date', y=['value', 'ma'], title='Trend Analysis')
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                # Seasonal decomposition
                st.write("**Seasonal Patterns:**")
                seasonal_pattern = np.sin(np.arange(365) * 2 * np.pi / 365) * 5
                fig = px.line(x=range(365), y=seasonal_pattern, title='Seasonal Pattern')
                st.plotly_chart(fig, width="stretch")
            
            # Forecasting
            st.subheader("Forecasting")
            forecast_days = st.slider("Forecast Days", 7, 90, 30)
            
            if st.button("Generate Forecast"):
                # Simple linear trend forecast
                X = np.arange(len(ts_data)).reshape(-1, 1)
                y = ts_data['value'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate forecast
                future_X = np.arange(len(ts_data), len(ts_data) + forecast_days).reshape(-1, 1)
                forecast = model.predict(future_X)
                
                # Create forecast dataframe
                future_dates = pd.date_range(start=ts_data['date'].iloc[-1] + timedelta(days=1), periods=forecast_days)
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecast': forecast,
                    'confidence_lower': forecast - 1.96 * np.std(y),
                    'confidence_upper': forecast + 1.96 * np.std(y)
                })
                
                # Combine historical and forecast data
                combined_df = pd.concat([
                    ts_data[['date', 'value']].assign(type='Historical'),
                    forecast_df[['date', 'forecast']].rename(columns={'forecast': 'value'}).assign(type='Forecast')
                ])
                
                fig = px.line(combined_df, x='date', y='value', color='type', title='Time Series Forecast')
                st.plotly_chart(fig, width="stretch")
                
                # Forecast metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Forecast Period", f"{forecast_days} days")
                with col2:
                    st.metric("Trend", "Upward" if model.coef_[0] > 0 else "Downward")
                with col3:
                    st.metric("Confidence", "95%")
    
    def render_prescriptive_analytics(self):
        """Render Prescriptive Analytics Interface"""
        st.header("Prescriptive Analytics")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Prescriptive Analytics Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("Prescriptive Analytics", {
                    "Has Data Manager": self.data_manager is not None,
                    "Available Datasets": len(self.data_manager.datasets) if self.data_manager else 0
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
        
        st.subheader("Resource Optimization")
        
        # Resource allocation problem
        st.write("**Healthcare Resource Allocation Problem**")
        
        # Define resources and constraints
        resources = {
            'Nurses': {'available': 50, 'cost_per_hour': 35},
            'Doctors': {'available': 15, 'cost_per_hour': 120},
            'Equipment': {'available': 25, 'cost_per_hour': 50},
            'Beds': {'available': 100, 'cost_per_hour': 10}
        }
        
        # Display resources
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Resources:**")
            for resource, info in resources.items():
                st.write(f"- {resource}: {info['available']} units")
        
        with col2:
            st.write("**Cost Structure:**")
            for resource, info in resources.items():
                st.write(f"- {resource}: ${info['cost_per_hour']}/hour")
        
        # Optimization scenario
        st.subheader("Optimization Scenario")
        
        scenario = st.selectbox(
            "Select Scenario",
            ["Normal Operations", "Peak Demand", "Emergency Response", "Cost Minimization"]
        )
        
        if scenario == "Normal Operations":
            demand_multiplier = 1.0
            priority = "Balanced"
        elif scenario == "Peak Demand":
            demand_multiplier = 1.5
            priority = "Capacity"
        elif scenario == "Emergency Response":
            demand_multiplier = 2.0
            priority = "Speed"
        else:  # Cost Minimization
            demand_multiplier = 0.8
            priority = "Cost"
        
        # Calculate optimal allocation
        optimal_allocation = {}
        total_cost = 0
        
        for resource, info in resources.items():
            if priority == "Capacity":
                allocation = min(info['available'], int(info['available'] * demand_multiplier))
            elif priority == "Speed":
                allocation = info['available']
            elif priority == "Cost":
                allocation = max(1, int(info['available'] * demand_multiplier))
            else:  # Balanced
                allocation = int(info['available'] * demand_multiplier)
            
            optimal_allocation[resource] = allocation
            total_cost += allocation * info['cost_per_hour']
        
        # Display results
        st.subheader("Optimal Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            allocation_df = pd.DataFrame([
                {'Resource': resource, 'Allocated': allocation, 'Available': info['available']}
                for resource, info in resources.items()
                for allocation in [optimal_allocation[resource]]
            ])
            
            fig = px.bar(
                allocation_df,
                x='Resource',
                y=['Allocated', 'Available'],
                title='Resource Allocation',
                barmode='group'
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.write("**Allocation Summary:**")
            for resource, allocation in optimal_allocation.items():
                utilization = (allocation / resources[resource]['available']) * 100
                st.write(f"- {resource}: {allocation} units ({utilization:.1f}% utilization)")
            
            st.metric("Total Hourly Cost", f"${total_cost:,}")
        
        # Recommendations
        st.subheader("Recommendations")
        
        recommendations = []
        for resource, allocation in optimal_allocation.items():
            utilization = (allocation / resources[resource]['available']) * 100
            
            if utilization > 90:
                recommendations.append(f"🚨 {resource}: Consider increasing capacity (utilization: {utilization:.1f}%)")
            elif utilization < 50:
                recommendations.append(f"💡 {resource}: Consider reducing allocation (utilization: {utilization:.1f}%)")
            else:
                recommendations.append(f"✅ {resource}: Optimal allocation (utilization: {utilization:.1f}%)")
        
        for rec in recommendations:
            st.write(rec)
    
    def render_optimization_models(self):
        """Render Optimization Models Interface"""
        st.header("Optimization Models")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Optimization Models Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("Optimization Models", {
                    "Has Data Manager": self.data_manager is not None,
                    "Available Datasets": len(self.data_manager.datasets) if self.data_manager else 0
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
        
        st.subheader("Staff Scheduling Optimization")
        
        # Staff scheduling problem
        shifts = ['Day', 'Evening', 'Night']
        departments = ['Emergency', 'Surgery', 'Cardiology', 'ICU']
        
        # Generate demand data
        demand_data = []
        for dept in departments:
            for shift in shifts:
                demand_data.append({
                    'Department': dept,
                    'Shift': shift,
                    'Demand': np.random.randint(5, 15),
                    'Current_Staff': np.random.randint(3, 12)
                })
        
        demand_df = pd.DataFrame(demand_data)
        
        # Display current vs demand
        fig = px.bar(
            demand_df,
            x='Department',
            y=['Demand', 'Current_Staff'],
            color='Shift',
            title='Staff Demand vs Current Allocation',
            barmode='group'
        )
        st.plotly_chart(fig, width="stretch")
        
        # Optimization parameters
        st.subheader("Optimization Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_staff = st.number_input("Minimum Staff per Shift", min_value=1, max_value=10, value=3)
            max_overtime = st.number_input("Maximum Overtime Hours", min_value=0, max_value=20, value=8)
        
        with col2:
            cost_per_hour = st.number_input("Cost per Hour", min_value=20, max_value=100, value=35)
            overtime_multiplier = st.number_input("Overtime Cost Multiplier", min_value=1.0, max_value=3.0, value=1.5)
        
        # Run optimization
        if st.button("Optimize Schedule"):
            # Simple optimization logic
            optimized_schedule = []
            total_cost = 0
            
            for _, row in demand_df.iterrows():
                demand = row['Demand']
                current = row['Current_Staff']
                
                if current < demand:
                    # Need more staff
                    additional = demand - current
                    overtime_hours = min(additional * 8, max_overtime)
                    regular_hours = (additional * 8) - overtime_hours
                    
                    cost = (regular_hours * cost_per_hour) + (overtime_hours * cost_per_hour * overtime_multiplier)
                    total_cost += cost
                    
                    optimized_schedule.append({
                        'Department': row['Department'],
                        'Shift': row['Shift'],
                        'Required': demand,
                        'Current': current,
                        'Additional': additional,
                        'Overtime_Hours': overtime_hours,
                        'Cost': cost
                    })
                else:
                    optimized_schedule.append({
                        'Department': row['Department'],
                        'Shift': row['Shift'],
                        'Required': demand,
                        'Current': current,
                        'Additional': 0,
                        'Overtime_Hours': 0,
                        'Cost': 0
                    })
            
            # Display results
            st.subheader("Optimization Results")
            
            results_df = pd.DataFrame(optimized_schedule)
            
            col1, col2 = st.columns(2)
            
            with col1:
                safe_dataframe_display(results_df)
            
            with col2:
                st.metric("Total Additional Cost", f"${total_cost:,}")
                
                understaffed = len(results_df[results_df['Additional'] > 0])
                st.metric("Understaffed Shifts", understaffed)
                
                total_overtime = results_df['Overtime_Hours'].sum()
                st.metric("Total Overtime Hours", total_overtime)
    
    def render_simulation_capabilities(self):
        """Render Simulation Capabilities Interface"""
        st.header("Simulation Capabilities")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Simulation Capabilities Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("Simulation Capabilities", {
                    "Has Data Manager": self.data_manager is not None,
                    "Available Datasets": len(self.data_manager.datasets) if self.data_manager else 0
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
        
        st.subheader("Patient Flow Simulation")
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_days = st.number_input("Simulation Days", min_value=1, max_value=365, value=30)
            arrival_rate = st.number_input("Patient Arrival Rate (per hour)", min_value=1, max_value=50, value=10)
        
        with col2:
            service_rate = st.number_input("Service Rate (per hour)", min_value=1, max_value=100, value=15)
            bed_capacity = st.number_input("Bed Capacity", min_value=10, max_value=200, value=50)
        
        # Run simulation
        if st.button("Run Simulation"):
            # Monte Carlo simulation
            np.random.seed(42)
            
            hours = simulation_days * 24
            simulation_data = []
            
            current_patients = 0
            total_arrivals = 0
            total_departures = 0
            max_queue = 0
            
            for hour in range(hours):
                # Arrivals
                arrivals = np.random.poisson(arrival_rate)
                total_arrivals += arrivals
                
                # Service completions
                if current_patients > 0:
                    departures = np.random.poisson(min(service_rate, current_patients))
                    total_departures += departures
                else:
                    departures = 0
                
                # Update patient count
                current_patients = max(0, current_patients + arrivals - departures)
                max_queue = max(max_queue, current_patients)
                
                simulation_data.append({
                    'Hour': hour,
                    'Arrivals': arrivals,
                    'Departures': departures,
                    'Queue_Length': current_patients,
                    'Utilization': min(100, (current_patients / bed_capacity) * 100)
                })
            
            sim_df = pd.DataFrame(simulation_data)
            
            # Display results
            st.subheader("Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Arrivals", total_arrivals)
            with col2:
                st.metric("Total Departures", total_departures)
            with col3:
                st.metric("Max Queue Length", max_queue)
            with col4:
                avg_utilization = sim_df['Utilization'].mean()
                st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
            
            # Visualization
            fig = px.line(
                sim_df,
                x='Hour',
                y=['Queue_Length', 'Utilization'],
                title='Patient Flow Simulation Results'
            )
            st.plotly_chart(fig, width="stretch")
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Average Queue Length', 'Peak Queue Length', 'Average Utilization', 'Peak Utilization'],
                'Value': [
                    sim_df['Queue_Length'].mean(),
                    sim_df['Queue_Length'].max(),
                    sim_df['Utilization'].mean(),
                    sim_df['Utilization'].max()
                ]
            })
            
            safe_dataframe_display(metrics_df)
    
    def render_ensemble_modeling(self):
        """Render Ensemble Modeling Interface"""
        st.header("Ensemble Modeling")
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            with st.expander("🔍 Ensemble Modeling Debug", expanded=False):
                debug_data = debug_manager.get_page_debug_data("Ensemble Modeling", {
                    "Has Data Manager": self.data_manager is not None,
                    "Available Datasets": len(self.data_manager.datasets) if self.data_manager else 0
                })
                
                for key, value in debug_data.items():
                    st.write(f"**{key}:** {value}")
                
                if self.data_manager:
                    st.markdown("---")
                    st.subheader("Available Datasets for Ensemble Modeling")
                    for dataset_name, df in self.data_manager.datasets.items():
                        st.write(f"**{dataset_name}**: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if not self.data_manager:
            st.warning("No data manager available")
            return
        
        # Dataset selection
        datasets = list(self.data_manager.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        if selected_dataset:
            df = self.data_manager.datasets[selected_dataset]
            
            # Select target and features
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 1:
                target_column = st.selectbox("Select Target Variable", numeric_columns)
                feature_columns = [col for col in numeric_columns if col != target_column]
                
                if feature_columns:
                    selected_features = st.multiselect("Select Features", feature_columns, default=feature_columns[:3])
                    
                    if selected_features and st.button("Train Ensemble Model"):
                        # Prepare data
                        X = df[selected_features].dropna()
                        y = df[target_column].dropna()
                        
                        # Align data
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        
                        if len(X) > 10:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Individual models
                            models = {
                                'Linear Regression': LinearRegression(),
                                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                            }
                            
                            # Train individual models
                            individual_predictions = {}
                            individual_scores = {}
                            
                            for name, model in models.items():
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                individual_predictions[name] = y_pred
                                individual_scores[name] = r2_score(y_test, y_pred)
                            
                            # Ensemble model
                            ensemble_model = VotingRegressor([
                                ('lr', models['Linear Regression']),
                                ('rf', models['Random Forest'])
                            ])
                            
                            ensemble_model.fit(X_train, y_train)
                            ensemble_pred = ensemble_model.predict(X_test)
                            ensemble_score = r2_score(y_test, ensemble_pred)
                            
                            # Display results
                            st.subheader("Model Performance Comparison")
                            
                            scores_df = pd.DataFrame([
                                {'Model': 'Linear Regression', 'R² Score': individual_scores['Linear Regression']},
                                {'Model': 'Random Forest', 'R² Score': individual_scores['Random Forest']},
                                {'Model': 'Ensemble', 'R² Score': ensemble_score}
                            ])
                            
                            fig = px.bar(
                                scores_df,
                                x='Model',
                                y='R² Score',
                                title='Model Performance Comparison'
                            )
                            st.plotly_chart(fig, width="stretch")
                            
                            # Prediction comparison
                            st.subheader("Prediction Comparison")
                            
                            comparison_df = pd.DataFrame({
                                'Actual': y_test.values[:20],
                                'Linear Regression': individual_predictions['Linear Regression'][:20],
                                'Random Forest': individual_predictions['Random Forest'][:20],
                                'Ensemble': ensemble_pred[:20]
                            })
                            
                            safe_dataframe_display(comparison_df)
                            
                            # Feature importance (for Random Forest)
                            if hasattr(models['Random Forest'], 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Feature': selected_features,
                                    'Importance': models['Random Forest'].feature_importances_
                                })
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title='Feature Importance (Random Forest)'
                                )
                                st.plotly_chart(fig, width="stretch")
                        else:
                            st.warning("Not enough data for training")
                    else:
                        st.warning("Please select features")
                else:
                    st.warning("No features available")
            else:
                st.warning("Need at least 2 numeric columns for ensemble modeling")


def show_advanced_analytics():
    """Show Advanced Analytics Interface"""
    module = AdvancedAnalyticsModule()
    
    # Analytics type selection
    analytics_type = st.selectbox(
        "Select Analytics Type",
        ["Time Series Analysis", "Prescriptive Analytics", "Optimization Models", "Simulation Capabilities", "Ensemble Modeling"]
    )
    
    if analytics_type == "Time Series Analysis":
        module.render_time_series_analysis()
    elif analytics_type == "Prescriptive Analytics":
        module.render_prescriptive_analytics()
    elif analytics_type == "Optimization Models":
        module.render_optimization_models()
    elif analytics_type == "Simulation Capabilities":
        module.render_simulation_capabilities()
    elif analytics_type == "Ensemble Modeling":
        module.render_ensemble_modeling()


def show_time_series_analysis():
    """Show Time Series Analysis"""
    module = AdvancedAnalyticsModule()
    module.render_time_series_analysis()


def show_prescriptive_analytics():
    """Show Prescriptive Analytics"""
    module = AdvancedAnalyticsModule()
    module.render_prescriptive_analytics()


def show_optimization_models():
    """Show Optimization Models"""
    module = AdvancedAnalyticsModule()
    module.render_optimization_models()


def show_simulation_capabilities():
    """Show Simulation Capabilities"""
    module = AdvancedAnalyticsModule()
    module.render_simulation_capabilities()


def show_ensemble_modeling():
    """Show Ensemble Modeling"""
    module = AdvancedAnalyticsModule()
    module.render_ensemble_modeling()
