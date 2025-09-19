"""
KPI Dashboard Module for Healthcare DSS

This module implements key performance indicators and dashboard metrics
based on real data from healthcare datasets.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class KPIDashboard:
    """
    KPI Dashboard for Healthcare DSS
    
    Calculates and visualizes key performance indicators based on real healthcare data.
    """
    
    def __init__(self, data_manager):
        """
        Initialize KPI Dashboard
        
        Args:
            data_manager: DataManager instance with loaded datasets
        """
        self.data_manager = data_manager
        self.kpi_metrics = {}
        
    def calculate_healthcare_kpis(self) -> Dict[str, Any]:
        """
        Calculate healthcare KPIs from available datasets
        
        Returns:
            Dictionary containing calculated KPI metrics
        """
        kpis = {}
        
        try:
            # Calculate diabetes-related KPIs
            if 'diabetes' in self.data_manager.datasets:
                diabetes_kpis = self._calculate_diabetes_kpis()
                kpis.update(diabetes_kpis)
            
            # Calculate breast cancer KPIs
            if 'breast_cancer' in self.data_manager.datasets:
                cancer_kpis = self._calculate_cancer_kpis()
                kpis.update(cancer_kpis)
            
            # Calculate healthcare expenditure KPIs
            if 'healthcare_expenditure' in self.data_manager.datasets:
                expenditure_kpis = self._calculate_expenditure_kpis()
                kpis.update(expenditure_kpis)
            
            # Calculate system performance KPIs
            system_kpis = self._calculate_system_kpis()
            kpis.update(system_kpis)
            
            self.kpi_metrics = kpis
            logger.info(f"Calculated {len(kpis)} KPI metrics")
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")
            raise
            
        return kpis
    
    def _calculate_diabetes_kpis(self) -> Dict[str, Any]:
        """Calculate diabetes-related KPIs"""
        df = self.data_manager.datasets['diabetes']
        
        # Calculate target variable statistics
        target_stats = df['target'].describe()
        
        # Calculate risk categories based on target values
        high_risk = (df['target'] > 200).sum()
        medium_risk = ((df['target'] >= 150) & (df['target'] <= 200)).sum()
        low_risk = (df['target'] < 150).sum()
        
        # Calculate feature correlations with target
        correlations = df.corr()['target'].abs().sort_values(ascending=False)
        
        return {
            'diabetes_patients_total': len(df),
            'diabetes_high_risk_count': high_risk,
            'diabetes_medium_risk_count': medium_risk,
            'diabetes_low_risk_count': low_risk,
            'diabetes_high_risk_percentage': (high_risk / len(df)) * 100,
            'diabetes_target_mean': target_stats['mean'],
            'diabetes_target_std': target_stats['std'],
            'diabetes_top_correlated_features': correlations.head(5).to_dict()
        }
    
    def _calculate_cancer_kpis(self) -> Dict[str, Any]:
        """Calculate breast cancer-related KPIs"""
        df = self.data_manager.datasets['breast_cancer']
        
        # Calculate target distribution
        target_counts = df['target'].value_counts()
        malignant_count = target_counts.get(0, 0)
        benign_count = target_counts.get(1, 0)
        
        # Calculate malignancy rate
        malignancy_rate = (malignant_count / len(df)) * 100
        
        # Calculate feature statistics for malignant vs benign
        malignant_df = df[df['target'] == 0]
        benign_df = df[df['target'] == 1]
        
        # Calculate mean radius difference (key diagnostic feature)
        mean_radius_diff = malignant_df['mean radius'].mean() - benign_df['mean radius'].mean()
        
        return {
            'cancer_patients_total': len(df),
            'cancer_malignant_count': malignant_count,
            'cancer_benign_count': benign_count,
            'cancer_malignancy_rate': malignancy_rate,
            'cancer_mean_radius_malignant': malignant_df['mean radius'].mean(),
            'cancer_mean_radius_benign': benign_df['mean radius'].mean(),
            'cancer_radius_difference': mean_radius_diff
        }
    
    def _calculate_expenditure_kpis(self) -> Dict[str, Any]:
        """Calculate healthcare expenditure KPIs"""
        df = self.data_manager.datasets['healthcare_expenditure'].copy()
        
        # Get year columns - handle the specific format in our dataset
        year_cols = [col for col in df.columns if '20' in col and '[' in col]
        
        # If no year columns found with brackets, try alternative patterns
        if not year_cols:
            year_cols = [col for col in df.columns if col.startswith('20') and len(col) == 4]
        
        # If still no year columns, look for any numeric columns that might be years
        if not year_cols:
            year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
        
        if not year_cols:
            logger.warning("No year columns found in healthcare expenditure dataset")
            return {
                'expenditure_countries_total': len(df),
                'expenditure_global_average': 0,
                'expenditure_global_std': 0,
                'expenditure_global_min': 0,
                'expenditure_global_max': 0,
                'expenditure_avg_growth_rate': 0,
                'expenditure_top_5_countries': {},
                'expenditure_bottom_5_countries': {}
            }
        
        # Clean and convert year columns to numeric
        available_year_cols = []
        for col in year_cols:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Only include columns that have at least some numeric data
                if not df[col].isna().all():
                    available_year_cols.append(col)
        
        if not available_year_cols:
            logger.warning("No valid numeric year columns found for expenditure calculation")
            return {
                'expenditure_countries_total': len(df),
                'expenditure_global_average': 0,
                'expenditure_global_std': 0,
                'expenditure_global_min': 0,
                'expenditure_global_max': 0,
                'expenditure_avg_growth_rate': 0,
                'expenditure_top_5_countries': {},
                'expenditure_bottom_5_countries': {}
            }
        
        # Calculate average expenditure by country
        country_avg = df.groupby('Country Name')[available_year_cols].mean()
        country_avg['avg_expenditure'] = country_avg[available_year_cols].mean(axis=1)
        
        # Calculate global statistics
        global_avg = country_avg['avg_expenditure'].mean()
        global_std = country_avg['avg_expenditure'].std()
        global_min = country_avg['avg_expenditure'].min()
        global_max = country_avg['avg_expenditure'].max()
        
        # Calculate expenditure growth rates
        growth_rates = []
        for country in country_avg.index:
            country_data = df[df['Country Name'] == country]
            if len(country_data) > 0 and len(available_year_cols) >= 2:
                first_year = country_data[available_year_cols[0]].mean()
                last_year = country_data[available_year_cols[-1]].mean()
                if pd.notna(first_year) and pd.notna(last_year) and first_year > 0:
                    growth_rate = ((last_year - first_year) / first_year) * 100
                    growth_rates.append(growth_rate)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        
        return {
            'expenditure_countries_total': len(country_avg),
            'expenditure_global_average': global_avg,
            'expenditure_global_std': global_std,
            'expenditure_global_min': global_min,
            'expenditure_global_max': global_max,
            'expenditure_avg_growth_rate': avg_growth_rate,
            'expenditure_top_5_countries': country_avg.nlargest(5, 'avg_expenditure')['avg_expenditure'].to_dict(),
            'expenditure_bottom_5_countries': country_avg.nsmallest(5, 'avg_expenditure')['avg_expenditure'].to_dict()
        }
    
    def _calculate_system_kpis(self) -> Dict[str, Any]:
        """Calculate system performance KPIs"""
        total_datasets = len(self.data_manager.datasets)
        total_records = sum(len(df) for df in self.data_manager.datasets.values())
        
        # Calculate data quality metrics
        data_quality_scores = []
        for name, df in self.data_manager.datasets.items():
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            data_quality_scores.append(completeness)
        
        avg_data_quality = np.mean(data_quality_scores)
        
        return {
            'system_datasets_loaded': total_datasets,
            'system_total_records': total_records,
            'system_avg_data_quality': avg_data_quality,
            'system_data_quality_scores': dict(zip(self.data_manager.datasets.keys(), data_quality_scores))
        }
    
    def create_kpi_dashboard(self) -> go.Figure:
        """
        Create interactive KPI dashboard
        
        Returns:
            Plotly figure with KPI dashboard
        """
        if not self.kpi_metrics:
            self.calculate_healthcare_kpis()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Diabetes Risk Distribution',
                'Cancer Malignancy Rate',
                'Healthcare Expenditure by Country',
                'System Performance Metrics',
                'Data Quality Scores',
                'Key Performance Indicators'
            ],
            specs=[
                [{"type": "pie"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # Diabetes risk distribution pie chart
        if 'diabetes_high_risk_count' in self.kpi_metrics:
            fig.add_trace(
                go.Pie(
                    labels=['High Risk', 'Medium Risk', 'Low Risk'],
                    values=[
                        self.kpi_metrics['diabetes_high_risk_count'],
                        self.kpi_metrics['diabetes_medium_risk_count'],
                        self.kpi_metrics['diabetes_low_risk_count']
                    ],
                    name="Diabetes Risk"
                ),
                row=1, col=1
            )
        
        # Cancer malignancy rate gauge
        if 'cancer_malignancy_rate' in self.kpi_metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=self.kpi_metrics['cancer_malignancy_rate'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Malignancy Rate (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ),
                row=1, col=2
            )
        
        # Healthcare expenditure bar chart
        if 'expenditure_top_5_countries' in self.kpi_metrics:
            countries = list(self.kpi_metrics['expenditure_top_5_countries'].keys())
            values = list(self.kpi_metrics['expenditure_top_5_countries'].values())
            
            fig.add_trace(
                go.Bar(
                    x=countries,
                    y=values,
                    name="Healthcare Expenditure",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # System performance gauge
        if 'system_avg_data_quality' in self.kpi_metrics:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=self.kpi_metrics['system_avg_data_quality'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 80], 'color': "lightgray"},
                            {'range': [80, 95], 'color': "yellow"},
                            {'range': [95, 100], 'color': "green"}
                        ]
                    }
                ),
                row=2, col=2
            )
        
        # Data quality scores bar chart
        if 'system_data_quality_scores' in self.kpi_metrics:
            datasets = list(self.kpi_metrics['system_data_quality_scores'].keys())
            scores = list(self.kpi_metrics['system_data_quality_scores'].values())
            
            fig.add_trace(
                go.Bar(
                    x=datasets,
                    y=scores,
                    name="Data Quality",
                    marker_color='lightgreen'
                ),
                row=3, col=1
            )
        
        # KPI summary table
        kpi_data = []
        if self.kpi_metrics:
            kpi_data = [
                ['Total Patients (Diabetes)', f"{self.kpi_metrics.get('diabetes_patients_total', 0):,}"],
                ['Total Patients (Cancer)', f"{self.kpi_metrics.get('cancer_patients_total', 0):,}"],
                ['Countries Analyzed', f"{self.kpi_metrics.get('expenditure_countries_total', 0):,}"],
                ['Total Records', f"{self.kpi_metrics.get('system_total_records', 0):,}"],
                ['Avg Data Quality', f"{self.kpi_metrics.get('system_avg_data_quality', 0):.1f}%"]
            ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=list(zip(*kpi_data)) if kpi_data else [[], []], fill_color='white')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Healthcare DSS KPI Dashboard",
            showlegend=False
        )
        
        return fig
    
    def generate_kpi_report(self) -> str:
        """
        Generate text-based KPI report
        
        Returns:
            Formatted KPI report string
        """
        if not self.kpi_metrics:
            self.calculate_healthcare_kpis()
        
        report = []
        report.append("=" * 60)
        report.append("HEALTHCARE DSS KPI REPORT")
        report.append("=" * 60)
        report.append("")
        
        # System Overview
        report.append("SYSTEM OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total Datasets Loaded: {self.kpi_metrics.get('system_datasets_loaded', 0)}")
        report.append(f"Total Records Processed: {self.kpi_metrics.get('system_total_records', 0):,}")
        report.append(f"Average Data Quality: {self.kpi_metrics.get('system_avg_data_quality', 0):.1f}%")
        report.append("")
        
        # Diabetes KPIs
        if 'diabetes_patients_total' in self.kpi_metrics:
            report.append("DIABETES ANALYSIS")
            report.append("-" * 20)
            report.append(f"Total Patients: {self.kpi_metrics['diabetes_patients_total']:,}")
            report.append(f"High Risk Patients: {self.kpi_metrics['diabetes_high_risk_count']:,} ({self.kpi_metrics['diabetes_high_risk_percentage']:.1f}%)")
            report.append(f"Medium Risk Patients: {self.kpi_metrics['diabetes_medium_risk_count']:,}")
            report.append(f"Low Risk Patients: {self.kpi_metrics['diabetes_low_risk_count']:,}")
            report.append(f"Average Target Value: {self.kpi_metrics['diabetes_target_mean']:.1f}")
            report.append("")
        
        # Cancer KPIs
        if 'cancer_patients_total' in self.kpi_metrics:
            report.append("BREAST CANCER ANALYSIS")
            report.append("-" * 25)
            report.append(f"Total Patients: {self.kpi_metrics['cancer_patients_total']:,}")
            report.append(f"Malignant Cases: {self.kpi_metrics['cancer_malignant_count']:,}")
            report.append(f"Benign Cases: {self.kpi_metrics['cancer_benign_count']:,}")
            report.append(f"Malignancy Rate: {self.kpi_metrics['cancer_malignancy_rate']:.1f}%")
            report.append(f"Mean Radius (Malignant): {self.kpi_metrics['cancer_mean_radius_malignant']:.2f}")
            report.append(f"Mean Radius (Benign): {self.kpi_metrics['cancer_mean_radius_benign']:.2f}")
            report.append("")
        
        # Healthcare Expenditure KPIs
        if 'expenditure_countries_total' in self.kpi_metrics:
            report.append("HEALTHCARE EXPENDITURE ANALYSIS")
            report.append("-" * 35)
            report.append(f"Countries Analyzed: {self.kpi_metrics['expenditure_countries_total']}")
            report.append(f"Global Average Expenditure: ${self.kpi_metrics['expenditure_global_average']:.0f} per capita")
            report.append(f"Global Min Expenditure: ${self.kpi_metrics['expenditure_global_min']:.0f} per capita")
            report.append(f"Global Max Expenditure: ${self.kpi_metrics['expenditure_global_max']:.0f} per capita")
            report.append(f"Average Growth Rate: {self.kpi_metrics['expenditure_avg_growth_rate']:.1f}%")
            report.append("")
            
            report.append("Top 5 Countries by Expenditure:")
            for country, expenditure in self.kpi_metrics['expenditure_top_5_countries'].items():
                report.append(f"  {country}: ${expenditure:.0f} per capita")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
