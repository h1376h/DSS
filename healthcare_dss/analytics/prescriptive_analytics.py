"""
Prescriptive Analytics Module for Healthcare DSS

This module implements prescriptive analytics and optimization models for
healthcare resource allocation, scheduling, and decision optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy.optimize import minimize, linprog
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PrescriptiveAnalyzer:
    """
    Prescriptive Analytics for Healthcare Decision Optimization
    
    Implements optimization models for resource allocation, scheduling,
    capacity planning, and treatment optimization in healthcare settings.
    """
    
    def __init__(self, data_manager):
        """
        Initialize Prescriptive Analyzer
        
        Args:
            data_manager: DataManager instance with loaded datasets
        """
        self.data_manager = data_manager
        self.optimization_results = {}
        self.optimization_models = {}
        
    def optimize_resource_allocation(self, dataset_name: str, 
                                   resource_constraints: Dict[str, float],
                                   objective_type: str = 'maximize_benefit') -> Dict[str, Any]:
        """
        Optimize resource allocation using linear programming
        
        Args:
            dataset_name: Name of the dataset to optimize
            resource_constraints: Dictionary of resource constraints
            objective_type: Type of optimization objective
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing resource allocation for {dataset_name}")
        
        if dataset_name == 'healthcare_expenditure':
            return self._optimize_healthcare_expenditure(resource_constraints, objective_type)
        elif dataset_name == 'diabetes':
            return self._optimize_diabetes_treatment(resource_constraints, objective_type)
        elif dataset_name == 'breast_cancer':
            return self._optimize_cancer_screening(resource_constraints, objective_type)
        else:
            raise ValueError(f"Resource allocation optimization not implemented for dataset: {dataset_name}")
    
    def _optimize_healthcare_expenditure(self, constraints: Dict[str, float], 
                                       objective_type: str) -> Dict[str, Any]:
        """
        Optimize healthcare expenditure allocation across countries
        """
        df = self.data_manager.datasets['healthcare_expenditure']
        
        # Get year columns - handle the specific format
        year_cols = [col for col in df.columns if '20' in col and '[' in col]
        
        # If no year columns found, try alternative approach
        if not year_cols:
            year_cols = [col for col in df.columns if col.startswith('20')]
        
        # If still no year columns, use all numeric columns except Country Name
        if not year_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            year_cols = [col for col in numeric_cols if col != 'Country Name']
        
        # Calculate average expenditure by country
        country_avg = df.groupby('Country Name').mean(numeric_only=True)
        
        # Only use available year columns
        available_year_cols = [col for col in year_cols if col in country_avg.columns]
        if available_year_cols:
            country_avg['avg_expenditure'] = country_avg[available_year_cols].mean(axis=1)
        else:
            # Fallback: use the first numeric column
            numeric_cols = country_avg.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                country_avg['avg_expenditure'] = country_avg[numeric_cols[0]]
            else:
                # Create dummy data if no numeric columns
                country_avg['avg_expenditure'] = np.random.uniform(100, 1000, len(country_avg))
        
        # Define optimization problem
        countries = country_avg.index.tolist()
        n_countries = len(countries)
        
        # Objective function coefficients (benefit per dollar spent)
        if objective_type == 'maximize_benefit':
            # Assume higher expenditure leads to better health outcomes
            c = -country_avg['avg_expenditure'].values  # Negative for maximization
        else:
            # Minimize total expenditure
            c = country_avg['avg_expenditure'].values
        
        # Constraints: total budget constraint
        A_ub = np.ones((1, n_countries))  # Sum of allocations <= total_budget
        b_ub = [constraints.get('total_budget', 1000000)]
        
        # Bounds: minimum and maximum allocation per country
        bounds = [(constraints.get('min_allocation', 0), 
                  constraints.get('max_allocation', 100000)) for _ in range(n_countries)]
        
        # Solve optimization problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        # Prepare results
        if result.success:
            optimal_allocations = result.x
            total_cost = np.sum(optimal_allocations)
            objective_value = -result.fun if objective_type == 'maximize_benefit' else result.fun
            
            allocation_results = []
            for i, country in enumerate(countries):
                allocation_results.append({
                    'country': country,
                    'current_expenditure': country_avg.loc[country, 'avg_expenditure'],
                    'optimal_allocation': optimal_allocations[i],
                    'allocation_change': optimal_allocations[i] - country_avg.loc[country, 'avg_expenditure']
                })
            
            results = {
                'optimization_type': 'resource_allocation',
                'objective': objective_type,
                'success': True,
                'total_cost': total_cost,
                'objective_value': objective_value,
                'allocations': allocation_results,
                'constraints_used': constraints
            }
        else:
            results = {
                'optimization_type': 'resource_allocation',
                'objective': objective_type,
                'success': False,
                'message': result.message,
                'constraints_used': constraints
            }
        
        self.optimization_results['healthcare_expenditure'] = results
        return results
    
    def _optimize_diabetes_treatment(self, constraints: Dict[str, float], 
                                   objective_type: str) -> Dict[str, Any]:
        """
        Optimize diabetes treatment allocation
        """
        df = self.data_manager.datasets['diabetes']
        
        # Create treatment options based on patient characteristics
        treatments = ['lifestyle', 'metformin', 'insulin', 'combination']
        
        # Define treatment effectiveness and cost
        treatment_data = {
            'lifestyle': {'effectiveness': 0.6, 'cost': 100, 'side_effects': 0.1},
            'metformin': {'effectiveness': 0.8, 'cost': 300, 'side_effects': 0.2},
            'insulin': {'effectiveness': 0.9, 'cost': 800, 'side_effects': 0.4},
            'combination': {'effectiveness': 0.85, 'cost': 500, 'side_effects': 0.3}
        }
        
        # Patient risk categories
        df['risk_category'] = pd.cut(df['target'], 
                                   bins=[-np.inf, 100, 150, 200, np.inf], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        risk_categories = df['risk_category'].value_counts()
        
        # Optimization variables: allocation of each treatment to each risk category
        n_treatments = len(treatments)
        n_categories = len(risk_categories)
        
        # Objective function: maximize total effectiveness
        c = []
        for treatment in treatments:
            for category in risk_categories.index:
                effectiveness = treatment_data[treatment]['effectiveness']
                # Weight by risk level
                risk_weight = {'Low': 1, 'Medium': 1.5, 'High': 2, 'Very High': 2.5}[category]
                c.append(-effectiveness * risk_weight)  # Negative for maximization
        
        # Constraints
        A_eq = []
        b_eq = []
        
        # Each patient must receive exactly one treatment
        for i, category in enumerate(risk_categories.index):
            constraint = [0] * (n_treatments * n_categories)
            for j, treatment in enumerate(treatments):
                constraint[i * n_treatments + j] = 1
            A_eq.append(constraint)
            b_eq.append(risk_categories[category])
        
        # Budget constraint
        A_ub = []
        b_ub = []
        
        cost_constraint = []
        for treatment in treatments:
            for category in risk_categories.index:
                cost_constraint.append(treatment_data[treatment]['cost'])
        A_ub.append(cost_constraint)
        b_ub.append(constraints.get('total_budget', 100000))
        
        # Bounds: non-negative allocations
        bounds = [(0, None) for _ in range(n_treatments * n_categories)]
        
        # Solve optimization
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            allocations = result.x.reshape(n_categories, n_treatments)
            
            allocation_results = []
            total_cost = 0
            total_effectiveness = 0
            
            for i, category in enumerate(risk_categories.index):
                category_allocations = {}
                for j, treatment in enumerate(treatments):
                    allocation = allocations[i, j]
                    if allocation > 0.1:  # Only show significant allocations
                        category_allocations[treatment] = {
                            'patients': int(allocation),
                            'cost': allocation * treatment_data[treatment]['cost'],
                            'effectiveness': allocation * treatment_data[treatment]['effectiveness']
                        }
                        total_cost += allocation * treatment_data[treatment]['cost']
                        total_effectiveness += allocation * treatment_data[treatment]['effectiveness']
                
                allocation_results.append({
                    'risk_category': category,
                    'total_patients': risk_categories[category],
                    'allocations': category_allocations
                })
            
            results = {
                'optimization_type': 'treatment_allocation',
                'objective': objective_type,
                'success': True,
                'total_cost': total_cost,
                'total_effectiveness': total_effectiveness,
                'allocations': allocation_results,
                'constraints_used': constraints
            }
        else:
            results = {
                'optimization_type': 'treatment_allocation',
                'objective': objective_type,
                'success': False,
                'message': result.message,
                'constraints_used': constraints
            }
        
        self.optimization_results['diabetes'] = results
        return results
    
    def _optimize_cancer_screening(self, constraints: Dict[str, float], 
                                 objective_type: str) -> Dict[str, Any]:
        """
        Optimize cancer screening resource allocation
        """
        df = self.data_manager.datasets['breast_cancer']
        
        # Define screening strategies
        strategies = ['basic_screening', 'advanced_screening', 'comprehensive_screening']
        
        strategy_data = {
            'basic_screening': {'sensitivity': 0.85, 'cost': 50, 'capacity': 200},
            'advanced_screening': {'sensitivity': 0.92, 'cost': 150, 'capacity': 100},
            'comprehensive_screening': {'sensitivity': 0.96, 'cost': 300, 'capacity': 50}
        }
        
        # Patient risk groups based on tumor characteristics
        df['risk_group'] = pd.cut(df['mean radius'], 
                                bins=[-np.inf, 10, 15, 20, np.inf], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
        
        risk_groups = df['risk_group'].value_counts()
        
        # Optimization: allocate screening strategies to risk groups
        n_strategies = len(strategies)
        n_groups = len(risk_groups)
        
        # Objective: maximize total sensitivity
        c = []
        for strategy in strategies:
            for group in risk_groups.index:
                sensitivity = strategy_data[strategy]['sensitivity']
                # Weight by risk level
                risk_weight = {'Low': 1, 'Medium': 1.5, 'High': 2, 'Very High': 2.5}[group]
                c.append(-sensitivity * risk_weight)
        
        # Constraints
        A_eq = []
        b_eq = []
        
        # Each patient must be assigned to exactly one strategy
        for i, group in enumerate(risk_groups.index):
            constraint = [0] * (n_strategies * n_groups)
            for j, strategy in enumerate(strategies):
                constraint[i * n_strategies + j] = 1
            A_eq.append(constraint)
            b_eq.append(risk_groups[group])
        
        # Capacity constraints
        A_ub = []
        b_ub = []
        
        for j, strategy in enumerate(strategies):
            capacity_constraint = [0] * (n_strategies * n_groups)
            for i in range(n_groups):
                capacity_constraint[i * n_strategies + j] = 1
            A_ub.append(capacity_constraint)
            b_ub.append(strategy_data[strategy]['capacity'])
        
        # Budget constraint
        cost_constraint = []
        for strategy in strategies:
            for group in risk_groups.index:
                cost_constraint.append(strategy_data[strategy]['cost'])
        A_ub.append(cost_constraint)
        b_ub.append(constraints.get('total_budget', 50000))
        
        bounds = [(0, None) for _ in range(n_strategies * n_groups)]
        
        # Solve optimization
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            allocations = result.x.reshape(n_groups, n_strategies)
            
            allocation_results = []
            total_cost = 0
            total_sensitivity = 0
            
            for i, group in enumerate(risk_groups.index):
                group_allocations = {}
                for j, strategy in enumerate(strategies):
                    allocation = allocations[i, j]
                    if allocation > 0.1:
                        group_allocations[strategy] = {
                            'patients': int(allocation),
                            'cost': allocation * strategy_data[strategy]['cost'],
                            'sensitivity': strategy_data[strategy]['sensitivity']
                        }
                        total_cost += allocation * strategy_data[strategy]['cost']
                        total_sensitivity += allocation * strategy_data[strategy]['sensitivity']
                
                allocation_results.append({
                    'risk_group': group,
                    'total_patients': risk_groups[group],
                    'allocations': group_allocations
                })
            
            results = {
                'optimization_type': 'screening_allocation',
                'objective': objective_type,
                'success': True,
                'total_cost': total_cost,
                'average_sensitivity': total_sensitivity / len(df),
                'allocations': allocation_results,
                'constraints_used': constraints
            }
        else:
            results = {
                'optimization_type': 'screening_allocation',
                'objective': objective_type,
                'success': False,
                'message': result.message,
                'constraints_used': constraints
            }
        
        self.optimization_results['breast_cancer'] = results
        return results
    
    def optimize_scheduling(self, dataset_name: str,
                          scheduling_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize scheduling using constraint satisfaction
        
        Args:
            dataset_name: Name of the dataset
            scheduling_constraints: Dictionary of scheduling constraints
            
        Returns:
            Dictionary with scheduling optimization results
        """
        logger.info(f"Optimizing scheduling for {dataset_name}")
        
        # Get the dataset
        if dataset_name not in self.data_manager.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.data_manager.datasets[dataset_name]
        
        # Generic scheduling optimization that works with any dataset
        return self._optimize_generic_scheduling(df, dataset_name, scheduling_constraints)
    
    def _optimize_generic_scheduling(self, df: pd.DataFrame, dataset_name: str, 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic scheduling optimization that adapts to any dataset
        
        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset
            constraints: Scheduling constraints
            
        Returns:
            Dictionary with scheduling optimization results
        """
        logger.info(f"Performing generic scheduling optimization for {dataset_name}")
        
        # Extract constraints with defaults
        max_hours = constraints.get('max_hours', 40)
        min_staff = constraints.get('min_staff', 5)
        max_appointments = constraints.get('max_appointments', 100)
        
        # Analyze dataset to determine scheduling parameters
        n_patients = len(df)
        
        # Find numeric columns that could represent priority/urgency
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use the first numeric column as priority indicator, or create one
        if numeric_cols:
            priority_col = numeric_cols[0]
            # Normalize priority scores (0-1 scale)
            priority_scores = (df[priority_col] - df[priority_col].min()) / (df[priority_col].max() - df[priority_col].min())
        else:
            # If no numeric columns, assign random priorities
            priority_scores = np.random.random(n_patients)
        
        # Define appointment types based on dataset characteristics
        appointment_types = self._get_appointment_types_for_dataset(dataset_name, df)
        
        # Assign appointment types based on priority
        df['priority_score'] = priority_scores
        df['appointment_type'] = df['priority_score'].apply(
            lambda x: 'emergency' if x > 0.8 else 'consultation' if x > 0.5 else 'follow_up'
        )
        
        # Calculate scheduling metrics
        total_appointments = min(n_patients, max_appointments)
        avg_appointment_duration = np.mean([appointment_types[t]['duration'] for t in appointment_types])
        total_hours_needed = total_appointments * avg_appointment_duration / 60
        
        # Calculate staffing requirements
        required_staff = max(min_staff, int(np.ceil(total_hours_needed / max_hours)))
        
        # Generate schedule
        schedule = self._generate_schedule(df, appointment_types, constraints)
        
        # Calculate optimization metrics
        utilization_rate = min(1.0, total_hours_needed / (required_staff * max_hours))
        efficiency_score = utilization_rate * 0.7 + (1 - (required_staff - min_staff) / min_staff) * 0.3
        
        results = {
            'optimization_type': 'scheduling',
            'dataset': dataset_name,
            'success': True,
            'total_appointments': total_appointments,
            'required_staff': required_staff,
            'total_hours_needed': total_hours_needed,
            'utilization_rate': utilization_rate,
            'efficiency_score': efficiency_score,
            'appointment_types': appointment_types,
            'schedule_summary': schedule,
            'constraints_met': {
                'max_hours': total_hours_needed <= max_hours * required_staff,
                'min_staff': required_staff >= min_staff,
                'max_appointments': total_appointments <= max_appointments
            }
        }
        
        # Store results
        self.optimization_results[f'{dataset_name}_scheduling'] = results
        return results
    
    def _get_appointment_types_for_dataset(self, dataset_name: str, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Determine appropriate appointment types based on dataset characteristics
        
        Args:
            dataset_name: Name of the dataset
            df: Dataset DataFrame
            
        Returns:
            Dictionary of appointment types with durations and priorities
        """
        # Base appointment types
        base_types = {
            'emergency': {'duration': 60, 'priority': 0},
            'consultation': {'duration': 30, 'priority': 1},
            'follow_up': {'duration': 15, 'priority': 2},
            'education': {'duration': 45, 'priority': 3}
        }
        
        # Adapt based on dataset characteristics
        if 'cancer' in dataset_name.lower():
            # Cancer-related datasets might need longer consultations
            base_types['consultation']['duration'] = 45
            base_types['treatment'] = {'duration': 90, 'priority': 1}
        elif 'diabetes' in dataset_name.lower():
            # Diabetes might need education sessions
            base_types['education']['duration'] = 60
        elif 'financial' in dataset_name.lower():
            # Financial datasets might represent administrative tasks
            base_types['meeting'] = {'duration': 30, 'priority': 2}
            base_types['review'] = {'duration': 20, 'priority': 3}
        
        return base_types
    
    def _generate_schedule(self, df: pd.DataFrame, appointment_types: Dict, 
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a basic schedule based on patient priorities and constraints
        
        Args:
            df: Dataset with priority scores and appointment types
            appointment_types: Available appointment types
            constraints: Scheduling constraints
            
        Returns:
            Schedule summary
        """
        # Sort by priority (lower priority score = higher priority)
        df_sorted = df.sort_values('priority_score')
        
        # Group by appointment type
        type_counts = df_sorted['appointment_type'].value_counts().to_dict()
        
        # Calculate time slots needed
        total_slots = sum(type_counts.values())
        max_hours = constraints.get('max_hours', 40)
        
        schedule = {
            'total_slots': total_slots,
            'appointment_distribution': type_counts,
            'estimated_duration_hours': sum(
                type_counts.get(appt_type, 0) * appointment_types[appt_type]['duration'] 
                for appt_type in appointment_types
            ) / 60,
            'schedule_feasible': total_slots <= max_hours * 4  # Assuming 15-min slots
        }
        
        return schedule
    
    def _optimize_diabetes_scheduling(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize diabetes patient scheduling
        """
        df = self.data_manager.datasets['diabetes']
        
        # Define appointment types and durations
        appointment_types = {
            'consultation': {'duration': 30, 'priority': 1},
            'follow_up': {'duration': 15, 'priority': 2},
            'education': {'duration': 45, 'priority': 3},
            'emergency': {'duration': 60, 'priority': 0}
        }
        
        # Patient priority based on target values
        df['priority'] = pd.cut(df['target'], 
                              bins=[-np.inf, 100, 150, 200, np.inf], 
                              labels=[3, 2, 1, 0])  # 0 = highest priority
        
        # Available time slots
        daily_hours = constraints.get('daily_hours', 8)
        time_slots = list(range(0, daily_hours * 60, 15))  # 15-minute slots
        
        # Simple scheduling algorithm
        scheduled_appointments = []
        available_slots = time_slots.copy()
        
        # Sort patients by priority
        patients_sorted = df.sort_values('priority').head(constraints.get('max_patients', 20))
        
        for _, patient in patients_sorted.iterrows():
            # Determine appointment type based on patient characteristics
            if patient['target'] > 200:
                appt_type = 'emergency'
            elif patient['target'] > 150:
                appt_type = 'consultation'
            elif patient['target'] > 100:
                appt_type = 'follow_up'
            else:
                appt_type = 'education'
            
            duration = appointment_types[appt_type]['duration']
            
            # Find available time slot
            for slot in available_slots:
                if slot + duration <= max(time_slots):
                    scheduled_appointments.append({
                        'patient_id': patient.name,
                        'appointment_type': appt_type,
                        'start_time': slot,
                        'duration': duration,
                        'priority': patient['priority'],
                        'target_value': patient['target']
                    })
                    
                    # Remove used time slots
                    for i in range(slot, slot + duration, 15):
                        if i in available_slots:
                            available_slots.remove(i)
                    break
        
        # Calculate scheduling metrics
        total_scheduled = len(scheduled_appointments)
        total_duration = sum(appt['duration'] for appt in scheduled_appointments)
        utilization_rate = total_duration / (daily_hours * 60) * 100
        
        results = {
            'optimization_type': 'scheduling',
            'success': True,
            'total_appointments': total_scheduled,
            'total_duration': total_duration,
            'utilization_rate': utilization_rate,
            'scheduled_appointments': scheduled_appointments,
            'constraints_used': constraints
        }
        
        self.optimization_results['scheduling'] = results
        return results
    
    def create_visualization(self, dataset_name: str, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create visualization of optimization results
        
        Args:
            dataset_name: Name of the dataset
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if dataset_name not in self.optimization_results:
            # Create placeholder figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No optimization results for {dataset_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Optimization Results Visualization')
            return fig
        
        results = self.optimization_results[dataset_name]
        
        if not results['success']:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Optimization failed: {results.get("message", "Unknown error")}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Optimization Results Visualization')
            return fig
        
        # Create subplots based on optimization type
        if results['optimization_type'] == 'resource_allocation':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Current vs Optimal allocations
            allocations = results['allocations']
            countries = [alloc['country'] for alloc in allocations]
            current = [alloc['current_expenditure'] for alloc in allocations]
            optimal = [alloc['optimal_allocation'] for alloc in allocations]
            
            x = np.arange(len(countries))
            width = 0.35
            
            ax1.bar(x - width/2, current, width, label='Current', alpha=0.7)
            ax1.bar(x + width/2, optimal, width, label='Optimal', alpha=0.7)
            ax1.set_xlabel('Countries')
            ax1.set_ylabel('Expenditure')
            ax1.set_title('Current vs Optimal Resource Allocation')
            ax1.set_xticks(x)
            ax1.set_xticklabels(countries, rotation=45)
            ax1.legend()
            
            # Plot 2: Allocation changes
            changes = [alloc['allocation_change'] for alloc in allocations]
            colors = ['green' if change > 0 else 'red' for change in changes]
            ax2.bar(countries, changes, color=colors, alpha=0.7)
            ax2.set_xlabel('Countries')
            ax2.set_ylabel('Allocation Change')
            ax2.set_title('Resource Allocation Changes')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Plot 3: Cost distribution
            ax3.pie([alloc['optimal_allocation'] for alloc in allocations], 
                   labels=countries, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Optimal Resource Distribution')
            
            # Plot 4: Summary metrics
            metrics = ['Total Cost', 'Objective Value']
            values = [results['total_cost'], results['objective_value']]
            ax4.bar(metrics, values, color=['skyblue', 'lightcoral'], alpha=0.7)
            ax4.set_ylabel('Value')
            ax4.set_title('Optimization Summary')
            
        elif results['optimization_type'] == 'treatment_allocation':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Treatment allocation by risk category
            categories = []
            treatments = []
            counts = []
            
            for alloc in results['allocations']:
                category = alloc['risk_category']
                for treatment, details in alloc['allocations'].items():
                    categories.append(category)
                    treatments.append(treatment)
                    counts.append(details['patients'])
            
            if categories:
                df_plot = pd.DataFrame({'Category': categories, 'Treatment': treatments, 'Count': counts})
                pivot_df = df_plot.pivot(index='Category', columns='Treatment', values='Count').fillna(0)
                pivot_df.plot(kind='bar', ax=ax1, stacked=True)
                ax1.set_title('Treatment Allocation by Risk Category')
                ax1.set_xlabel('Risk Category')
                ax1.set_ylabel('Number of Patients')
                ax1.legend(title='Treatment')
            
            # Plot 2: Cost distribution
            total_costs = []
            treatment_names = []
            for alloc in results['allocations']:
                for treatment, details in alloc['allocations'].items():
                    if treatment not in treatment_names:
                        treatment_names.append(treatment)
                        total_costs.append(0)
                    idx = treatment_names.index(treatment)
                    total_costs[idx] += details['cost']
            
            if treatment_names:
                ax2.pie(total_costs, labels=treatment_names, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Cost Distribution by Treatment')
            
            # Plot 3: Effectiveness vs Cost
            effectiveness = []
            costs = []
            for alloc in results['allocations']:
                for treatment, details in alloc['allocations'].items():
                    effectiveness.append(details['effectiveness'])
                    costs.append(details['cost'])
            
            if effectiveness:
                ax3.scatter(costs, effectiveness, alpha=0.7)
                ax3.set_xlabel('Cost')
                ax3.set_ylabel('Effectiveness')
                ax3.set_title('Effectiveness vs Cost')
            
            # Plot 4: Summary metrics
            metrics = ['Total Cost', 'Total Effectiveness']
            values = [results['total_cost'], results['total_effectiveness']]
            ax4.bar(metrics, values, color=['skyblue', 'lightgreen'], alpha=0.7)
            ax4.set_ylabel('Value')
            ax4.set_title('Optimization Summary')
        
        else:
            # Generic visualization
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Optimization Type: {results["optimization_type"]}\nSuccess: {results["success"]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Optimization Results')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, dataset_name: str) -> str:
        """
        Generate comprehensive prescriptive analytics report
        
        Args:
            dataset_name: Name of the analyzed dataset
            
        Returns:
            Formatted report string
        """
        if dataset_name not in self.optimization_results:
            return f"No optimization results found for dataset: {dataset_name}"
        
        results = self.optimization_results[dataset_name]
        
        report = []
        report.append("=" * 60)
        report.append("PRESCRIPTIVE ANALYTICS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Optimization summary
        report.append("OPTIMIZATION SUMMARY")
        report.append("-" * 20)
        report.append(f"Dataset: {dataset_name}")
        report.append(f"Optimization Type: {results['optimization_type']}")
        report.append(f"Objective: {results['objective']}")
        report.append(f"Success: {results['success']}")
        report.append("")
        
        if not results['success']:
            report.append(f"Error: {results.get('message', 'Unknown error')}")
            report.append("")
            report.append("=" * 60)
            return "\n".join(report)
        
        # Results details
        if results['optimization_type'] == 'resource_allocation':
            report.append("RESOURCE ALLOCATION RESULTS")
            report.append("-" * 30)
            report.append(f"Total Cost: ${results['total_cost']:,.2f}")
            report.append(f"Objective Value: {results['objective_value']:,.2f}")
            report.append("")
            
            report.append("Allocation Details:")
            for alloc in results['allocations']:
                report.append(f"  {alloc['country']}:")
                report.append(f"    Current: ${alloc['current_expenditure']:,.2f}")
                report.append(f"    Optimal: ${alloc['optimal_allocation']:,.2f}")
                report.append(f"    Change: ${alloc['allocation_change']:,.2f}")
                report.append("")
        
        elif results['optimization_type'] == 'treatment_allocation':
            report.append("TREATMENT ALLOCATION RESULTS")
            report.append("-" * 30)
            report.append(f"Total Cost: ${results['total_cost']:,.2f}")
            report.append(f"Total Effectiveness: {results['total_effectiveness']:,.2f}")
            report.append("")
            
            report.append("Allocation Details:")
            for alloc in results['allocations']:
                report.append(f"  {alloc['risk_category']} Risk ({alloc['total_patients']} patients):")
                for treatment, details in alloc['allocations'].items():
                    report.append(f"    {treatment}: {details['patients']} patients")
                    report.append(f"      Cost: ${details['cost']:,.2f}")
                    report.append(f"      Effectiveness: {details['effectiveness']:.2f}")
                report.append("")
        
        elif results['optimization_type'] == 'scheduling':
            report.append("SCHEDULING OPTIMIZATION RESULTS")
            report.append("-" * 35)
            report.append(f"Total Appointments: {results['total_appointments']}")
            report.append(f"Total Duration: {results['total_duration']} minutes")
            report.append(f"Utilization Rate: {results['utilization_rate']:.1f}%")
            report.append("")
            
            report.append("Scheduled Appointments:")
            for appt in results['scheduled_appointments'][:10]:  # Show first 10
                report.append(f"  Patient {appt['patient_id']}: {appt['appointment_type']}")
                report.append(f"    Time: {appt['start_time']}-{appt['start_time'] + appt['duration']} min")
                report.append(f"    Priority: {appt['priority']}")
                report.append("")
        
        # Constraints used
        report.append("CONSTRAINTS USED")
        report.append("-" * 15)
        for key, value in results['constraints_used'].items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
