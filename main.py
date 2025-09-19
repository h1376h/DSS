"""
Healthcare Decision Support System - Main Application
===================================================

This is the main entry point for the Healthcare DSS application.
It demonstrates the integration of all four subsystems and provides
a comprehensive example of the system capabilities.
"""

import os
import sys
import warnings
import traceback

# CRITICAL: Suppress warnings BEFORE any other imports
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Fix SSL certificate issues for dataset downloads
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress all deprecation warnings at the system level
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Parsing dates involving a day of month without a year.*")
warnings.filterwarnings("ignore", message=".*ipykernel.comm.Comm.*")
warnings.filterwarnings("ignore", message=".*Jupyter is migrating its paths.*")
warnings.filterwarnings("ignore", message=".*The `ipykernel.comm.Comm` class has been deprecated.*")
warnings.filterwarnings("ignore", message=".*datetime.strptime.*")
warnings.filterwarnings("ignore", module="ipykernel.*")
warnings.filterwarnings("ignore", module="dash.*")
warnings.filterwarnings("ignore", module="jupyter.*")
warnings.filterwarnings("ignore", module="streamlit.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Thread 'MainThread': missing ScriptRunContext.*")

from pathlib import Path
import logging
import argparse
import time
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from healthcare_dss import DataManager, ModelManager, KnowledgeManager, DashboardManager
from healthcare_dss.config import setup_logging, get_config, ensure_directories, LoggerMixin
from healthcare_dss.ui import KPIDashboard
from healthcare_dss.utils.crisp_dm_workflow import CRISPDMWorkflow
# DatasetManager functionality is now integrated into DataManager
from healthcare_dss.analytics import ClassificationEvaluator, AssociationRulesMiner, ClusteringAnalyzer, TimeSeriesAnalyzer, PrescriptiveAnalyzer
from healthcare_dss.utils.debug_manager import debug_manager, debug_write, log_database_query, log_model_training, update_performance_metric
from healthcare_dss.config.clinical_config import ClinicalConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthcare_dss.log'),
        logging.StreamHandler()
    ]
)

# Suppress Streamlit warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class HealthcareDSS(LoggerMixin):
    """
    Main Healthcare Decision Support System class
    
    Implements the comprehensive DSS methodology following Herbert Simon's four-phase model:
    1. Intelligence Phase: Problem identification and data gathering
    2. Design Phase: Model formulation and alternative development
    3. Choice Phase: Solution evaluation and selection
    4. Implementation Phase: Deployment and monitoring
    
    Integrates all four subsystems:
    1. Data Management Subsystem
    2. Model Management Subsystem
    3. Knowledge-Based Management Subsystem
    4. User Interface Subsystem
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the Healthcare DSS
        
        Args:
            data_dir: Directory containing healthcare datasets (optional, uses config default)
        """
        # Setup configuration and logging
        self.config = get_config()
        ensure_directories()
        setup_logging()
        
        # Initialize debug system
        self.debug_manager = debug_manager
        debug_write("Healthcare DSS initialization started", "SYSTEM")
        
        self.data_dir = data_dir or str(self.config['data_dir'])
        self.data_manager = None
        self.model_manager = None
        self.knowledge_manager = None
        self.dashboard_manager = None
        self.kpi_dashboard = None
        self.crisp_dm_workflow = None
        self.classification_evaluator = None
        self.association_rules_miner = None
        self.clustering_analyzer = None
        self.time_series_analyzer = None
        self.prescriptive_analyzer = None
        
        self.log_info("Initializing Healthcare Decision Support System...")
        debug_write("Starting subsystem initialization", "SYSTEM")
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """Initialize all four subsystems"""
        try:
            # Initialize Data Management Subsystem
            self.log_info("Initializing Data Management Subsystem...")
            debug_write("Initializing Data Management Subsystem", "SYSTEM")
            self.data_manager = DataManager()
            update_performance_metric("Data Manager Initialized", 1)
            
            # Initialize Model Management Subsystem
            self.log_info("Initializing Model Management Subsystem...")
            debug_write("Initializing Model Management Subsystem", "SYSTEM")
            self.model_manager = ModelManager(self.data_manager)
            update_performance_metric("Model Manager Initialized", 1)
            
            # Initialize Knowledge-Based Management Subsystem
            self.log_info("Initializing Knowledge-Based Management Subsystem...")
            debug_write("Initializing Knowledge-Based Management Subsystem", "SYSTEM")
            self.knowledge_manager = KnowledgeManager(self.data_manager, self.model_manager)
            update_performance_metric("Knowledge Manager Initialized", 1)
            
            # Initialize User Interface Subsystem
            self.log_info("Initializing User Interface Subsystem...")
            debug_write("Initializing User Interface Subsystem", "SYSTEM")
            self.dashboard_manager = DashboardManager(
                self.data_manager, 
                self.model_manager, 
                self.knowledge_manager
            )
            update_performance_metric("Dashboard Manager Initialized", 1)
            
            # Initialize additional modules
            self.log_info("Initializing KPI Dashboard...")
            debug_write("Initializing KPI Dashboard", "SYSTEM")
            self.kpi_dashboard = KPIDashboard(self.data_manager)
            
            self.log_info("Initializing CRISP-DM Workflow...")
            debug_write("Initializing CRISP-DM Workflow", "SYSTEM")
            self.crisp_dm_workflow = CRISPDMWorkflow(self.data_manager)
            
            self.log_info("Initializing Classification Evaluator...")
            debug_write("Initializing Classification Evaluator", "SYSTEM")
            self.classification_evaluator = ClassificationEvaluator()
            
            self.log_info("Initializing Association Rules Miner...")
            debug_write("Initializing Association Rules Miner", "SYSTEM")
            self.association_rules_miner = AssociationRulesMiner(self.data_manager)
            
            self.log_info("Initializing Clustering Analyzer...")
            debug_write("Initializing Clustering Analyzer", "SYSTEM")
            self.clustering_analyzer = ClusteringAnalyzer(self.data_manager)
            
            self.log_info("Initializing Time Series Analyzer...")
            debug_write("Initializing Time Series Analyzer", "SYSTEM")
            self.time_series_analyzer = TimeSeriesAnalyzer(self.data_manager)
            
            self.log_info("Initializing Prescriptive Analyzer...")
            debug_write("Initializing Prescriptive Analyzer", "SYSTEM")
            self.prescriptive_analyzer = PrescriptiveAnalyzer(self.data_manager)
            
            self.log_info("All subsystems initialized successfully!")
            debug_write("All subsystems initialized successfully", "SYSTEM")
            update_performance_metric("Total Subsystems Initialized", 8)
            
        except Exception as e:
            self.log_error(f"Error initializing subsystems: {e}")
            debug_write(f"Error initializing subsystems: {e}", "ERROR", {"error": str(e), "traceback": traceback.format_exc()})
            raise
    
    def run_dss_methodology_demo(self):
        """Run demonstration following DSS methodology phases"""
        logger.info("Starting DSS methodology demonstration...")
        
        print("\n" + "="*80)
        print("HEALTHCARE DSS METHODOLOGY DEMONSTRATION")
        print("Following Herbert Simon's Four-Phase Decision-Making Model")
        print("="*80)
        
        # Phase 1: Intelligence Phase
        self._demonstrate_intelligence_phase()
        
        # Phase 2: Design Phase
        self._demonstrate_design_phase()
        
        # Phase 3: Choice Phase
        self._demonstrate_choice_phase()
        
        # Phase 4: Implementation Phase
        self._demonstrate_implementation_phase()
        
        print("\n" + "="*80)
        print("DSS METHODOLOGY DEMONSTRATION COMPLETED!")
        print("="*80)
    
    def _demonstrate_intelligence_phase(self):
        """Demonstrate Intelligence Phase: Problem identification and data gathering"""
        print("\nPHASE 1: INTELLIGENCE - PROBLEM IDENTIFICATION")
        print("-" * 60)
        
        # Problem identification example
        print("Problem: High patient readmission rates in cardiology department")
        print("Symptom Analysis:")
        print("  - 30-day readmission rate: 25% (above national average of 12%)")
        print("  - Patient satisfaction scores declining")
        print("  - Increased healthcare costs")
        
        print("\nRoot Cause Analysis:")
        print("  - Poor discharge planning")
        print("  - Inadequate patient education")
        print("  - Lack of follow-up care coordination")
        
        # Data gathering demonstration
        print("\nData Gathering:")
        print("  - Patient demographics and medical history")
        print("  - Admission/discharge patterns")
        print("  - Readmission reasons and timing")
        print("  - Resource utilization data")
        
        # Stakeholder identification
        print("\nStakeholder Analysis:")
        print("  - Chief Medical Officer (Problem Owner)")
        print("  - Cardiology Department Head")
        print("  - Nursing Staff")
        print("  - Patient Care Coordinators")
        
        # Decision classification
        print("\nDecision Classification:")
        print("  - Type: Semi-structured")
        print("  - Level: Managerial Control")
        print("  - Complexity: High (multiple variables, stakeholders)")
    
    def _demonstrate_design_phase(self):
        """Demonstrate Design Phase: Model formulation and alternative development"""
        print("\nPHASE 2: DESIGN - MODEL FORMULATION & ALTERNATIVES")
        print("-" * 60)
        
        # Model formulation
        print("Model Formulation:")
        print("  - Predictive model for readmission risk")
        print("  - Patient flow optimization model")
        print("  - Resource allocation model")
        
        # Alternative solutions
        print("\nAlternative Solutions:")
        alternatives = [
            {
                'id': 'predictive_model',
                'name': 'Predictive Analytics + Care Coordinator',
                'description': 'ML model to identify high-risk patients + dedicated care coordinator',
                'cost': 'Medium',
                'effectiveness': 'High',
                'implementation_time': '6 months'
            },
            {
                'id': 'education_program',
                'name': 'Enhanced Patient Education Program',
                'description': 'Comprehensive education materials and follow-up calls',
                'cost': 'Low',
                'effectiveness': 'Medium',
                'implementation_time': '2 months'
            },
            {
                'id': 'telemedicine',
                'name': 'Telemedicine Follow-up System',
                'description': 'Remote monitoring and virtual consultations',
                'cost': 'High',
                'effectiveness': 'High',
                'implementation_time': '9 months'
            }
        ]
        
        for i, alt in enumerate(alternatives, 1):
            print(f"  {i}. {alt['name']}")
            print(f"     Description: {alt['description']}")
            print(f"     Cost: {alt['cost']}, Effectiveness: {alt['effectiveness']}")
            print(f"     Implementation: {alt['implementation_time']}")
            print()
        
        # Success criteria
        print("Success Criteria:")
        print("  - Reduce 30-day readmission rate to <12%")
        print("  - Improve patient satisfaction scores by 20%")
        print("  - Achieve positive ROI within 18 months")
        print("  - Maintain or improve clinical outcomes")
    
    def _demonstrate_choice_phase(self):
        """Demonstrate Choice Phase: Solution evaluation and selection"""
        print("\nPHASE 3: CHOICE - SOLUTION EVALUATION & SELECTION")
        print("-" * 60)
        
        # Sensitivity analysis
        print("Sensitivity Analysis:")
        print("  - Model accuracy varies with data quality (85-95%)")
        print("  - Implementation success depends on staff adoption")
        print("  - ROI sensitive to readmission rate reduction")
        
        # What-if analysis
        print("\nWhat-If Analysis:")
        scenarios = [
            "What if readmission rate only reduces to 15%?",
            "What if implementation takes 12 months instead of 6?",
            "What if staff resistance is higher than expected?"
        ]
        
        for scenario in scenarios:
            print(f"  - {scenario}")
        
        # Multi-criteria decision analysis
        print("\nMulti-Criteria Decision Analysis:")
        print("  Criteria Weights:")
        print("    - Effectiveness: 40%")
        print("    - Cost: 25%")
        print("    - Implementation Time: 20%")
        print("    - Risk: 15%")
        
        print("\n  Alternative Rankings:")
        print("    1. Predictive Analytics + Care Coordinator (Score: 8.5)")
        print("    2. Telemedicine Follow-up System (Score: 7.8)")
        print("    3. Enhanced Patient Education Program (Score: 6.2)")
        
        print("\n  Selected Solution: Predictive Analytics + Care Coordinator")
        print("  Rationale: Best balance of effectiveness, cost, and implementation time")
    
    def _demonstrate_implementation_phase(self):
        """Demonstrate Implementation Phase: Deployment and monitoring"""
        print("\nPHASE 4: IMPLEMENTATION - DEPLOYMENT & MONITORING")
        print("-" * 60)
        
        # Implementation plan
        print("Implementation Plan:")
        phases = [
            ("Months 1-2", "System development and testing"),
            ("Months 3-4", "Staff training and pilot testing"),
            ("Months 5-6", "Full deployment and optimization"),
            ("Ongoing", "Monitoring and continuous improvement")
        ]
        
        for phase, description in phases:
            print(f"  {phase}: {description}")
        
        # Change management
        print("\nChange Management Strategy:")
        print("  - Clinical champion identification and training")
        print("  - Comprehensive staff education program")
        print("  - Gradual rollout with feedback collection")
        print("  - Performance monitoring and adjustment")
        
        # Monitoring and evaluation
        print("\nMonitoring & Evaluation:")
        print("  - Real-time readmission rate tracking")
        print("  - Patient satisfaction surveys")
        print("  - Staff feedback and adoption metrics")
        print("  - Financial impact assessment")
        
        # Success metrics
        print("\nSuccess Metrics:")
        print("  - 30-day readmission rate: Target <12%")
        print("  - Patient satisfaction: Target >8.5/10")
        print("  - Staff adoption: Target >90%")
        print("  - ROI: Target positive within 18 months")

    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all system capabilities"""
        logger.info("Starting comprehensive system demonstration...")
        
        print("\n" + "="*80)
        print("HEALTHCARE DECISION SUPPORT SYSTEM - COMPREHENSIVE DEMO")
        print("="*80)
        
        # 1. Data Management Demo
        self._demo_data_management()
        
        # 2. Model Management Demo
        self._demo_model_management()
        
        # 3. Knowledge Management Demo
        self._demo_knowledge_management()
        
        # 4. Integrated Decision Support Demo
        self._demo_integrated_decision_support()
        
        # 5. KPI Dashboard Demo
        self._demo_kpi_dashboard()
        
        # 6. CRISP-DM Workflow Demo
        self._demo_crisp_dm_workflow()
        
        # 7. Classification Evaluation Demo
        self._demo_classification_evaluation()
        
        # 8. Association Rules Mining Demo
        self._demo_association_rules()
        
        # 9. Clustering Analysis Demo
        self._demo_clustering_analysis()
        
        # 10. Time Series Analysis Demo
        self._demo_time_series_analysis()
        
        # 11. Prescriptive Analytics Demo
        self._demo_prescriptive_analytics()
        
        # 12. System Performance Summary
        self._demo_system_performance()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def run_workflow_demo(self):
        """Run workflow demonstration with all analytics modules"""
        logger.info("Starting workflow demonstration...")
        
        print("\n" + "="*80)
        print("HEALTHCARE DSS WORKFLOW DEMONSTRATION")
        print("="*80)
        
        # 1. CRISP-DM Workflow Demo
        self._demo_crisp_dm_workflow()
        
        # 2. Classification Evaluation Demo
        self._demo_classification_evaluation()
        
        # 3. Association Rules Mining Demo
        self._demo_association_rules()
        
        # 4. Clustering Analysis Demo
        self._demo_clustering_analysis()
        
        # 5. Time Series Analysis Demo
        self._demo_time_series_analysis()
        
        # 6. Prescriptive Analytics Demo
        self._demo_prescriptive_analytics()
        
        print("\n" + "="*80)
        print("WORKFLOW DEMONSTRATION COMPLETED!")
        print("="*80)
    
    def _demo_data_management(self):
        """Demonstrate Data Management Subsystem capabilities"""
        print("\nDATA MANAGEMENT SUBSYSTEM DEMONSTRATION")
        print("-" * 50)
        
        # Show available datasets
        print(f"Available datasets: {list(self.data_manager.datasets.keys())}")
        
        # Data quality assessment
        for dataset_name in self.data_manager.datasets.keys():
            print(f"\nAssessing data quality for {dataset_name}...")
            quality_metrics = self.data_manager.assess_data_quality(dataset_name)
            print(f"  - Shape: {quality_metrics['shape']}")
            print(f"  - Completeness: {quality_metrics['completeness_score']:.1f}%")
            print(f"  - Missing values: {sum(quality_metrics['missing_values'].values())}")
            
            # Show data preprocessing checklist
            print(f"\nData Preprocessing Checklist for {dataset_name}:")
            checklist = self.data_manager.get_data_preprocessing_checklist(dataset_name)
            
            # Show preprocessing tasks status
            for task_type, tasks in checklist['preprocessing_tasks'].items():
                print(f"  {task_type.title()} Tasks:")
                for task, status in tasks.items():
                    if isinstance(status, bool):
                        status_str = "✅" if status else "❌"
                        print(f"    {status_str} {task.replace('_', ' ').title()}")
                    elif isinstance(status, dict):
                        print(f"    📊 {task.replace('_', ' ').title()}: {len(status)} columns analyzed")
            
            # Show recommendations
            if checklist['recommendations']:
                print(f"\n  Recommendations:")
                for rec in checklist['recommendations']:
                    print(f"    • {rec}")
            
            # Show healthcare considerations
            if checklist['healthcare_specific_considerations']:
                print(f"\n  Healthcare Considerations:")
                for consideration in checklist['healthcare_specific_considerations'][:3]:  # Show first 3
                    print(f"    • {consideration}")
        
        # Healthcare expenditure analysis
        print("\nHealthcare Expenditure Analysis:")
        expenditure_analysis = self.data_manager.get_healthcare_expenditure_analysis()
        print(f"  - Countries analyzed: {expenditure_analysis['total_countries']}")
        print(f"  - Years covered: {len(expenditure_analysis['years_covered'])}")
        
        # Show top 5 countries by expenditure
        top_countries = sorted(
            expenditure_analysis['expenditure_trends'].items(),
            key=lambda x: x[1]['avg_expenditure'],
            reverse=True
        )[:5]
        
        print("  - Top 5 countries by average expenditure:")
        for country, data in top_countries:
            print(f"    {country}: ${data['avg_expenditure']:.0f} per capita")
    
    def _demo_model_management(self):
        """Demonstrate Model Management Subsystem capabilities"""
        print("\nMODEL MANAGEMENT SUBSYSTEM DEMONSTRATION")
        print("-" * 50)
        
        # Get smart target suggestions
        try:
            from healthcare_dss.utils.smart_target_manager import SmartDatasetTargetManager
            smart_manager = SmartDatasetTargetManager()
            
            # Get smart suggestions for diabetes dataset
            diabetes_targets = smart_manager.get_dataset_targets("diabetes")
            diabetes_models = smart_manager.get_recommended_models("diabetes")
            diabetes_insights = smart_manager.get_smart_insights("diabetes")
            
            print("Smart Target Suggestions:")
            for target in diabetes_targets:
                print(f"  - {target['column']}: {target['target_type']} - {target.get('business_meaning', 'Target variable')}")
            
            print(f"\nRecommended Models: {diabetes_models}")
            
            if diabetes_insights.get('business_value'):
                print("Business Value:")
                for value in diabetes_insights['business_value']:
                    print(f"  - {value}")
            
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"Could not load smart suggestions: {e}")
            print("Using default configuration...")
        
        # Train models on diabetes dataset
        print("Training models on diabetes dataset...")
        
        # Use smart recommended models if available
        try:
            if 'diabetes_models' in locals() and diabetes_models:
                # Map smart recommendations to our model names
                smart_model_mapping = {
                    'RandomForestRegressor': 'random_forest',
                    'XGBRegressor': 'xgboost',
                    'LinearRegression': 'linear_regression',
                    'SVR': 'svm'
                }
                
                models_to_train = []
                for model in diabetes_models:
                    mapped_model = smart_model_mapping.get(model, model.lower().replace('regressor', ''))
                    if mapped_model in ['random_forest', 'xgboost', 'lightgbm', 'linear_regression', 'svm']:
                        models_to_train.append(mapped_model)
                
                if not models_to_train:
                    models_to_train = ['random_forest', 'xgboost', 'lightgbm']
            else:
                models_to_train = ['random_forest', 'xgboost', 'lightgbm']
        except:
            models_to_train = ['random_forest', 'xgboost', 'lightgbm']
        
        model_results = {}
        
        for model_name in models_to_train:
            try:
                print(f"  Training {model_name}...")
                debug_write(f"Starting training for {model_name}", "MODEL", {"model": model_name, "dataset": "diabetes"})
                
                start_time = time.time()
                result = self.model_manager.train_model(
                    dataset_name='diabetes',
                    model_name=model_name,
                    task_type='regression',
                    target_column='target'
                )
                training_time = time.time() - start_time
                
                model_results[model_name] = result['metrics']['r2_score']
                print(f"    R² Score: {result['metrics']['r2_score']:.4f}")
                
                # Log model training with debug system
                log_model_training(model_name, 'diabetes', result['metrics'], training_time)
                debug_write(f"Completed training for {model_name}", "MODEL", {
                    "model": model_name, 
                    "r2_score": result['metrics']['r2_score'],
                    "training_time": training_time
                })
                
            except Exception as e:
                print(f"    Error: {e}")
                debug_write(f"Error training {model_name}: {e}", "ERROR", {"model": model_name, "error": str(e)})
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        try:
            ensemble_result = self.model_manager.create_ensemble_model(
                dataset_name='diabetes',
                models=['random_forest', 'xgboost', 'linear_regression', 'decision_tree'],
                task_type='regression',
                target_column='target'
            )
            print(f"  Ensemble R² Score: {ensemble_result['metrics']['r2_score']:.4f}")
        except Exception as e:
            print(f"  Error creating ensemble: {e}")
        
        # Model comparison
        print("\nModel Performance Comparison:")
        try:
            comparison = self.model_manager.compare_models(
                dataset_name='diabetes',
                task_type='regression',
                target_column='target'
            )
            print(comparison.to_string(index=False))
        except Exception as e:
            print(f"  Error in model comparison: {e}")
        
        # AI Technology Selection Matrix Demo
        print("\nAI Technology Selection Matrix:")
        ai_matrix = self.model_manager.get_ai_technology_selection_matrix()
        
        print("Healthcare Problems and AI Solutions:")
        for problem_key, problem_data in ai_matrix['healthcare_problems'].items():
            print(f"  {problem_data['problem']}")
            print(f"    AI Technology: {problem_data['ai_technology']}")
            print(f"    Expected ROI: {problem_data['expected_roi']}")
            print()
        
        # Knowledge Acquisition Plan Demo
        print("Knowledge Acquisition Plan:")
        knowledge_plan = self.model_manager.get_knowledge_acquisition_plan("Diabetes Management DSS")
        print(f"  Project: {knowledge_plan['project_name']}")
        print("  Knowledge Sources:")
        for source_key, source_data in knowledge_plan['knowledge_sources'].items():
            print(f"    {source_key.replace('_', ' ').title()}: {source_data['acquisition_method']}")
        
        # AI vs Human Intelligence Comparison
        print("\nAI vs Human Intelligence Comparison:")
        comparison_data = ai_matrix['ai_vs_human_comparison']
        for aspect, data in comparison_data.items():
            print(f"  {aspect.replace('_', ' ').title()}:")
            print(f"    AI: {data['ai']}")
            print(f"    Human: {data['human']}")
            print(f"    Healthcare Context: {data['healthcare_context']}")
            print()
        
        # AI Model Performance Evaluation
        print("AI Model Performance Evaluation:")
        performance = self.model_manager.evaluate_ai_model_performance("Random Forest", "Diabetes Prediction")
        print(f"  Model: {performance['model_name']}")
        print(f"  Context: {performance['healthcare_context']}")
        print("  Key Metrics:")
        for metric, data in performance['metrics'].items():
            print(f"    {metric.title()}: {data['actual_performance']} (Target: {data['target']})")
        
        print(f"\n  Turing Test Results: {performance['turing_test_results']['indistinguishability_score']}% indistinguishable")
    
    def _demo_knowledge_management(self):
        """Demonstrate Knowledge-Based Management Subsystem capabilities"""
        print("\nKNOWLEDGE-BASED MANAGEMENT SUBSYSTEM DEMONSTRATION")
        print("-" * 50)
        
        # Knowledge base summary
        knowledge_summary = self.knowledge_manager.get_knowledge_summary()
        print(f"Knowledge Base Contents:")
        print(f"  - Clinical Rules: {knowledge_summary['clinical_rules']}")
        print(f"  - Clinical Guidelines: {knowledge_summary['clinical_guidelines']}")
        print(f"  - Decision Trees: {knowledge_summary['decision_trees']}")
        print(f"  - Active Rules: {knowledge_summary['active_rules']}")
        
        # Test patient scenario
        print("\nTesting patient scenario:")
        patient_data = {
            'age': 55,
            'bmi': 28.5,
            'systolic_bp': 145,
            'diastolic_bp': 95,
            'family_history': True,
            'hba1c': 7.2
        }
        
        print(f"Patient data: {patient_data}")
        
        # Evaluate clinical rules
        triggered_rules = self.knowledge_manager.evaluate_clinical_rules(patient_data)
        print(f"\nTriggered clinical rules: {len(triggered_rules)}")
        for rule in triggered_rules:
            print(f"  - {rule['name']} (Severity: {rule['severity']})")
        
        # Get clinical recommendations
        recommendations = self.knowledge_manager.get_clinical_recommendations(patient_data)
        print(f"\nClinical recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['recommendation']}")
        
        # Decision tree evaluation
        print("\nDecision tree evaluation:")
        try:
            diabetes_result = self.knowledge_manager.evaluate_decision_tree(
                "diabetes_risk_tree", patient_data
            )
            print(f"  - Risk assessment: {diabetes_result['outcome']}")
            print(f"  - Recommendations: {', '.join(diabetes_result['recommendations'])}")
        except Exception as e:
            print(f"  Error in decision tree evaluation: {e}")
        
        # Clinical guidelines integration demonstration
        print("\n" + "-"*40)
        print("CLINICAL GUIDELINES INTEGRATION")
        print("-"*40)
        
        guidelines = self.knowledge_manager.get_clinical_guidelines_integration()
        print("Evidence-Based Protocols Available:")
        for condition, protocol in guidelines['evidence_based_protocols'].items():
            print(f"  - {condition}: {protocol['guideline_source']} ({protocol['evidence_level']})")
        
        # Demonstrate evidence-based protocols
        diabetes_protocols = self.knowledge_manager.get_evidence_based_protocols("diabetes")
        print(f"\nDiabetes Management Protocols:")
        print(f"  Diagnostic Criteria: {diabetes_protocols['diagnostic_criteria']}")
        print(f"  Treatment Steps: {len(diabetes_protocols['treatment_algorithm'])} steps")
        
        # Demonstrate expert system architecture
        expert_system = self.knowledge_manager.get_expert_system_architecture()
        print(f"\nExpert System Components:")
        print(f"  Knowledge Base: {len(expert_system['knowledge_base'])} components")
        print(f"  Inference Engine: {len(expert_system['inference_engine'])} methods")
        print(f"  Explanation Facility: {len(expert_system['explanation_facility'])} features")
        
        # Demonstrate decision support frameworks
        print("\n" + "-"*40)
        print("DECISION SUPPORT FRAMEWORKS")
        print("-"*40)
        
        frameworks = self.knowledge_manager.get_decision_support_frameworks()
        print("Available Decision Support Frameworks:")
        print(f"  - Herbert Simon Model: {len(frameworks['herbert_simon_model'])} phases")
        print(f"  - Decision Analysis Methods: {len(frameworks['decision_analysis_methods'])} methods")
        print(f"  - Clinical Decision Support: {len(frameworks['clinical_decision_support'])} frameworks")
        print(f"  - Risk Analysis Frameworks: {len(frameworks['risk_analysis_frameworks'])} methods")
        
        # Demonstrate sensitivity analysis
        print("\nSensitivity Analysis Example:")
        alternatives = [
            {'id': 'treatment_a', 'effectiveness': 0.8, 'cost': 0.3, 'safety': 0.9},
            {'id': 'treatment_b', 'effectiveness': 0.7, 'cost': 0.2, 'safety': 0.8},
            {'id': 'treatment_c', 'effectiveness': 0.9, 'cost': 0.5, 'safety': 0.7}
        ]
        criteria = [
            {'name': 'effectiveness', 'maximize': True},
            {'name': 'cost', 'maximize': False},
            {'name': 'safety', 'maximize': True}
        ]
        weights = {'effectiveness': 0.5, 'cost': 0.3, 'safety': 0.2}
        
        sensitivity_result = self.knowledge_manager.perform_sensitivity_analysis(
            alternatives, criteria, weights
        )
        print(f"  Base Recommendation: {sensitivity_result['base_analysis']['recommendation']}")
        print(f"  Most Sensitive Criteria: {', '.join(sensitivity_result['most_sensitive_criteria'])}")
    
    def _demo_integrated_decision_support(self):
        """Demonstrate integrated decision support capabilities"""
        print("\nINTEGRATED DECISION SUPPORT DEMONSTRATION")
        print("-" * 50)
        
        # Simulate a clinical decision support scenario
        print("Simulating clinical decision support scenario...")
        
        # Patient data
        patient_data = {
            'age': 45,
            'bmi': 32.1,
            'systolic_bp': 135,
            'diastolic_bp': 88,
            'family_history': True,
            'hba1c': 6.8
        }
        
        print(f"Patient profile: {patient_data}")
        
        # Step 1: Data preprocessing
        print("\n1. Data preprocessing...")
        features, _ = self.data_manager.preprocess_data('diabetes', 'target')
        patient_features = features.head(1)  # Use first patient as example
        
        # Step 2: Model prediction
        print("2. Model prediction...")
        try:
            prediction = self.model_manager.predict(
                'diabetes_random_forest_regression', 
                patient_features
            )
            print(f"   Predicted diabetes progression: {prediction['predictions'][0]:.2f}")
        except Exception as e:
            print(f"   Error in prediction: {e}")
        
        # Step 3: Knowledge-based recommendations
        print("3. Knowledge-based recommendations...")
        recommendations = self.knowledge_manager.get_clinical_recommendations(patient_data)
        print(f"   Generated {len(recommendations)} recommendations")
        
        # Step 4: Multi-criteria decision analysis
        print("4. Multi-criteria decision analysis...")
        alternatives = [
            {'id': 'lifestyle', 'effectiveness': 0.7, 'cost': 0.2, 'side_effects': 0.1},
            {'id': 'metformin', 'effectiveness': 0.8, 'cost': 0.4, 'side_effects': 0.3},
            {'id': 'insulin', 'effectiveness': 0.9, 'cost': 0.8, 'side_effects': 0.5}
        ]
        
        criteria = [
            {'name': 'effectiveness', 'maximize': True},
            {'name': 'cost', 'maximize': False},
            {'name': 'side_effects', 'maximize': False}
        ]
        
        weights = {'effectiveness': 0.5, 'cost': 0.3, 'side_effects': 0.2}
        
        mcda_result = self.knowledge_manager.multi_criteria_decision_analysis(
            alternatives, criteria, weights
        )
        print(f"   Recommended treatment: {mcda_result['recommendation']}")
        print(f"   Treatment rankings: {mcda_result['ranked_alternatives']}")
    
    def _demo_system_performance(self):
        """Demonstrate system performance and capabilities"""
        print("\nSYSTEM PERFORMANCE SUMMARY")
        print("-" * 50)
        
        # Data management performance
        print("Data Management Performance:")
        total_datasets = len(self.data_manager.datasets)
        total_records = sum([df.shape[0] for df in self.data_manager.datasets.values()])
        print(f"  - Datasets managed: {total_datasets}")
        print(f"  - Total records: {total_records:,}")
        
        # Model management performance
        print("\nModel Management Performance:")
        trained_models = len(self.model_manager.registry.list_models())
        print(f"  - Trained models: {trained_models}")
        
        if hasattr(self.model_manager, 'get_model_performance_summary'):
            try:
                model_summary = self.model_manager.get_model_performance_summary()
                if not model_summary.empty:
                    best_model = model_summary.iloc[0]['Model']
                    best_score = model_summary.iloc[0].get('r2_score', model_summary.iloc[0].get('accuracy', 0))
                    print(f"  - Best performing model: {best_model}")
                    print(f"  - Best score: {best_score:.4f}")
            except Exception as e:
                print(f"  - Error getting model summary: {e}")
        
        # Knowledge management performance
        print("\nKnowledge Management Performance:")
        knowledge_summary = self.knowledge_manager.get_knowledge_summary()
        print(f"  - Clinical rules: {knowledge_summary['clinical_rules']}")
        print(f"  - Clinical guidelines: {knowledge_summary['clinical_guidelines']}")
        print(f"  - Decision trees: {knowledge_summary['decision_trees']}")
        
        # System integration status
        print("\nSystem Integration Status:")
        print("  ✅ Data Management Subsystem: Operational")
        print("  ✅ Model Management Subsystem: Operational")
        print("  ✅ Knowledge-Based Management Subsystem: Operational")
        print("  ✅ User Interface Subsystem: Operational")
        print("  ✅ All subsystems integrated successfully")
    
    def run_streamlit_dashboard(self):
        """Run the Streamlit dashboard"""
        logger.info("Starting Streamlit dashboard...")
        try:
            import streamlit.web.cli as stcli
            import sys
            
            # Set up Streamlit configuration
            os.environ['STREAMLIT_SERVER_PORT'] = '8501'
            os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
            
            # Run Streamlit app
            sys.argv = ['streamlit', 'run', 'healthcare_dss/ui/streamlit_app.py']
            stcli.main()
            
        except Exception as e:
            logger.error(f"Error running Streamlit dashboard: {e}")
            print(f"Error: {e}")
            print("Please install Streamlit: pip install streamlit")
    
    def run_workflow_demo(self):
        """Run workflow demonstration with all analytics modules"""
        logger.info("Starting workflow demonstration...")
        
        print("\n" + "="*80)
        print("HEALTHCARE DSS WORKFLOW DEMONSTRATION")
        print("="*80)
        
        # 1. CRISP-DM Workflow Demo
        self._demo_crisp_dm_workflow()
        
        # 2. Classification Evaluation Demo
        self._demo_classification_evaluation()
        
        # 3. Association Rules Mining Demo
        self._demo_association_rules()
        
        # 4. Clustering Analysis Demo
        self._demo_clustering_analysis()
        
        # 5. Time Series Analysis Demo
        self._demo_time_series_analysis()
        
        # 6. Prescriptive Analytics Demo
        self._demo_prescriptive_analytics()
        
        print("\n" + "="*80)
        print("WORKFLOW DEMONSTRATION COMPLETED!")
        print("="*80)
    
    def run_clinical_demo(self):
        """Run clinical decision support demonstration - MISSING FEATURE from old implementation"""
        logger.info("Starting clinical decision support demonstration...")
        
        print("\n" + "="*80)
        print("CLINICAL DECISION SUPPORT DEMONSTRATION")
        print("="*80)
        
        # Simulate patient assessment
        print("\n1. Patient Assessment Simulation")
        print("-" * 50)
        
        patient_data = {
            'age': 55,
            'bmi': 28.5,
            'systolic_bp': 145,
            'diastolic_bp': 95,
            'family_history': True,
            'hba1c': 7.2,
            'cholesterol': 220,
            'diabetes': True,
            'hypertension': True
        }
        
        print(f"Patient Profile:")
        print(f"  - Age: {patient_data['age']} years")
        print(f"  - BMI: {patient_data['bmi']}")
        print(f"  - Blood Pressure: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")
        print(f"  - HbA1c: {patient_data['hba1c']}%")
        print(f"  - Total Cholesterol: {patient_data['cholesterol']} mg/dL")
        
        # Clinical decision support
        print("\n2. Clinical Decision Support Analysis")
        print("-" * 50)
        
        # Risk assessment
        risk_scores = self._calculate_clinical_risk_scores(patient_data)
        print(f"Risk Assessment:")
        print(f"  - Diabetes Risk: {risk_scores['diabetes_risk']:.1f}%")
        print(f"  - Cardiovascular Risk: {risk_scores['cv_risk']:.1f}%")
        print(f"  - Overall Risk Score: {risk_scores['overall_risk']:.1f}/10")
        
        # Generate recommendations
        recommendations = self._generate_clinical_recommendations(patient_data)
        print(f"\nClinical Recommendations:")
        for priority, recs in recommendations.items():
            if recs:
                print(f"  {priority} Priority:")
                for rec in recs:
                    print(f"    • {rec}")
        
        # Evidence-based guidelines
        print("\n3. Evidence-Based Guidelines")
        print("-" * 50)
        
        guidelines = self._get_relevant_guidelines(patient_data)
        for guideline in guidelines:
            print(f"  - {guideline['title']}")
            print(f"    Source: {guideline['source']}")
            print(f"    Recommendation: {guideline['recommendation']}")
        
        # Drug interaction check
        print("\n4. Medication Safety Check")
        print("-" * 50)
        
        medications = ["Metformin", "ACE Inhibitor", "Statin"]
        interactions = self._check_drug_interactions(medications)
        
        print(f"Current Medications: {', '.join(medications)}")
        if interactions:
            print("⚠️ Potential Drug Interactions:")
            for interaction in interactions:
                print(f"  • {interaction}")
        else:
            print("✅ No significant drug interactions detected")
        
        print("\n" + "="*80)
        print("CLINICAL DECISION SUPPORT DEMONSTRATION COMPLETED!")
        print("="*80)
    
    def _calculate_clinical_risk_scores(self, patient_data):
        """Calculate clinical risk scores"""
        diabetes_risk = 0
        cv_risk = 0
        
        # Diabetes risk calculation
        if patient_data['age'] > 45:
            diabetes_risk += 20
        if patient_data['bmi'] > 25:
            diabetes_risk += 15
        if patient_data['family_history']:
            diabetes_risk += 25
        if patient_data['hba1c'] > 5.7:
            diabetes_risk += 30
        
        # Cardiovascular risk calculation
        if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90:
            cv_risk += 25
        if patient_data['cholesterol'] > 200:
            cv_risk += 20
        if patient_data['family_history']:
            cv_risk += 15
        if patient_data['age'] > 50:
            cv_risk += 20
        if patient_data['diabetes']:
            cv_risk += 30
        
        overall_risk = (diabetes_risk + cv_risk) / 20  # Scale to 0-10
        
        return {
            'diabetes_risk': min(diabetes_risk, 100),
            'cv_risk': min(cv_risk, 100),
            'overall_risk': min(overall_risk, 10)
        }
    
    def _generate_clinical_recommendations(self, patient_data):
        """Generate clinical recommendations"""
        recommendations = {
            'High': [],
            'Medium': [],
            'Low': []
        }
        
        # High priority recommendations
        if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90:
            recommendations['High'].append("Initiate antihypertensive therapy")
        
        if patient_data['hba1c'] > 7.0:
            recommendations['High'].append("Optimize diabetes management")
        
        if patient_data['bmi'] > 30:
            recommendations['High'].append("Refer to weight management program")
        
        # Medium priority recommendations
        if patient_data['cholesterol'] > 200:
            recommendations['Medium'].append("Consider statin therapy")
        
        if patient_data['family_history']:
            recommendations['Medium'].append("Enhanced cardiovascular monitoring")
        
        # Low priority recommendations
        recommendations['Low'].append("Annual comprehensive metabolic panel")
        recommendations['Low'].append("Regular physical activity counseling")
        
        return recommendations
    
    def _get_relevant_guidelines(self, patient_data):
        """Get relevant clinical guidelines"""
        guidelines = []
        
        if patient_data['diabetes'] or patient_data['hba1c'] > 5.7:
            guidelines.append({
                'title': 'ADA Diabetes Management Guidelines 2023',
                'source': 'American Diabetes Association',
                'recommendation': 'HbA1c target <7% for most adults'
            })
        
        if patient_data['systolic_bp'] > 130 or patient_data['diastolic_bp'] > 80:
            guidelines.append({
                'title': 'AHA/ACC Hypertension Guidelines 2017',
                'source': 'American Heart Association',
                'recommendation': 'Blood pressure target <130/80 mmHg'
            })
        
        return guidelines
    
    def _check_drug_interactions(self, medications):
        """Check for drug interactions"""
        interactions = []
        
        interaction_pairs = [
            (["Metformin", "Insulin"], "May increase risk of hypoglycemia"),
            (["ACE Inhibitor", "Potassium"], "Risk of hyperkalemia"),
            (["Warfarin", "Aspirin"], "Increased bleeding risk")
        ]
        
        for med_pair, interaction in interaction_pairs:
            if all(med in medications for med in med_pair):
                interactions.append(f"{' + '.join(med_pair)}: {interaction}")
        
        return interactions
    
    def _demo_kpi_dashboard(self):
        """Demonstrate KPI Dashboard capabilities"""
        print("\nKPI DASHBOARD DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Calculate KPIs
            kpis = self.kpi_dashboard.calculate_healthcare_kpis()
            
            print("Key Performance Indicators:")
            print(f"  - Total Datasets: {kpis.get('system_datasets_loaded', 0)}")
            print(f"  - Total Records: {kpis.get('system_total_records', 0):,}")
            print(f"  - Average Data Quality: {kpis.get('system_avg_data_quality', 0):.1f}%")
            
            if 'diabetes_patients_total' in kpis:
                print(f"\nDiabetes Analysis:")
                print(f"  - Total Patients: {kpis['diabetes_patients_total']:,}")
                print(f"  - High Risk: {kpis['diabetes_high_risk_count']:,} ({kpis['diabetes_high_risk_percentage']:.1f}%)")
                print(f"  - Average Target: {kpis['diabetes_target_mean']:.1f}")
            
            if 'cancer_patients_total' in kpis:
                print(f"\nCancer Analysis:")
                print(f"  - Total Patients: {kpis['cancer_patients_total']:,}")
                print(f"  - Malignancy Rate: {kpis['cancer_malignancy_rate']:.1f}%")
                print(f"  - Malignant Cases: {kpis['cancer_malignant_count']:,}")
            
            if 'expenditure_countries_total' in kpis:
                print(f"\nHealthcare Expenditure:")
                print(f"  - Countries Analyzed: {kpis['expenditure_countries_total']}")
                print(f"  - Global Average: ${kpis['expenditure_global_average']:.0f} per capita")
                print(f"  - Growth Rate: {kpis['expenditure_avg_growth_rate']:.1f}%")
            
            # Generate KPI report
            print("\nGenerating KPI Report...")
            kpi_report = self.kpi_dashboard.generate_kpi_report()
            print(kpi_report)
            
        except Exception as e:
            print(f"  Error in KPI dashboard: {e}")
    
    def _demo_crisp_dm_workflow(self):
        """Demonstrate CRISP-DM workflow"""
        print("\nCRISP-DM WORKFLOW DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Run CRISP-DM workflow on breast cancer dataset
            print("Running CRISP-DM workflow on breast cancer dataset...")
            workflow_results = self.crisp_dm_workflow.execute_full_workflow(
                dataset_name='breast_cancer',
                target_column='target',
                business_objective='Predict breast cancer malignancy for early detection'
            )
            
            print("Workflow completed successfully!")
            print(f"  - Best Model: {workflow_results['evaluation']['best_model']}")
            print(f"  - Best Accuracy: {workflow_results['evaluation']['best_accuracy']:.3f}")
            print(f"  - Models Evaluated: {workflow_results['evaluation']['evaluation_summary']['models_evaluated']}")
            
            # Generate workflow report
            print("\nGenerating CRISP-DM Report...")
            workflow_report = self.crisp_dm_workflow.generate_workflow_report()
            print(workflow_report)
            
        except Exception as e:
            print(f"  Error in CRISP-DM workflow: {e}")
    
    def _demo_classification_evaluation(self):
        """Demonstrate classification evaluation capabilities"""
        print("\nCLASSIFICATION EVALUATION DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Train a model for evaluation
            print("Training Random Forest model for evaluation...")
            from sklearn.ensemble import RandomForestClassifier
            
            # Get breast cancer data
            df = self.data_manager.datasets['breast_cancer']
            X = df.drop(columns=['target', 'target_name'])
            y = df['target']
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            print("Evaluating model...")
            evaluation_results = self.classification_evaluator.evaluate_classification_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name="Random Forest",
                class_names=['Malignant', 'Benign']
            )
            
            print("Evaluation completed!")
            print(f"  - Accuracy: {evaluation_results['basic_metrics']['accuracy']:.3f}")
            print(f"  - Precision: {evaluation_results['basic_metrics']['precision']:.3f}")
            print(f"  - Recall: {evaluation_results['basic_metrics']['recall']:.3f}")
            print(f"  - F1-Score: {evaluation_results['basic_metrics']['f1_score']:.3f}")
            
            # Generate evaluation report
            print("\nGenerating Classification Evaluation Report...")
            eval_report = self.classification_evaluator.generate_evaluation_report("Random Forest")
            print(eval_report)
            
        except Exception as e:
            print(f"  Error in classification evaluation: {e}")
    
    def _demo_association_rules(self):
        """Demonstrate association rules mining capabilities"""
        print("\nASSOCIATION RULES MINING DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Analyze healthcare patterns
            print("Mining association rules from diabetes dataset...")
            diabetes_results = self.association_rules_miner.analyze_healthcare_patterns('diabetes')
            
            print("Association rules analysis completed!")
            print(f"  - Total transactions: {diabetes_results['total_transactions']}")
            print(f"  - Total items: {diabetes_results['total_items']}")
            print(f"  - Frequent itemsets: {diabetes_results['frequent_itemsets_count']}")
            print(f"  - Association rules: {diabetes_results['association_rules_count']}")
            
            # Generate insights
            insights = self.association_rules_miner.get_insights()
            print("\nKey insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight}")
            
            # Generate report
            print("\nGenerating Association Rules Report...")
            report = self.association_rules_miner.generate_report()
            print(report)
            
        except Exception as e:
            print(f"  Error in association rules mining: {e}")
    
    def _demo_clustering_analysis(self):
        """Demonstrate clustering analysis capabilities"""
        print("\nCLUSTERING ANALYSIS DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Prepare data for clustering
            print("Preparing data for clustering analysis...")
            scaled_data, original_data = self.clustering_analyzer.prepare_data_for_clustering('diabetes')
            
            # Find optimal number of clusters
            print("Finding optimal number of clusters...")
            optimal_results = self.clustering_analyzer.find_optimal_clusters(scaled_data)
            optimal_k = optimal_results['optimal_k']
            
            print(f"Optimal number of clusters: {optimal_k}")
            print(f"Best silhouette score: {optimal_results['best_silhouette_score']:.3f}")
            
            # Perform K-means clustering
            print("Performing K-means clustering...")
            kmeans_results = self.clustering_analyzer.perform_kmeans_clustering(scaled_data, optimal_k)
            
            print("Clustering analysis completed!")
            print(f"  - Number of clusters: {kmeans_results['n_clusters']}")
            print(f"  - Silhouette score: {kmeans_results['silhouette_score']:.3f}")
            print(f"  - Calinski-Harabasz score: {kmeans_results['calinski_harabasz_score']:.3f}")
            
            # Analyze patient segments
            segment_analysis = self.clustering_analyzer.analyze_patient_segments(
                'diabetes', kmeans_results['cluster_labels'], original_data
            )
            
            print("\nPatient segment analysis:")
            for cluster_id, size in segment_analysis['cluster_sizes'].items():
                percentage = segment_analysis['cluster_percentages'][cluster_id]
                print(f"  Cluster {cluster_id}: {size} patients ({percentage}%)")
            
            # Generate report
            print("\nGenerating Clustering Analysis Report...")
            report = self.clustering_analyzer.generate_report('diabetes')
            print(report)
            
        except Exception as e:
            print(f"  Error in clustering analysis: {e}")
    
    def _demo_time_series_analysis(self):
        """Demonstrate time series analysis capabilities"""
        print("\nTIME SERIES ANALYSIS DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Prepare time series data
            print("Preparing time series data...")
            self.time_series_analyzer.prepare_time_series_data('healthcare_expenditure')
            
            # Analyze temporal patterns
            print("Analyzing temporal patterns...")
            pattern_results = self.time_series_analyzer.analyze_temporal_patterns('healthcare_expenditure')
            
            print("Time series analysis completed!")
            print(f"  - Total data points: {pattern_results['total_data_points']}")
            print(f"  - Countries analyzed: {pattern_results['countries_analyzed']}")
            print(f"  - Trend direction: {pattern_results['trend_analysis']['direction']}")
            print(f"  - Trend strength: {pattern_results['trend_analysis']['strength']}")
            print(f"  - Volatility level: {pattern_results['volatility_analysis']['volatility_level']}")
            
            # Detect anomalies
            print("Detecting anomalies...")
            anomaly_results = self.time_series_analyzer.detect_anomalies('healthcare_expenditure')
            print(f"  - Total anomalies: {anomaly_results['total_anomalies']}")
            print(f"  - Anomaly rate: {anomaly_results['anomaly_rate']:.1f}%")
            
            # Generate forecasts
            print("Generating forecasts...")
            forecast_results = self.time_series_analyzer.forecast_values('healthcare_expenditure', periods=3)
            print(f"  - Forecast periods: {forecast_results['forecast_periods']}")
            print(f"  - Average model R²: {forecast_results['model_performance']['average_r2']:.3f}")
            
            # Generate report
            print("\nGenerating Time Series Analysis Report...")
            report = self.time_series_analyzer.generate_report('healthcare_expenditure')
            print(report)
            
        except Exception as e:
            print(f"  Error in time series analysis: {e}")
    
    def _demo_prescriptive_analytics(self):
        """Demonstrate prescriptive analytics capabilities"""
        print("\nPRESCRIPTIVE ANALYTICS DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Optimize resource allocation
            print("Optimizing healthcare expenditure allocation...")
            resource_constraints = {
                'total_budget': 1000000,
                'min_allocation': 10000,
                'max_allocation': 200000
            }
            
            allocation_results = self.prescriptive_analyzer.optimize_resource_allocation(
                'healthcare_expenditure', resource_constraints, 'maximize_benefit'
            )
            
            if allocation_results['success']:
                print("Resource allocation optimization completed!")
                print(f"  - Total cost: ${allocation_results['total_cost']:,.2f}")
                print(f"  - Objective value: {allocation_results['objective_value']:,.2f}")
                print(f"  - Countries optimized: {len(allocation_results['allocations'])}")
            else:
                print(f"  Optimization failed: {allocation_results.get('message', 'Unknown error')}")
            
            # Optimize treatment allocation
            print("\nOptimizing diabetes treatment allocation...")
            treatment_constraints = {
                'total_budget': 500000
            }
            
            treatment_results = self.prescriptive_analyzer.optimize_resource_allocation(
                'diabetes', treatment_constraints, 'maximize_benefit'
            )
            
            if treatment_results['success']:
                print("Treatment allocation optimization completed!")
                print(f"  - Total cost: ${treatment_results['total_cost']:,.2f}")
                print(f"  - Total effectiveness: {treatment_results['total_effectiveness']:.2f}")
                print(f"  - Risk categories: {len(treatment_results['allocations'])}")
            else:
                print(f"  Optimization failed: {treatment_results.get('message', 'Unknown error')}")
            
            # Optimize scheduling
            print("\nOptimizing patient scheduling...")
            scheduling_constraints = {
                'daily_hours': 8,
                'max_patients': 15
            }
            
            scheduling_results = self.prescriptive_analyzer.optimize_scheduling(
                'diabetes', scheduling_constraints
            )
            
            if scheduling_results['success']:
                print("Scheduling optimization completed!")
                print(f"  - Total appointments: {scheduling_results['total_appointments']}")
                print(f"  - Total duration: {scheduling_results['total_duration']} minutes")
                print(f"  - Utilization rate: {scheduling_results['utilization_rate']:.1f}%")
            else:
                print(f"  Optimization failed: {scheduling_results.get('message', 'Unknown error')}")
            
            # Generate report
            print("\nGenerating Prescriptive Analytics Report...")
            report = self.prescriptive_analyzer.generate_report('healthcare_expenditure')
            print(report)
            
        except Exception as e:
            print(f"  Error in prescriptive analytics: {e}")
    
    def _demo_crisp_dm_workflow(self):
        """Demonstrate CRISP-DM workflow"""
        print("\nCRISP-DM WORKFLOW DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Run CRISP-DM workflow on breast cancer dataset
            print("Running CRISP-DM workflow on breast cancer dataset...")
            workflow_results = self.crisp_dm_workflow.execute_full_workflow(
                dataset_name='breast_cancer',
                target_column='target',
                business_objective='Predict breast cancer malignancy for early detection'
            )
            
            print("Workflow completed successfully!")
            print(f"  - Best Model: {workflow_results['evaluation']['best_model']}")
            print(f"  - Best Accuracy: {workflow_results['evaluation']['best_accuracy']:.3f}")
            print(f"  - Models Evaluated: {workflow_results['evaluation']['evaluation_summary']['models_evaluated']}")
            
            # Generate workflow report
            print("\nGenerating CRISP-DM Report...")
            workflow_report = self.crisp_dm_workflow.generate_workflow_report()
            print(workflow_report)
            
        except Exception as e:
            print(f"  Error in CRISP-DM workflow: {e}")
    
    def _demo_classification_evaluation(self):
        """Demonstrate classification evaluation capabilities"""
        print("\nCLASSIFICATION EVALUATION DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Train a model for evaluation
            print("Training Random Forest model for evaluation...")
            from sklearn.ensemble import RandomForestClassifier
            
            # Get breast cancer data
            df = self.data_manager.datasets['breast_cancer']
            X = df.drop(columns=['target', 'target_name'])
            y = df['target']
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            print("Evaluating model...")
            evaluation_results = self.classification_evaluator.evaluate_classification_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name="Random Forest",
                class_names=['Malignant', 'Benign']
            )
            
            print("Evaluation completed!")
            print(f"  - Accuracy: {evaluation_results['basic_metrics']['accuracy']:.3f}")
            print(f"  - Precision: {evaluation_results['basic_metrics']['precision']:.3f}")
            print(f"  - Recall: {evaluation_results['basic_metrics']['recall']:.3f}")
            print(f"  - F1-Score: {evaluation_results['basic_metrics']['f1_score']:.3f}")
            
            # Generate evaluation report
            print("\nGenerating Classification Evaluation Report...")
            eval_report = self.classification_evaluator.generate_evaluation_report("Random Forest")
            print(eval_report)
            
        except Exception as e:
            print(f"  Error in classification evaluation: {e}")
    
    def _demo_association_rules(self):
        """Demonstrate association rules mining capabilities"""
        print("\nASSOCIATION RULES MINING DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Analyze healthcare patterns
            print("Mining association rules from diabetes dataset...")
            diabetes_results = self.association_rules_miner.analyze_healthcare_patterns('diabetes')
            
            print("Association rules analysis completed!")
            print(f"  - Total transactions: {diabetes_results['total_transactions']}")
            print(f"  - Total items: {diabetes_results['total_items']}")
            print(f"  - Frequent itemsets: {diabetes_results['frequent_itemsets_count']}")
            print(f"  - Association rules: {diabetes_results['association_rules_count']}")
            
            # Generate insights
            insights = self.association_rules_miner.get_insights()
            print("\nKey insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight}")
            
            # Generate report
            print("\nGenerating Association Rules Report...")
            report = self.association_rules_miner.generate_report()
            print(report)
            
        except Exception as e:
            print(f"  Error in association rules mining: {e}")
    
    def _demo_clustering_analysis(self):
        """Demonstrate clustering analysis capabilities"""
        print("\nCLUSTERING ANALYSIS DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Prepare data for clustering
            print("Preparing data for clustering analysis...")
            scaled_data, original_data = self.clustering_analyzer.prepare_data_for_clustering('diabetes')
            
            # Find optimal number of clusters
            print("Finding optimal number of clusters...")
            optimal_results = self.clustering_analyzer.find_optimal_clusters(scaled_data)
            optimal_k = optimal_results['optimal_k']
            
            print(f"Optimal number of clusters: {optimal_k}")
            print(f"Best silhouette score: {optimal_results['best_silhouette_score']:.3f}")
            
            # Perform K-means clustering
            print("Performing K-means clustering...")
            kmeans_results = self.clustering_analyzer.perform_kmeans_clustering(scaled_data, optimal_k)
            
            print("Clustering analysis completed!")
            print(f"  - Number of clusters: {kmeans_results['n_clusters']}")
            print(f"  - Silhouette score: {kmeans_results['silhouette_score']:.3f}")
            print(f"  - Calinski-Harabasz score: {kmeans_results['calinski_harabasz_score']:.3f}")
            
            # Analyze patient segments
            segment_analysis = self.clustering_analyzer.analyze_patient_segments(
                'diabetes', kmeans_results['cluster_labels'], original_data
            )
            
            print("\nPatient segment analysis:")
            for cluster_id, size in segment_analysis['cluster_sizes'].items():
                percentage = segment_analysis['cluster_percentages'][cluster_id]
                print(f"  Cluster {cluster_id}: {size} patients ({percentage}%)")
            
            # Generate report
            print("\nGenerating Clustering Analysis Report...")
            report = self.clustering_analyzer.generate_report('diabetes')
            print(report)
            
        except Exception as e:
            print(f"  Error in clustering analysis: {e}")
    
    def _demo_time_series_analysis(self):
        """Demonstrate time series analysis capabilities"""
        print("\nTIME SERIES ANALYSIS DEMONSTRATION")
        print("-" * 50)
        
        try:
            # Prepare time series data
            print("Preparing time series data...")
            self.time_series_analyzer.prepare_time_series_data('healthcare_expenditure')
            
            # Analyze temporal patterns
            print("Analyzing temporal patterns...")
            pattern_results = self.time_series_analyzer.analyze_temporal_patterns('healthcare_expenditure')
            
            print("Time series analysis completed!")
            print(f"  - Total data points: {pattern_results['total_data_points']}")
            print(f"  - Countries analyzed: {pattern_results['countries_analyzed']}")
            print(f"  - Trend direction: {pattern_results['trend_analysis']['direction']}")
            print(f"  - Trend strength: {pattern_results['trend_analysis']['strength']}")
            print(f"  - Volatility level: {pattern_results['volatility_analysis']['volatility_level']}")
            
            # Detect anomalies
            print("Detecting anomalies...")
            anomaly_results = self.time_series_analyzer.detect_anomalies('healthcare_expenditure')
            print(f"  - Total anomalies: {anomaly_results['total_anomalies']}")
            print(f"  - Anomaly rate: {anomaly_results['anomaly_rate']:.1f}%")
            
            # Generate forecasts
            print("Generating forecasts...")
            forecast_results = self.time_series_analyzer.forecast_values('healthcare_expenditure', periods=3)
            print(f"  - Forecast periods: {forecast_results['forecast_periods']}")
            print(f"  - Average model R²: {forecast_results['model_performance']['average_r2']:.3f}")
            
            # Generate report
            print("\nGenerating Time Series Analysis Report...")
            report = self.time_series_analyzer.generate_report('healthcare_expenditure')
            print(report)
            
        except Exception as e:
            print(f"  Error in time series analysis: {e}")
    
    def enable_debug_mode(self):
        """Enable debug mode for comprehensive system monitoring"""
        self.debug_manager.enable_debug()
        debug_write("Debug mode enabled from main application", "SYSTEM")
        print("🔍 Debug mode enabled - comprehensive system monitoring active")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_manager.disable_debug()
        debug_write("Debug mode disabled from main application", "SYSTEM")
        print("🔍 Debug mode disabled")
    
    def get_debug_summary(self):
        """Get debug system summary"""
        summary = self.debug_manager.get_debug_summary()
        debug_write("Debug summary requested", "SYSTEM", summary)
        return summary
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up system resources...")
        debug_write("Starting system cleanup", "SYSTEM")
        
        if self.data_manager:
            self.data_manager.close_connection()
            debug_write("Data manager connection closed", "SYSTEM")
        
        if self.knowledge_manager:
            self.knowledge_manager.close_connection()
            debug_write("Knowledge manager connection closed", "SYSTEM")
        
        # Export debug logs before cleanup
        try:
            debug_summary = self.get_debug_summary()
            debug_write("System cleanup completed", "SYSTEM", debug_summary)
        except Exception as e:
            debug_write(f"Error during cleanup: {e}", "ERROR")
        
        logger.info("Cleanup completed")
    
    def run_clinical_demo(self):
        """Run clinical decision support demonstration - MISSING FEATURE from old implementation"""
        logger.info("Starting clinical decision support demonstration...")
        
        print("\n" + "="*80)
        print("CLINICAL DECISION SUPPORT DEMONSTRATION")
        print("="*80)
        
        # Simulate patient assessment
        print("\n1. Patient Assessment Simulation")
        print("-" * 50)
        
        patient_data = {
            'age': 55,
            'bmi': 28.5,
            'systolic_bp': 145,
            'diastolic_bp': 95,
            'family_history': True,
            'hba1c': 7.2,
            'cholesterol': 220,
            'diabetes': True,
            'hypertension': True
        }
        
        print(f"Patient Profile:")
        print(f"  - Age: {patient_data['age']} years")
        print(f"  - BMI: {patient_data['bmi']}")
        print(f"  - Blood Pressure: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")
        print(f"  - HbA1c: {patient_data['hba1c']}%")
        print(f"  - Total Cholesterol: {patient_data['cholesterol']} mg/dL")
        
        # Clinical decision support
        print("\n2. Clinical Decision Support Analysis")
        print("-" * 50)
        
        # Risk assessment
        risk_scores = self._calculate_clinical_risk_scores(patient_data)
        print(f"Risk Assessment:")
        print(f"  - Diabetes Risk: {risk_scores['diabetes_risk']:.1f}%")
        print(f"  - Cardiovascular Risk: {risk_scores['cv_risk']:.1f}%")
        print(f"  - Overall Risk Score: {risk_scores['overall_risk']:.1f}/10")
        
        # Generate recommendations
        recommendations = self._generate_clinical_recommendations(patient_data)
        print(f"\nClinical Recommendations:")
        for priority, recs in recommendations.items():
            if recs:
                print(f"  {priority} Priority:")
                for rec in recs:
                    print(f"    • {rec}")
        
        # Evidence-based guidelines
        print("\n3. Evidence-Based Guidelines")
        print("-" * 50)
        
        guidelines = self._get_relevant_guidelines(patient_data)
        for guideline in guidelines:
            print(f"  - {guideline['title']}")
            print(f"    Source: {guideline['source']}")
            print(f"    Recommendation: {guideline['recommendation']}")
        
        # Drug interaction check
        print("\n4. Medication Safety Check")
        print("-" * 50)
        
        medications = ["Metformin", "ACE Inhibitor", "Statin"]
        interactions = self._check_drug_interactions(medications)
        
        print(f"Current Medications: {', '.join(medications)}")
        if interactions:
            print("⚠️ Potential Drug Interactions:")
            for interaction in interactions:
                print(f"  • {interaction}")
        else:
            print("✅ No significant drug interactions detected")
        
        print("\n" + "="*80)
        print("CLINICAL DECISION SUPPORT DEMONSTRATION COMPLETED!")
        print("="*80)
    
    def _calculate_clinical_risk_scores(self, patient_data):
        """Calculate clinical risk scores"""
        diabetes_risk = 0
        cv_risk = 0
        
        # Diabetes risk calculation
        if patient_data['age'] > 45:
            diabetes_risk += 20
        if patient_data['bmi'] > 25:
            diabetes_risk += 15
        if patient_data['family_history']:
            diabetes_risk += 25
        if patient_data['hba1c'] > 5.7:
            diabetes_risk += 30
        
        # Cardiovascular risk calculation
        if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90:
            cv_risk += 25
        if patient_data['cholesterol'] > 200:
            cv_risk += 20
        if patient_data['family_history']:
            cv_risk += 15
        if patient_data['age'] > 50:
            cv_risk += 20
        if patient_data['diabetes']:
            cv_risk += 30
        
        overall_risk = (diabetes_risk + cv_risk) / 20  # Scale to 0-10
        
        return {
            'diabetes_risk': min(diabetes_risk, 100),
            'cv_risk': min(cv_risk, 100),
            'overall_risk': min(overall_risk, 10)
        }
    
    def _generate_clinical_recommendations(self, patient_data):
        """Generate clinical recommendations"""
        recommendations = {
            'High': [],
            'Medium': [],
            'Low': []
        }
        
        # High priority recommendations
        if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90:
            recommendations['High'].append("Initiate antihypertensive therapy")
        
        if patient_data['hba1c'] > 7.0:
            recommendations['High'].append("Optimize diabetes management")
        
        if patient_data['bmi'] > 30:
            recommendations['High'].append("Refer to weight management program")
        
        # Medium priority recommendations
        if patient_data['cholesterol'] > 200:
            recommendations['Medium'].append("Consider statin therapy")
        
        if patient_data['family_history']:
            recommendations['Medium'].append("Enhanced cardiovascular monitoring")
        
        # Low priority recommendations
        recommendations['Low'].append("Annual comprehensive metabolic panel")
        recommendations['Low'].append("Regular physical activity counseling")
        
        return recommendations
    
    def _get_relevant_guidelines(self, patient_data):
        """Get relevant clinical guidelines"""
        guidelines = []
        
        if patient_data['diabetes'] or patient_data['hba1c'] > 5.7:
            guidelines.append({
                'title': 'ADA Diabetes Management Guidelines 2023',
                'source': 'American Diabetes Association',
                'recommendation': 'HbA1c target <7% for most adults'
            })
        
        if patient_data['systolic_bp'] > 130 or patient_data['diastolic_bp'] > 80:
            guidelines.append({
                'title': 'AHA/ACC Hypertension Guidelines 2017',
                'source': 'American Heart Association',
                'recommendation': 'Blood pressure target <130/80 mmHg'
            })
        
        return guidelines
    
    def _check_drug_interactions(self, medications):
        """Check for drug interactions"""
        interactions = []
        
        interaction_pairs = [
            (["Metformin", "Insulin"], "May increase risk of hypoglycemia"),
            (["ACE Inhibitor", "Potassium"], "Risk of hyperkalemia"),
            (["Warfarin", "Aspirin"], "Increased bleeding risk")
        ]
        
        for med_pair, interaction in interaction_pairs:
            if all(med in medications for med in med_pair):
                interactions.append(f"{' + '.join(med_pair)}: {interaction}")
        
        return interactions


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Healthcare Decision Support System')
    parser.add_argument('--mode', choices=['demo', 'methodology', 'dashboard', 'workflow', 'clinical'], default='demo',
                       help='Run mode: demo, methodology, dashboard, workflow, or clinical')
    parser.add_argument('--data-dir', default='datasets',
                       help='Directory containing healthcare datasets')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for comprehensive system monitoring')
    parser.add_argument('--debug-summary', action='store_true',
                       help='Show debug system summary and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize Healthcare DSS
        healthcare_dss = HealthcareDSS(data_dir=args.data_dir)
        
        # Enable debug mode if requested
        if args.debug:
            healthcare_dss.enable_debug_mode()
        
        # Show debug summary if requested
        if args.debug_summary:
            summary = healthcare_dss.get_debug_summary()
            print("\n" + "="*60)
            print("DEBUG SYSTEM SUMMARY")
            print("="*60)
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("="*60)
            return
        
        if args.mode == 'demo':
            # Run comprehensive demonstration
            healthcare_dss.run_comprehensive_demo()
        elif args.mode == 'methodology':
            # Run DSS methodology demonstration
            healthcare_dss.run_dss_methodology_demo()
        elif args.mode == 'dashboard':
            # Run Streamlit dashboard
            healthcare_dss.run_streamlit_dashboard()
        elif args.mode == 'workflow':
            # Run workflow demonstration
            healthcare_dss.run_workflow_demo()
        elif args.mode == 'clinical':
            # Run clinical decision support demonstration
            healthcare_dss.run_clinical_demo()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        debug_write("Application interrupted by user", "SYSTEM")
    except Exception as e:
        logger.error(f"Application error: {e}")
        debug_write(f"Application error: {e}", "ERROR", {"error": str(e), "traceback": traceback.format_exc()})
        raise
    finally:
        # Cleanup
        if 'healthcare_dss' in locals():
            healthcare_dss.cleanup()


    def _calculate_clinical_risk_scores_configurable(self, patient_data, clinical_config):
        """Calculate clinical risk scores using configurable parameters"""
        diabetes_risk = 0
        cv_risk = 0
        
        # Get configurable risk factors
        diabetes_factors = clinical_config['risk_calculation']['diabetes_risk_factors']
        cv_factors = clinical_config['risk_calculation']['cv_risk_factors']
        
        # Diabetes risk calculation using configurable thresholds and weights
        if patient_data['age'] > diabetes_factors['age_threshold']:
            diabetes_risk += diabetes_factors['age_weight']
        if patient_data['bmi'] > diabetes_factors['bmi_threshold']:
            diabetes_risk += diabetes_factors['bmi_weight']
        if patient_data['family_history']:
            diabetes_risk += diabetes_factors['family_history_weight']
        if patient_data['hba1c'] > diabetes_factors['hba1c_threshold']:
            diabetes_risk += diabetes_factors['hba1c_weight']
        
        # Cardiovascular risk calculation using configurable thresholds and weights
        if patient_data['systolic_bp'] > cv_factors['bp_systolic_threshold'] or patient_data['diastolic_bp'] > cv_factors['bp_diastolic_threshold']:
            cv_risk += cv_factors['bp_weight']
        if patient_data['cholesterol'] > cv_factors['cholesterol_threshold']:
            cv_risk += cv_factors['cholesterol_weight']
        if patient_data['family_history']:
            cv_risk += cv_factors['family_history_weight']
        if patient_data['age'] > cv_factors['age_threshold']:
            cv_risk += cv_factors['age_weight']
        if patient_data['diabetes']:
            cv_risk += cv_factors['diabetes_weight']
        
        overall_risk = (diabetes_risk + cv_risk) / 20  # Scale to 0-10
        
        return {
            'diabetes_risk': min(diabetes_risk, 100),
            'cv_risk': min(cv_risk, 100),
            'overall_risk': min(overall_risk, 10)
        }
    
    def _generate_clinical_recommendations_configurable(self, patient_data, clinical_config):
        """Generate clinical recommendations using configurable rules"""
        recommendations = {
            'High': [],
            'Medium': [],
            'Low': []
        }
        
        # Get configurable recommendation rules
        rules = clinical_config['recommendation_rules']
        
        # Apply high priority rules
        for rule_name, rule in rules['high_priority'].items():
            if rule['condition'](patient_data):
                recommendations['High'].append(rule['recommendation'])
        
        # Apply medium priority rules
        for rule_name, rule in rules['medium_priority'].items():
            if rule['condition'](patient_data):
                recommendations['Medium'].append(rule['recommendation'])
        
        # Apply low priority rules
        for rule_name, rule in rules['low_priority'].items():
            if rule['condition'](patient_data):
                recommendations['Low'].append(rule['recommendation'])
        
        return recommendations
    
    def _get_relevant_guidelines_configurable(self, patient_data, clinical_config):
        """Get relevant clinical guidelines using configurable guidelines"""
        guidelines = []
        
        # Get configurable clinical guidelines
        guideline_configs = clinical_config['clinical_guidelines']
        
        for guideline_name, guideline in guideline_configs.items():
            if guideline['condition'](patient_data):
                guidelines.append({
                    'title': guideline['title'],
                    'source': guideline['source'],
                    'recommendation': guideline['recommendation']
                })
        
        return guidelines
    
    def _check_drug_interactions_configurable(self, medications, clinical_config):
        """Check for drug interactions using configurable interaction rules"""
        interactions = []
        
        # Get configurable drug interactions
        interaction_rules = clinical_config['drug_interactions']
        
        for interaction_rule in interaction_rules:
            if all(med in medications for med in interaction_rule['medications']):
                interactions.append(f"{' + '.join(interaction_rule['medications'])}: {interaction_rule['interaction']}")
        
        return interactions


if __name__ == "__main__":
    main()
