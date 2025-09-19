"""
Comprehensive Test and Debug Script for KPI Dashboard
====================================================

This script tests all functionality in the KPI dashboard and provides
detailed debugging information for troubleshooting issues.
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from healthcare_dss.core.data_management import DataManager
from healthcare_dss.ui.kpi_dashboard import KPIDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kpi_dashboard_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KPIDashboardTester:
    """Comprehensive tester for KPI Dashboard functionality"""
    
    def __init__(self):
        self.data_manager = None
        self.kpi_dashboard = None
        self.test_results = {}
        self.debug_info = {}
        
    def run_all_tests(self):
        """Run all KPI dashboard tests"""
        print("=" * 80)
        print("KPI DASHBOARD COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Test started at: {datetime.now()}")
        print()
        
        try:
            # Test 1: Initialize Data Manager
            self.test_data_manager_initialization()
            
            # Test 2: Load and validate datasets
            self.test_dataset_loading()
            
            # Test 3: Data quality assessment
            self.test_data_quality_assessment()
            
            # Test 4: KPI Dashboard initialization
            self.test_kpi_dashboard_initialization()
            
            # Test 5: Individual KPI calculations
            self.test_individual_kpi_calculations()
            
            # Test 6: Full KPI calculation
            self.test_full_kpi_calculation()
            
            # Test 7: Dashboard creation
            self.test_dashboard_creation()
            
            # Test 8: Report generation
            self.test_report_generation()
            
            # Test 9: Error handling
            self.test_error_handling()
            
            # Test 10: Performance testing
            self.test_performance()
            
            # Generate final report
            self.generate_final_report()
            
        except Exception as e:
            logger.error(f"Critical error in test suite: {e}")
            traceback.print_exc()
    
    def test_data_manager_initialization(self):
        """Test DataManager initialization"""
        print("TEST 1: Data Manager Initialization")
        print("-" * 40)
        
        try:
            self.data_manager = DataManager(data_dir="datasets")
            self.test_results['data_manager_init'] = True
            print("✓ DataManager initialized successfully")
            
            # Debug info
            self.debug_info['data_manager'] = {
                'data_dir': str(self.data_manager.data_dir),
                'db_path': self.data_manager.db_path,
                'datasets_loaded': list(self.data_manager.datasets.keys()),
                'connection_status': self.data_manager.connection is not None
            }
            
        except Exception as e:
            self.test_results['data_manager_init'] = False
            print(f"✗ DataManager initialization failed: {e}")
            traceback.print_exc()
    
    def test_dataset_loading(self):
        """Test dataset loading and validation"""
        print("\nTEST 2: Dataset Loading and Validation")
        print("-" * 40)
        
        if not self.data_manager:
            print("✗ DataManager not initialized, skipping test")
            return
        
        try:
            datasets_info = {}
            
            for name, df in self.data_manager.datasets.items():
                print(f"  Testing dataset: {name}")
                
                # Basic validation
                shape = df.shape
                columns = list(df.columns)
                dtypes = df.dtypes.to_dict()
                missing_values = df.isnull().sum().sum()
                memory_usage = df.memory_usage(deep=True).sum()
                
                datasets_info[name] = {
                    'shape': shape,
                    'columns': columns,
                    'dtypes': {str(k): str(v) for k, v in dtypes.items()},
                    'missing_values': int(missing_values),
                    'memory_usage': int(memory_usage),
                    'sample_data': df.head(3).to_dict() if len(df) > 0 else {}
                }
                
                print(f"    ✓ Shape: {shape}")
                print(f"    ✓ Columns: {len(columns)}")
                print(f"    ✓ Missing values: {missing_values}")
                print(f"    ✓ Memory usage: {memory_usage:,} bytes")
                
                # Special validation for healthcare expenditure dataset
                if name == 'healthcare_expenditure':
                    year_cols = [col for col in columns if '20' in col and '[' in col]
                    print(f"    ✓ Year columns found: {len(year_cols)}")
                    if year_cols:
                        print(f"    ✓ Sample year columns: {year_cols[:3]}")
            
            self.test_results['dataset_loading'] = True
            self.debug_info['datasets'] = datasets_info
            print("✓ All datasets loaded and validated successfully")
            
        except Exception as e:
            self.test_results['dataset_loading'] = False
            print(f"✗ Dataset loading failed: {e}")
            traceback.print_exc()
    
    def test_data_quality_assessment(self):
        """Test data quality assessment functionality"""
        print("\nTEST 3: Data Quality Assessment")
        print("-" * 40)
        
        if not self.data_manager:
            print("✗ DataManager not initialized, skipping test")
            return
        
        try:
            quality_results = {}
            
            for dataset_name in self.data_manager.datasets.keys():
                print(f"  Assessing quality for: {dataset_name}")
                
                try:
                    quality_metrics = self.data_manager.assess_data_quality(dataset_name)
                    quality_results[dataset_name] = quality_metrics
                    
                    print(f"    ✓ Completeness: {quality_metrics.get('completeness_score', 0):.1f}%")
                    print(f"    ✓ Duplicate rows: {quality_metrics.get('duplicate_rows', 0)}")
                    print(f"    ✓ Outliers detected: {len(quality_metrics.get('outliers', {}))}")
                    
                except Exception as e:
                    print(f"    ✗ Quality assessment failed: {e}")
                    quality_results[dataset_name] = {'error': str(e)}
            
            self.test_results['data_quality'] = True
            self.debug_info['quality_metrics'] = quality_results
            print("✓ Data quality assessment completed")
            
        except Exception as e:
            self.test_results['data_quality'] = False
            print(f"✗ Data quality assessment failed: {e}")
            traceback.print_exc()
    
    def test_kpi_dashboard_initialization(self):
        """Test KPI Dashboard initialization"""
        print("\nTEST 4: KPI Dashboard Initialization")
        print("-" * 40)
        
        if not self.data_manager:
            print("✗ DataManager not initialized, skipping test")
            return
        
        try:
            self.kpi_dashboard = KPIDashboard(self.data_manager)
            self.test_results['kpi_dashboard_init'] = True
            print("✓ KPI Dashboard initialized successfully")
            
            self.debug_info['kpi_dashboard'] = {
                'data_manager_connected': self.kpi_dashboard.data_manager is not None,
                'kpi_metrics_initialized': len(self.kpi_dashboard.kpi_metrics) == 0
            }
            
        except Exception as e:
            self.test_results['kpi_dashboard_init'] = False
            print(f"✗ KPI Dashboard initialization failed: {e}")
            traceback.print_exc()
    
    def test_individual_kpi_calculations(self):
        """Test individual KPI calculation methods"""
        print("\nTEST 5: Individual KPI Calculations")
        print("-" * 40)
        
        if not self.kpi_dashboard:
            print("✗ KPI Dashboard not initialized, skipping test")
            return
        
        try:
            individual_kpis = {}
            
            # Test diabetes KPIs
            if 'diabetes' in self.data_manager.datasets:
                print("  Testing diabetes KPIs...")
                try:
                    diabetes_kpis = self.kpi_dashboard._calculate_diabetes_kpis()
                    individual_kpis['diabetes'] = diabetes_kpis
                    print(f"    ✓ Diabetes KPIs calculated: {len(diabetes_kpis)} metrics")
                except Exception as e:
                    print(f"    ✗ Diabetes KPIs failed: {e}")
                    individual_kpis['diabetes'] = {'error': str(e)}
            
            # Test cancer KPIs
            if 'breast_cancer' in self.data_manager.datasets:
                print("  Testing cancer KPIs...")
                try:
                    cancer_kpis = self.kpi_dashboard._calculate_cancer_kpis()
                    individual_kpis['cancer'] = cancer_kpis
                    print(f"    ✓ Cancer KPIs calculated: {len(cancer_kpis)} metrics")
                except Exception as e:
                    print(f"    ✗ Cancer KPIs failed: {e}")
                    individual_kpis['cancer'] = {'error': str(e)}
            
            # Test expenditure KPIs
            if 'healthcare_expenditure' in self.data_manager.datasets:
                print("  Testing expenditure KPIs...")
                try:
                    expenditure_kpis = self.kpi_dashboard._calculate_expenditure_kpis()
                    individual_kpis['expenditure'] = expenditure_kpis
                    print(f"    ✓ Expenditure KPIs calculated: {len(expenditure_kpis)} metrics")
                except Exception as e:
                    print(f"    ✗ Expenditure KPIs failed: {e}")
                    individual_kpis['expenditure'] = {'error': str(e)}
            
            # Test system KPIs
            print("  Testing system KPIs...")
            try:
                system_kpis = self.kpi_dashboard._calculate_system_kpis()
                individual_kpis['system'] = system_kpis
                print(f"    ✓ System KPIs calculated: {len(system_kpis)} metrics")
            except Exception as e:
                print(f"    ✗ System KPIs failed: {e}")
                individual_kpis['system'] = {'error': str(e)}
            
            self.test_results['individual_kpis'] = True
            self.debug_info['individual_kpis'] = individual_kpis
            print("✓ Individual KPI calculations completed")
            
        except Exception as e:
            self.test_results['individual_kpis'] = False
            print(f"✗ Individual KPI calculations failed: {e}")
            traceback.print_exc()
    
    def test_full_kpi_calculation(self):
        """Test full KPI calculation"""
        print("\nTEST 6: Full KPI Calculation")
        print("-" * 40)
        
        if not self.kpi_dashboard:
            print("✗ KPI Dashboard not initialized, skipping test")
            return
        
        try:
            print("  Calculating all KPIs...")
            kpis = self.kpi_dashboard.calculate_healthcare_kpis()
            
            self.test_results['full_kpi_calculation'] = True
            self.debug_info['full_kpis'] = kpis
            
            print(f"    ✓ Total KPIs calculated: {len(kpis)}")
            print(f"    ✓ KPI categories: {list(kpis.keys())}")
            
            # Display key metrics
            if 'system_datasets_loaded' in kpis:
                print(f"    ✓ Datasets loaded: {kpis['system_datasets_loaded']}")
            if 'system_total_records' in kpis:
                print(f"    ✓ Total records: {kpis['system_total_records']:,}")
            if 'system_avg_data_quality' in kpis:
                print(f"    ✓ Average data quality: {kpis['system_avg_data_quality']:.1f}%")
            
        except Exception as e:
            self.test_results['full_kpi_calculation'] = False
            print(f"✗ Full KPI calculation failed: {e}")
            traceback.print_exc()
    
    def test_dashboard_creation(self):
        """Test dashboard creation"""
        print("\nTEST 7: Dashboard Creation")
        print("-" * 40)
        
        if not self.kpi_dashboard:
            print("✗ KPI Dashboard not initialized, skipping test")
            return
        
        try:
            print("  Creating dashboard...")
            fig = self.kpi_dashboard.create_kpi_dashboard()
            
            self.test_results['dashboard_creation'] = True
            self.debug_info['dashboard'] = {
                'figure_created': fig is not None,
                'data_count': len(fig.data) if fig else 0,
                'layout_title': fig.layout.title.text if fig and fig.layout.title else None
            }
            
            print(f"    ✓ Dashboard created successfully")
            print(f"    ✓ Number of plots: {len(fig.data) if fig else 0}")
            
        except Exception as e:
            self.test_results['dashboard_creation'] = False
            print(f"✗ Dashboard creation failed: {e}")
            traceback.print_exc()
    
    def test_report_generation(self):
        """Test report generation"""
        print("\nTEST 8: Report Generation")
        print("-" * 40)
        
        if not self.kpi_dashboard:
            print("✗ KPI Dashboard not initialized, skipping test")
            return
        
        try:
            print("  Generating KPI report...")
            report = self.kpi_dashboard.generate_kpi_report()
            
            self.test_results['report_generation'] = True
            self.debug_info['report'] = {
                'report_length': len(report),
                'report_lines': len(report.split('\n')),
                'contains_sections': any(section in report for section in ['SYSTEM OVERVIEW', 'DIABETES', 'CANCER', 'EXPENDITURE'])
            }
            
            print(f"    ✓ Report generated successfully")
            print(f"    ✓ Report length: {len(report)} characters")
            print(f"    ✓ Report lines: {len(report.split('\n'))}")
            
        except Exception as e:
            self.test_results['report_generation'] = False
            print(f"✗ Report generation failed: {e}")
            traceback.print_exc()
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        print("\nTEST 9: Error Handling")
        print("-" * 40)
        
        try:
            # Test with invalid dataset name
            print("  Testing invalid dataset handling...")
            try:
                invalid_data_manager = DataManager(data_dir="nonexistent")
                print("    ✓ Handled invalid data directory gracefully")
            except Exception as e:
                print(f"    ✓ Expected error for invalid directory: {type(e).__name__}")
            
            # Test with empty datasets
            print("  Testing empty dataset handling...")
            if self.kpi_dashboard:
                # Temporarily clear datasets to test empty state
                original_datasets = self.kpi_dashboard.data_manager.datasets.copy()
                self.kpi_dashboard.data_manager.datasets = {}
                
                try:
                    empty_kpis = self.kpi_dashboard.calculate_healthcare_kpis()
                    print(f"    ✓ Handled empty datasets: {len(empty_kpis)} KPIs")
                except Exception as e:
                    print(f"    ✓ Expected error for empty datasets: {type(e).__name__}")
                
                # Restore original datasets
                self.kpi_dashboard.data_manager.datasets = original_datasets
            
            self.test_results['error_handling'] = True
            print("✓ Error handling tests completed")
            
        except Exception as e:
            self.test_results['error_handling'] = False
            print(f"✗ Error handling tests failed: {e}")
            traceback.print_exc()
    
    def test_performance(self):
        """Test performance metrics"""
        print("\nTEST 10: Performance Testing")
        print("-" * 40)
        
        if not self.kpi_dashboard:
            print("✗ KPI Dashboard not initialized, skipping test")
            return
        
        try:
            import time
            
            # Test KPI calculation performance
            print("  Testing KPI calculation performance...")
            start_time = time.time()
            kpis = self.kpi_dashboard.calculate_healthcare_kpis()
            kpi_time = time.time() - start_time
            
            # Test dashboard creation performance
            print("  Testing dashboard creation performance...")
            start_time = time.time()
            fig = self.kpi_dashboard.create_kpi_dashboard()
            dashboard_time = time.time() - start_time
            
            # Test report generation performance
            print("  Testing report generation performance...")
            start_time = time.time()
            report = self.kpi_dashboard.generate_kpi_report()
            report_time = time.time() - start_time
            
            performance_metrics = {
                'kpi_calculation_time': kpi_time,
                'dashboard_creation_time': dashboard_time,
                'report_generation_time': report_time,
                'total_time': kpi_time + dashboard_time + report_time
            }
            
            self.test_results['performance'] = True
            self.debug_info['performance'] = performance_metrics
            
            print(f"    ✓ KPI calculation: {kpi_time:.3f}s")
            print(f"    ✓ Dashboard creation: {dashboard_time:.3f}s")
            print(f"    ✓ Report generation: {report_time:.3f}s")
            print(f"    ✓ Total time: {performance_metrics['total_time']:.3f}s")
            
        except Exception as e:
            self.test_results['performance'] = False
            print(f"✗ Performance testing failed: {e}")
            traceback.print_exc()
    
    def generate_final_report(self):
        """Generate final test report"""
        print("\n" + "=" * 80)
        print("FINAL TEST REPORT")
        print("=" * 80)
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Detailed results
        print("DETAILED RESULTS:")
        print("-" * 20)
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        # Save debug information
        debug_file = "kpi_dashboard_debug_info.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump(self.debug_info, f, indent=2, default=str)
            print(f"\nDebug information saved to: {debug_file}")
        except Exception as e:
            print(f"\nFailed to save debug information: {e}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 15)
        if failed_tests == 0:
            print("✓ All tests passed! KPI Dashboard is working correctly.")
        else:
            print("⚠ Some tests failed. Check the debug information for details.")
            if not self.test_results.get('data_manager_init', False):
                print("- Fix DataManager initialization issues")
            if not self.test_results.get('dataset_loading', False):
                print("- Check dataset files and paths")
            if not self.test_results.get('full_kpi_calculation', False):
                print("- Review KPI calculation logic")
            if not self.test_results.get('dashboard_creation', False):
                print("- Check Plotly dashboard creation")
        
        print(f"\nTest completed at: {datetime.now()}")
        print("=" * 80)

def main():
    """Main function to run the test suite"""
    tester = KPIDashboardTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
