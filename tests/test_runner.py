"""
Comprehensive Test Runner for Healthcare DSS
=============================================

This module provides a unified test runner that consolidates all testing functionality
including comprehensive tests, binning-specific tests, and basic functionality tests.
"""

import unittest
import sys
import os
import time
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import traceback

# Suppress specific warnings for cleaner test output
warnings.filterwarnings("ignore", message="Bins whose width are too small")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthcareDSSTestRunner:
    """Comprehensive test runner for Healthcare DSS"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def discover_tests(self, category: Optional[str] = None) -> List[unittest.TestSuite]:
        """Discover all test suites in the tests directory"""
        test_suites = []
        
        # Define test directories and their patterns
        test_directories = [
            ('tests/core', 'test_*.py'),
            ('tests/analytics', 'test_*.py'),
            ('tests/ui', 'test_*.py'),
            ('tests/utils', 'test_*.py'),
            ('tests/integration', 'test_*.py'),
            ('tests/modules', 'test_*.py'),
            ('tests/edge_cases', 'test_*.py'),
            ('tests', 'test_*.py')  # Root level tests
        ]
        
        # Filter by category if specified
        if category:
            test_directories = [(f'tests/{category}', 'test_*.py')]
        
        for test_dir, pattern in test_directories:
            if os.path.exists(test_dir):
                loader = unittest.TestLoader()
                suite = loader.discover(test_dir, pattern=pattern)
                if suite.countTestCases() > 0:
                    test_suites.append(suite)
                    logger.info(f"Discovered {suite.countTestCases()} tests in {test_dir}")
        
        return test_suites
    
    def run_test_suite(self, suite: unittest.TestSuite, suite_name: str) -> Dict[str, Any]:
        """Run a single test suite and return results"""
        logger.info(f"Running test suite: {suite_name}")
        
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        end_time = time.time()
        
        suite_results = {
            'suite_name': suite_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
            'execution_time': end_time - start_time,
            'failures_details': [{'test': str(f[0]), 'error': str(f[1])} for f in result.failures],
            'errors_details': [{'test': str(e[0]), 'error': str(e[1])} for e in result.errors]
        }
        
        return suite_results
    
    def run_binning_tests(self) -> Dict[str, Any]:
        """Run binning-specific tests"""
        print("Running Binning System Tests")
        print("-" * 40)
        
        # Import binning test classes
        try:
            from tests.utils.test_utils_modules import TestIntelligentBinning
            from tests.ui.test_binning_ui_integration import TestBinningUIIntegration
            
            # Create test suites
            loader = unittest.TestLoader()
            core_suite = loader.loadTestsFromTestCase(TestIntelligentBinning)
            ui_suite = loader.loadTestsFromTestCase(TestBinningUIIntegration)
            
            # Run core binning tests
            print("Running Core Binning Tests...")
            core_result = self.run_test_suite(core_suite, "CoreBinningTests")
            
            # Run UI integration tests
            print("Running UI Integration Tests...")
            ui_result = self.run_test_suite(ui_suite, "UIBinningIntegrationTests")
            
            # Compile results
            total_tests = core_result['tests_run'] + ui_result['tests_run']
            total_failures = core_result['failures'] + ui_result['failures']
            total_errors = core_result['errors'] + ui_result['errors']
            
            binning_results = {
                'category': 'binning',
                'total_tests': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'success_rate': (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0,
                'core_tests': core_result,
                'ui_tests': ui_result,
                'execution_time': core_result['execution_time'] + ui_result['execution_time']
            }
            
            print(f"Binning Tests Summary:")
            print(f"  Total Tests: {total_tests}")
            print(f"  Failures: {total_failures}")
            print(f"  Errors: {total_errors}")
            print(f"  Success Rate: {binning_results['success_rate']:.1f}%")
            
            return binning_results
            
        except ImportError as e:
            print(f"Error importing binning tests: {e}")
            return {'error': f'Failed to import binning tests: {e}'}
    
    def run_basic_functionality_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests"""
        print("Running Basic Functionality Tests")
        print("-" * 40)
        
        results = {}
        
        try:
            # Test imports
            print("Testing imports...")
            from healthcare_dss import DataManager, ModelManager
            from healthcare_dss import PreprocessingEngine, ModelTrainingEngine, ModelEvaluationEngine, ModelRegistry
            print("All imports successful")
            results['imports'] = True
            
        except Exception as e:
            print(f"Import test failed: {e}")
            results['imports'] = False
            return results
        
        try:
            # Test initialization
            print("Testing component initialization...")
            data_manager = DataManager()
            model_manager = ModelManager(data_manager)
            preprocessing_engine = PreprocessingEngine()
            training_engine = ModelTrainingEngine()
            evaluation_engine = ModelEvaluationEngine()
            registry = ModelRegistry()
            print("All components initialized successfully")
            results['initialization'] = True
            
        except Exception as e:
            print(f"Initialization test failed: {e}")
            results['initialization'] = False
            return results
        
        try:
            # Test data loading
            print("Testing data loading...")
            datasets = list(data_manager.datasets.keys())
            print(f"Datasets loaded: {datasets}")
            results['data_loading'] = True
            
        except Exception as e:
            print(f"Data loading test failed: {e}")
            results['data_loading'] = False
        
        # Calculate success rate
        successful_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"Basic Functionality Tests Summary:")
        print(f"  Passed: {successful_tests}/{total_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        return {
            'category': 'basic_functionality',
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'test_results': results
        }
    
    def run_all_tests(self, include_binning: bool = True, include_basic: bool = True) -> Dict[str, Any]:
        """Run all tests"""
        logger.info("Starting comprehensive test execution...")
        self.start_time = time.time()
        
        all_results = {}
        
        # Run comprehensive test suites
        print("Running Comprehensive Test Suites")
        print("=" * 50)
        
        test_suites = self.discover_tests()
        
        if test_suites:
            suite_results = []
            total_tests = 0
            total_failures = 0
            total_errors = 0
            
            for i, suite in enumerate(test_suites):
                suite_name = f"TestSuite_{i+1}"
                suite_result = self.run_test_suite(suite, suite_name)
                suite_results.append(suite_result)
                
                total_tests += suite_result['tests_run']
                total_failures += suite_result['failures']
                total_errors += suite_result['errors']
            
            all_results['comprehensive_tests'] = {
                'total_suites': len(test_suites),
                'total_tests': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'success_rate': (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0,
                'suite_results': suite_results
            }
        else:
            all_results['comprehensive_tests'] = {'error': 'No test suites found'}
        
        # Run binning tests if requested
        if include_binning:
            print("\n" + "=" * 50)
            binning_results = self.run_binning_tests()
            all_results['binning_tests'] = binning_results
        
        # Run basic functionality tests if requested
        if include_basic:
            print("\n" + "=" * 50)
            basic_results = self.run_basic_functionality_tests()
            all_results['basic_functionality_tests'] = basic_results
        
        self.end_time = time.time()
        
        # Compile overall results
        overall_results = {
            'execution_summary': self._compile_execution_summary(all_results),
            'test_categories': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_execution_time': self.end_time - self.start_time
        }
        
        self.test_results = overall_results
        return overall_results
    
    def _compile_execution_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile execution summary from all test results"""
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        # Count comprehensive tests
        if 'comprehensive_tests' in all_results and 'error' not in all_results['comprehensive_tests']:
            comp = all_results['comprehensive_tests']
            total_tests += comp.get('total_tests', 0)
            total_failures += comp.get('total_failures', 0)
            total_errors += comp.get('total_errors', 0)
        
        # Count binning tests
        if 'binning_tests' in all_results and 'error' not in all_results['binning_tests']:
            binning = all_results['binning_tests']
            total_tests += binning.get('total_tests', 0)
            total_failures += binning.get('total_failures', 0)
            total_errors += binning.get('total_errors', 0)
        
        # Count basic functionality tests
        if 'basic_functionality_tests' in all_results:
            basic = all_results['basic_functionality_tests']
            total_tests += basic.get('total_tests', 0)
            total_failures += basic.get('total_tests', 0) - basic.get('successful_tests', 0)
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'overall_success_rate': (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
        }
    
    def generate_report(self, results: Dict[str, Any] = None) -> str:
        """Generate a comprehensive test report"""
        if results is None:
            results = self.test_results
        
        if not results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("HEALTHCARE DSS COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # Execution summary
        summary = results.get('execution_summary', {})
        report.append("EXECUTION SUMMARY:")
        report.append("-" * 20)
        report.append(f"Total Tests: {summary.get('total_tests', 0)}")
        report.append(f"Passed: {summary.get('total_tests', 0) - summary.get('total_failures', 0) - summary.get('total_errors', 0)}")
        report.append(f"Failed: {summary.get('total_failures', 0)}")
        report.append(f"Errors: {summary.get('total_errors', 0)}")
        report.append(f"Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        report.append(f"Total Execution Time: {results.get('total_execution_time', 0):.2f} seconds")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 15)
        
        if summary.get('total_failures', 0) > 0:
            report.append("Some tests failed. Review failure details above.")
        if summary.get('total_errors', 0) > 0:
            report.append("Some tests had errors. Check error details above.")
        if summary.get('overall_success_rate', 0) == 100:
            report.append("All tests passed! System is working correctly.")
        elif summary.get('overall_success_rate', 0) >= 90:
            report.append("Most tests passed. System is mostly functional.")
        elif summary.get('overall_success_rate', 0) >= 70:
            report.append("Some tests failed. System needs attention.")
        else:
            report.append("Many tests failed. System requires significant fixes.")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def print_report(self, results: Dict[str, Any] = None):
        """Print the test report to console"""
        report = self.generate_report(results)
        print(report)
    
    def save_results(self, results: Dict[str, Any] = None, filename: str = None):
        """Save test results to JSON file"""
        if results is None:
            results = self.test_results
        
        if filename is None:
            filename = f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {filepath}")
        return filepath


def run_comprehensive_tests():
    """Main function to run comprehensive tests"""
    print("Healthcare DSS Comprehensive Test Suite")
    print("=" * 60)
    
    # Initialize test runner
    runner = HealthcareDSSTestRunner()
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Generate and print report
        runner.print_report(results)
        
        # Save results
        results_file = runner.save_results(results)
        
        # Print summary
        summary = results.get('execution_summary', {})
        print(f"\nTest Summary:")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"   Execution Time: {results.get('total_execution_time', 0):.2f}s")
        print(f"   Results saved to: {results_file}")
        
        # Return success status
        return summary.get('overall_success_rate', 0) >= 90
        
    except Exception as e:
        logger.error(f"Error running comprehensive tests: {e}")
        print(f"Error running tests: {e}")
        return False


def run_specific_test_category(category: str):
    """Run tests for a specific category"""
    categories = {
        'core': 'tests/core',
        'analytics': 'tests/analytics',
        'ui': 'tests/ui',
        'utils': 'tests/utils',
        'integration': 'tests/integration',
        'modules': 'tests/modules',
        'edge_cases': 'tests/edge_cases',
        'binning': 'binning_specific'
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    print(f"Running {category} tests...")
    
    runner = HealthcareDSSTestRunner()
    
    if category == 'binning':
        # Run binning-specific tests
        results = runner.run_binning_tests()
        success_rate = results.get('success_rate', 0)
    else:
        # Run category tests
        test_dir = categories[category]
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return False
        
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='test_*.py')
        
        if suite.countTestCases() == 0:
            print(f"No tests found in {test_dir}")
            return True
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(suite)
        
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
        
        print(f"\n{category.title()} Test Results:")
        print(f"   Tests Run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 90


def run_binning_tests():
    """Run only binning tests"""
    print("Running Binning System Tests")
    print("=" * 40)
    
    runner = HealthcareDSSTestRunner()
    results = runner.run_binning_tests()
    
    success_rate = results.get('success_rate', 0)
    print(f"\nBinning Tests Summary:")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 90


def run_basic_tests():
    """Run only basic functionality tests"""
    print("Running Basic Functionality Tests")
    print("=" * 40)
    
    runner = HealthcareDSSTestRunner()
    results = runner.run_basic_functionality_tests()
    
    success_rate = results.get('success_rate', 0)
    print(f"\nBasic Tests Summary:")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 90


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description='Healthcare DSS Test Runner')
    parser.add_argument('--category', '-c', 
                       choices=['core', 'analytics', 'ui', 'utils', 'integration', 'modules', 'edge_cases', 'binning'],
                       help='Run tests for a specific category')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run all tests comprehensively')
    parser.add_argument('--binning-only', action='store_true',
                       help='Run only binning tests')
    parser.add_argument('--basic-only', action='store_true',
                       help='Run only basic functionality tests')
    parser.add_argument('--no-binning', action='store_true',
                       help='Exclude binning tests from comprehensive run')
    parser.add_argument('--no-basic', action='store_true',
                       help='Exclude basic functionality tests from comprehensive run')
    
    args = parser.parse_args()
    
    if args.category:
        success = run_specific_test_category(args.category)
    elif args.binning_only:
        success = run_binning_tests()
    elif args.basic_only:
        success = run_basic_tests()
    elif args.comprehensive:
        runner = HealthcareDSSTestRunner()
        results = runner.run_all_tests(
            include_binning=not args.no_binning,
            include_basic=not args.no_basic
        )
        runner.print_report(results)
        summary = results.get('execution_summary', {})
        success = summary.get('overall_success_rate', 0) >= 90
    else:
        # Default: run comprehensive tests
        success = run_comprehensive_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
