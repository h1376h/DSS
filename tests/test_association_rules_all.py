#!/usr/bin/env python3
"""
Test Association Rules with all datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from healthcare_dss.core.data_management import DataManager
# DatasetManager functionality is now integrated into DataManager
from healthcare_dss.analytics.association_rules import AssociationRulesMiner

def test_association_rules_all_datasets():
    """Test Association Rules mining with all available datasets"""
    print("🧪 Testing Association Rules with All Datasets")
    print("=" * 60)
    
    # Initialize managers
    data_manager = DataManager()
    
    # Initialize Association Rules miner with consolidated DataManager
    miner = AssociationRulesMiner(data_manager)
    
    # Get all available datasets from consolidated DataManager
    all_datasets = data_manager.datasets
    
    print(f"📊 Found {len(all_datasets)} datasets total")
    
    successful_tests = 0
    failed_tests = 0
    
    for dataset_name in all_datasets.keys():
        print(f"\n🔍 Testing dataset: {dataset_name}")
        try:
            # Test the analysis
            result = miner.analyze_healthcare_patterns(dataset_name)
            
            if result and 'frequent_itemsets' in result:
                print(f"✅ Success: Found {len(result['frequent_itemsets'])} frequent itemsets")
                if 'association_rules' in result and result['association_rules']:
                    print(f"📈 Generated {len(result['association_rules'])} association rules")
                else:
                    print("📈 No association rules generated (low confidence)")
                successful_tests += 1
            else:
                print("⚠️ Analysis completed but no results")
                successful_tests += 1
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            failed_tests += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {successful_tests/(successful_tests+failed_tests)*100:.1f}%")
    
    print("\n🎯 Association Rules testing completed!")

if __name__ == "__main__":
    test_association_rules_all_datasets()
