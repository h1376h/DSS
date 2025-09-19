#!/usr/bin/env python3
"""
Test script to verify real model data loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from healthcare_dss.ui.analytics.analytics_dashboard import load_real_model_data, get_real_model_performance, get_real_performance_trends

def test_real_data_loading():
    """Test if real model data is being loaded correctly"""
    print("🧪 Testing Real Model Data Loading")
    print("=" * 50)
    
    # Test loading real model data
    real_data = load_real_model_data()
    if real_data:
        print("✅ Real model data loaded successfully")
        print(f"📊 Data keys: {list(real_data.keys())}")
        
        # Test model performance data
        performance = get_real_model_performance()
        if performance:
            print(f"✅ Model performance data: {len(performance)} entries")
            print("📈 Sample performance data:")
            for i, entry in enumerate(performance[:3]):
                print(f"  {i+1}. {entry['Dataset']} - {entry['Model']}: {entry['Accuracy']}")
        else:
            print("❌ No model performance data found")
        
        # Test performance trends data
        trends = get_real_performance_trends()
        if trends:
            print(f"✅ Performance trends data: {len(trends)} entries")
            print(f"📊 Date range: {trends[0]['date']} to {trends[-1]['date']}")
            print(f"📈 Accuracy range: {min(t['accuracy'] for t in trends):.3f} to {max(t['accuracy'] for t in trends):.3f}")
        else:
            print("❌ No performance trends data found")
            
    else:
        print("❌ Failed to load real model data")
        print("💡 Make sure 'dashboard_model_data.json' exists in the project root")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_real_data_loading()
