"""
Clinical Rule Engine for Healthcare DSS
======================================

This module handles rule evaluation and condition processing:
- Rule condition evaluation
- Clinical rule processing
- Rule-based decision support
- Condition parsing and validation
"""

import logging
from typing import Dict, List, Any, Optional
import re

from healthcare_dss.core.knowledge_models import ClinicalRule, RuleType, SeverityLevel

# Configure logging
logger = logging.getLogger(__name__)


class ClinicalRuleEngine:
    """
    Engine for evaluating clinical rules and processing conditions
    """
    
    def __init__(self):
        """Initialize the Clinical Rule Engine"""
        self.rules = {}
        self.active_rules = {}
    
    def load_rules(self, rules: Dict[str, ClinicalRule]):
        """Load clinical rules into the engine"""
        self.rules = rules
        self.active_rules = {k: v for k, v in rules.items() if v.active}
        logger.info(f"Loaded {len(self.rules)} rules, {len(self.active_rules)} active")
    
    def evaluate_clinical_rules(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate clinical rules against patient data
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            List of triggered rules with recommendations
        """
        triggered_rules = []
        
        for rule_id, rule in self.active_rules.items():
            # Check if rule conditions are met
            if self._evaluate_conditions(rule.conditions, patient_data):
                triggered_rules.append({
                    'rule_id': rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity.value,
                    'actions': rule.actions,
                    'evidence_level': rule.evidence_level
                })
        
        return triggered_rules
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """Evaluate rule conditions against patient data"""
        for condition_key, condition_value in conditions.items():
            if condition_key not in patient_data:
                return False
            
            patient_value = patient_data[condition_key]
            
            # Parse condition (e.g., "> 140", ">= 25", "== True")
            if isinstance(condition_value, str):
                if not self._evaluate_string_condition(patient_value, condition_value):
                    return False
            else:
                # Direct comparison
                if not (patient_value == condition_value):
                    return False
        
        return True
    
    def _evaluate_string_condition(self, patient_value: Any, condition_value: str) -> bool:
        """Evaluate string-based conditions"""
        try:
            # Handle >= first, then > to avoid parsing issues
            if ">=" in condition_value:
                threshold = float(condition_value.replace(">=", "").strip())
                return patient_value >= threshold
            elif ">" in condition_value:
                threshold = float(condition_value.replace(">", "").strip())
                return patient_value > threshold
            elif "<=" in condition_value:
                threshold = float(condition_value.replace("<=", "").strip())
                return patient_value <= threshold
            elif "<" in condition_value:
                threshold = float(condition_value.replace("<", "").strip())
                return patient_value < threshold
            elif "==" in condition_value:
                threshold = self._parse_threshold_value(condition_value.replace("==", "").strip())
                return patient_value == threshold
            elif "=" in condition_value and "==" not in condition_value:
                # Handle single = as equality
                threshold = self._parse_threshold_value(condition_value.replace("=", "").strip())
                return patient_value == threshold
            else:
                # Direct string comparison
                return str(patient_value).lower() == condition_value.lower()
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Error evaluating condition '{condition_value}': {e}")
            return False
    
    def _parse_threshold_value(self, threshold_str: str) -> Any:
        """Parse threshold value from string"""
        threshold_str = threshold_str.strip()
        
        if threshold_str.lower() == 'true':
            return True
        elif threshold_str.lower() == 'false':
            return False
        elif threshold_str.isdigit():
            return float(threshold_str)
        elif '.' in threshold_str and threshold_str.replace('.', '').isdigit():
            return float(threshold_str)
        else:
            return threshold_str
    
    def evaluate_guideline_conditions(self, conditions: List[str], patient_data: Dict[str, Any]) -> bool:
        """Evaluate guideline conditions against patient data"""
        for condition in conditions:
            if not self._evaluate_guideline_condition(condition, patient_data):
                return False
        return True
    
    def _evaluate_guideline_condition(self, condition: str, patient_data: Dict[str, Any]) -> bool:
        """Evaluate a single guideline condition"""
        try:
            # Age-based conditions
            if "age >=" in condition:
                age_threshold = int(condition.split("age >=")[1].strip())
                return patient_data.get('age', 0) >= age_threshold
            elif "age >" in condition:
                age_threshold = int(condition.split("age >")[1].strip())
                return patient_data.get('age', 0) > age_threshold
            
            # BMI-based conditions
            elif "bmi >=" in condition:
                bmi_threshold = float(condition.split("bmi >=")[1].strip())
                return patient_data.get('bmi', 0) >= bmi_threshold
            elif "bmi >" in condition:
                bmi_threshold = float(condition.split("bmi >")[1].strip())
                return patient_data.get('bmi', 0) > bmi_threshold
            
            # HbA1c-based conditions
            elif "hba1c >=" in condition:
                hba1c_threshold = float(condition.split("hba1c >=")[1].strip().replace('%', ''))
                return patient_data.get('hba1c', 0) >= hba1c_threshold
            elif "hba1c >" in condition:
                hba1c_threshold = float(condition.split("hba1c >")[1].strip().replace('%', ''))
                return patient_data.get('hba1c', 0) > hba1c_threshold
            
            # Boolean conditions
            elif "family_history" in condition:
                return patient_data.get('family_history', False)
            elif "confirmed_diabetes" in condition:
                # Assume confirmed if hba1c > 6.5%
                return patient_data.get('hba1c', 0) > 6.5
            elif "sedentary_lifestyle" in condition:
                # Assume sedentary if no exercise data provided
                return patient_data.get('sedentary_lifestyle', True)
            elif "genetic_predisposition" in condition:
                # Assume genetic predisposition if family history exists
                return patient_data.get('family_history', False)
            elif "suspicious_imaging" in condition:
                # Assume suspicious imaging if certain conditions are met
                return patient_data.get('suspicious_imaging', False)
            elif "high_risk_features" in condition:
                # Assume high risk features if certain conditions are met
                return patient_data.get('high_risk_features', False)
            
            return True  # Default to true for unrecognized conditions
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error evaluating guideline condition '{condition}': {e}")
            return False
    
    def get_rule_by_id(self, rule_id: str) -> Optional[ClinicalRule]:
        """Get a clinical rule by ID"""
        return self.rules.get(rule_id)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[ClinicalRule]:
        """Get all rules of a specific type"""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]
    
    def get_rules_by_severity(self, severity: SeverityLevel) -> List[ClinicalRule]:
        """Get all rules of a specific severity level"""
        return [rule for rule in self.rules.values() if rule.severity == severity]
    
    def activate_rule(self, rule_id: str) -> bool:
        """Activate a clinical rule"""
        if rule_id in self.rules:
            self.rules[rule_id].active = True
            self.active_rules[rule_id] = self.rules[rule_id]
            logger.info(f"Activated rule: {rule_id}")
            return True
        return False
    
    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a clinical rule"""
        if rule_id in self.rules:
            self.rules[rule_id].active = False
            if rule_id in self.active_rules:
                del self.active_rules[rule_id]
            logger.info(f"Deactivated rule: {rule_id}")
            return True
        return False
    
    def validate_rule_conditions(self, conditions: Dict[str, Any]) -> List[str]:
        """Validate rule conditions and return any errors"""
        errors = []
        
        for condition_key, condition_value in conditions.items():
            if isinstance(condition_value, str):
                if not self._validate_string_condition(condition_value):
                    errors.append(f"Invalid condition format for '{condition_key}': {condition_value}")
        
        return errors
    
    def _validate_string_condition(self, condition_value: str) -> bool:
        """Validate string condition format"""
        # Check for valid operators
        valid_operators = ['>=', '>', '<=', '<', '==', '=']
        has_valid_operator = any(op in condition_value for op in valid_operators)
        
        if not has_valid_operator:
            return False
        
        # Additional validation can be added here
        return True
    
    def get_rule_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded rules"""
        stats = {
            'total_rules': len(self.rules),
            'active_rules': len(self.active_rules),
            'inactive_rules': len(self.rules) - len(self.active_rules)
        }
        
        # Count by rule type
        for rule_type in RuleType:
            stats[f'{rule_type.value}_rules'] = len(self.get_rules_by_type(rule_type))
        
        # Count by severity
        for severity in SeverityLevel:
            stats[f'{severity.value}_severity_rules'] = len(self.get_rules_by_severity(severity))
        
        return stats
