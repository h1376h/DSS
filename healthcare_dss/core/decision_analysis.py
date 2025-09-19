"""
Decision Analysis Engine for Healthcare DSS
===========================================

This module handles decision trees and multi-criteria decision analysis:
- Decision tree evaluation
- Multi-criteria decision analysis (MCDA)
- Sensitivity analysis
- Decision support frameworks
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class DecisionAnalysisEngine:
    """
    Engine for decision analysis including decision trees and MCDA
    """
    
    def __init__(self):
        """Initialize the Decision Analysis Engine"""
        self.decision_trees = {}
        self._create_default_decision_trees()
    
    def _create_default_decision_trees(self):
        """Create default decision trees for clinical decision support"""
        # Diabetes risk assessment decision tree
        diabetes_tree = {
            "tree_id": "diabetes_risk_tree",
            "name": "Diabetes Risk Assessment",
            "description": "Decision tree for assessing diabetes risk",
            "root": {
                "condition": "age",
                "operator": ">=",
                "threshold": 45,
                "true_branch": {
                    "condition": "bmi",
                    "operator": ">=",
                    "threshold": 25,
                    "true_branch": {
                        "condition": "family_history",
                        "operator": "==",
                        "threshold": True,
                        "true_branch": {"outcome": "high_risk", "recommendations": ["HbA1c test", "Lifestyle counseling"]},
                        "false_branch": {"outcome": "moderate_risk", "recommendations": ["Annual screening"]}
                    },
                    "false_branch": {"outcome": "low_risk", "recommendations": ["Routine care"]}
                },
                "false_branch": {
                    "condition": "bmi",
                    "operator": ">=",
                    "threshold": 30,
                    "true_branch": {"outcome": "moderate_risk", "recommendations": ["Weight management"]},
                    "false_branch": {"outcome": "low_risk", "recommendations": ["Routine care"]}
                }
            }
        }
        
        self.decision_trees["diabetes_risk_tree"] = diabetes_tree
        
        # Breast cancer risk assessment decision tree
        breast_cancer_tree = {
            "tree_id": "breast_cancer_risk_tree",
            "name": "Breast Cancer Risk Assessment",
            "description": "Decision tree for breast cancer risk assessment",
            "root": {
                "condition": "age",
                "operator": ">=",
                "threshold": 40,
                "true_branch": {
                    "condition": "family_history",
                    "operator": "==",
                    "threshold": True,
                    "true_branch": {"outcome": "high_risk", "recommendations": ["Annual mammography", "Genetic counseling"]},
                    "false_branch": {"outcome": "moderate_risk", "recommendations": ["Biennial mammography"]}
                },
                "false_branch": {"outcome": "low_risk", "recommendations": ["Routine care"]}
            }
        }
        
        self.decision_trees["breast_cancer_risk_tree"] = breast_cancer_tree
        logger.info("Default decision trees created")
    
    def evaluate_decision_tree(self, tree_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a decision tree with patient data
        
        Args:
            tree_id: ID of the decision tree to evaluate
            patient_data: Dictionary containing patient information
            
        Returns:
            Decision tree evaluation result
        """
        if tree_id not in self.decision_trees:
            raise ValueError(f"Decision tree {tree_id} not found")
        
        tree = self.decision_trees[tree_id]
        result = self._traverse_decision_tree(tree['root'], patient_data)
        
        return {
            'tree_id': tree_id,
            'tree_name': tree['name'],
            'outcome': result['outcome'],
            'recommendations': result['recommendations'],
            'confidence': result.get('confidence', 0.8)
        }
    
    def _traverse_decision_tree(self, node: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively traverse decision tree"""
        if 'outcome' in node:
            return node
        
        condition = node['condition']
        operator = node['operator']
        threshold = node['threshold']
        
        patient_value = patient_data.get(condition, 0)
        
        # Evaluate condition
        condition_met = False
        if operator == '>=':
            condition_met = patient_value >= threshold
        elif operator == '>':
            condition_met = patient_value > threshold
        elif operator == '<=':
            condition_met = patient_value <= threshold
        elif operator == '<':
            condition_met = patient_value < threshold
        elif operator == '==':
            condition_met = patient_value == threshold
        
        # Traverse appropriate branch
        if condition_met and 'true_branch' in node:
            return self._traverse_decision_tree(node['true_branch'], patient_data)
        elif not condition_met and 'false_branch' in node:
            return self._traverse_decision_tree(node['false_branch'], patient_data)
        else:
            return {'outcome': 'unknown', 'recommendations': ['Further evaluation needed']}
    
    def multi_criteria_decision_analysis(self, alternatives: List[Dict[str, Any]], 
                                       criteria: List[Dict[str, Any]], 
                                       weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform multi-criteria decision analysis for healthcare decisions
        
        Args:
            alternatives: List of alternative options
            criteria: List of evaluation criteria
            weights: Weights for each criterion
            
        Returns:
            MCDA analysis results
        """
        # Normalize criteria weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate scores for each alternative
        alternative_scores = {}
        for alt in alternatives:
            alt_id = alt['id']
            score = 0
            
            for criterion in criteria:
                criterion_name = criterion['name']
                if criterion_name in alt and criterion_name in normalized_weights:
                    # Normalize criterion value (assuming 0-1 scale)
                    criterion_value = alt[criterion_name]
                    if criterion['maximize']:
                        score += criterion_value * normalized_weights[criterion_name]
                    else:
                        score += (1 - criterion_value) * normalized_weights[criterion_name]
            
            alternative_scores[alt_id] = score
        
        # Rank alternatives
        ranked_alternatives = sorted(alternative_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranked_alternatives': ranked_alternatives,
            'scores': alternative_scores,
            'weights': normalized_weights,
            'recommendation': ranked_alternatives[0][0] if ranked_alternatives else None
        }
    
    def perform_sensitivity_analysis(self, alternatives: List[Dict[str, Any]], 
                                   criteria: List[Dict[str, Any]], 
                                   weights: Dict[str, float],
                                   sensitivity_range: float = 0.2) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on decision criteria weights
        
        Args:
            alternatives: List of alternative options
            criteria: List of evaluation criteria
            weights: Base weights for each criterion
            sensitivity_range: Range for sensitivity testing (±20% by default)
            
        Returns:
            Sensitivity analysis results
        """
        base_result = self.multi_criteria_decision_analysis(alternatives, criteria, weights)
        sensitivity_results = {}
        
        for criterion_name in weights.keys():
            # Test weight variations
            test_weights = weights.copy()
            original_weight = test_weights[criterion_name]
            
            # Test increased weight
            test_weights[criterion_name] = original_weight * (1 + sensitivity_range)
            increased_result = self.multi_criteria_decision_analysis(alternatives, criteria, test_weights)
            
            # Test decreased weight
            test_weights[criterion_name] = original_weight * (1 - sensitivity_range)
            decreased_result = self.multi_criteria_decision_analysis(alternatives, criteria, test_weights)
            
            sensitivity_results[criterion_name] = {
                'base_recommendation': base_result['recommendation'],
                'increased_weight_recommendation': increased_result['recommendation'],
                'decreased_weight_recommendation': decreased_result['recommendation'],
                'sensitivity_score': self._calculate_sensitivity_score(
                    base_result, increased_result, decreased_result
                )
            }
        
        return {
            'base_analysis': base_result,
            'sensitivity_results': sensitivity_results,
            'most_sensitive_criteria': self._identify_most_sensitive_criteria(sensitivity_results)
        }
    
    def _calculate_sensitivity_score(self, base_result: Dict, increased_result: Dict, decreased_result: Dict) -> float:
        """Calculate sensitivity score for a criterion"""
        base_rec = base_result['recommendation']
        inc_rec = increased_result['recommendation']
        dec_rec = decreased_result['recommendation']
        
        # Count how many times recommendation changes
        changes = 0
        if base_rec != inc_rec:
            changes += 1
        if base_rec != dec_rec:
            changes += 1
        
        return changes / 2.0  # Normalize to 0-1 scale
    
    def _identify_most_sensitive_criteria(self, sensitivity_results: Dict) -> List[str]:
        """Identify criteria with highest sensitivity scores"""
        sorted_criteria = sorted(
            sensitivity_results.items(),
            key=lambda x: x[1]['sensitivity_score'],
            reverse=True
        )
        return [criterion for criterion, _ in sorted_criteria[:3]]  # Top 3 most sensitive
    
    def get_decision_support_frameworks(self) -> Dict[str, Any]:
        """
        Get comprehensive decision support frameworks following DSS documentation
        
        Returns:
            Dictionary containing various decision support frameworks
        """
        frameworks = {
            'herbert_simon_model': {
                'phase_1_intelligence': {
                    'description': 'Problem identification and data gathering',
                    'activities': [
                        'Identify decision problem',
                        'Gather relevant data',
                        'Define problem scope',
                        'Identify stakeholders'
                    ],
                    'tools': [
                        'Data mining',
                        'Statistical analysis',
                        'Trend analysis',
                        'Benchmarking'
                    ]
                },
                'phase_2_design': {
                    'description': 'Model formulation and alternative generation',
                    'activities': [
                        'Develop decision models',
                        'Generate alternatives',
                        'Define evaluation criteria',
                        'Set constraints'
                    ],
                    'tools': [
                        'Mathematical modeling',
                        'Simulation',
                        'Optimization',
                        'Scenario analysis'
                    ]
                },
                'phase_3_choice': {
                    'description': 'Alternative evaluation and selection',
                    'activities': [
                        'Evaluate alternatives',
                        'Apply decision criteria',
                        'Perform sensitivity analysis',
                        'Select best alternative'
                    ],
                    'tools': [
                        'Multi-criteria analysis',
                        'Cost-benefit analysis',
                        'Risk analysis',
                        'Decision trees'
                    ]
                },
                'phase_4_implementation': {
                    'description': 'Solution implementation and monitoring',
                    'activities': [
                        'Implement solution',
                        'Monitor progress',
                        'Evaluate outcomes',
                        'Make adjustments'
                    ],
                    'tools': [
                        'Project management',
                        'Performance monitoring',
                        'Feedback systems',
                        'Continuous improvement'
                    ]
                }
            },
            'decision_analysis_methods': {
                'analytical_hierarchy_process': {
                    'description': 'Structured approach for complex decision making',
                    'steps': [
                        'Define decision hierarchy',
                        'Pairwise comparison of criteria',
                        'Calculate priority weights',
                        'Evaluate alternatives',
                        'Synthesize results'
                    ],
                    'applications': [
                        'Treatment selection',
                        'Resource allocation',
                        'Technology evaluation',
                        'Policy decisions'
                    ]
                },
                'topsis_method': {
                    'description': 'Technique for Order Preference by Similarity to Ideal Solution',
                    'steps': [
                        'Normalize decision matrix',
                        'Calculate weighted normalized matrix',
                        'Determine ideal and negative ideal solutions',
                        'Calculate separation measures',
                        'Calculate relative closeness to ideal solution'
                    ],
                    'applications': [
                        'Drug selection',
                        'Equipment evaluation',
                        'Service quality assessment',
                        'Performance ranking'
                    ]
                },
                'electre_method': {
                    'description': 'Elimination and Choice Expressing Reality',
                    'steps': [
                        'Normalize decision matrix',
                        'Calculate concordance and discordance matrices',
                        'Determine concordance and discordance thresholds',
                        'Build outranking relations',
                        'Perform ranking'
                    ],
                    'applications': [
                        'Multi-objective optimization',
                        'Complex healthcare decisions',
                        'Resource allocation',
                        'Strategic planning'
                    ]
                }
            },
            'clinical_decision_support': {
                'evidence_based_medicine': {
                    'framework': 'PICO (Patient, Intervention, Comparison, Outcome)',
                    'steps': [
                        'Formulate clinical question',
                        'Search for evidence',
                        'Appraise evidence quality',
                        'Apply evidence to patient',
                        'Evaluate outcomes'
                    ],
                    'tools': [
                        'Systematic reviews',
                        'Meta-analyses',
                        'Clinical guidelines',
                        'Decision aids'
                    ]
                },
                'shared_decision_making': {
                    'framework': 'Patient-centered care approach',
                    'components': [
                        'Patient preferences',
                        'Clinical evidence',
                        'Provider expertise',
                        'Risk communication'
                    ],
                    'tools': [
                        'Decision aids',
                        'Risk calculators',
                        'Patient education materials',
                        'Communication training'
                    ]
                },
                'clinical_pathways': {
                    'framework': 'Standardized care processes',
                    'components': [
                        'Evidence-based protocols',
                        'Quality indicators',
                        'Outcome measures',
                        'Variance tracking'
                    ],
                    'benefits': [
                        'Improved outcomes',
                        'Reduced variation',
                        'Cost efficiency',
                        'Quality assurance'
                    ]
                }
            },
            'risk_analysis_frameworks': {
                'fmea_analysis': {
                    'description': 'Failure Mode and Effects Analysis',
                    'steps': [
                        'Identify potential failures',
                        'Assess failure effects',
                        'Determine failure causes',
                        'Calculate risk priority numbers',
                        'Develop mitigation strategies'
                    ],
                    'applications': [
                        'Patient safety',
                        'Medication errors',
                        'Equipment failures',
                        'Process improvements'
                    ]
                },
                'fault_tree_analysis': {
                    'description': 'Top-down approach to failure analysis',
                    'steps': [
                        'Define top event',
                        'Identify contributing factors',
                        'Build fault tree',
                        'Calculate probabilities',
                        'Identify critical paths'
                    ],
                    'applications': [
                        'Root cause analysis',
                        'Safety assessment',
                        'Reliability analysis',
                        'Incident investigation'
                    ]
                },
                'monte_carlo_simulation': {
                    'description': 'Probabilistic risk assessment',
                    'steps': [
                        'Define probability distributions',
                        'Generate random scenarios',
                        'Run simulations',
                        'Analyze results',
                        'Calculate risk metrics'
                    ],
                    'applications': [
                        'Financial risk',
                        'Operational risk',
                        'Clinical outcomes',
                        'Resource planning'
                    ]
                }
            }
        }
        
        return frameworks
    
    def add_decision_tree(self, tree_id: str, tree_structure: Dict[str, Any]):
        """Add a new decision tree to the engine"""
        self.decision_trees[tree_id] = tree_structure
        logger.info(f"Added decision tree: {tree_id}")
    
    def get_decision_tree(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """Get a decision tree by ID"""
        return self.decision_trees.get(tree_id)
    
    def get_decision_tree_statistics(self) -> Dict[str, int]:
        """Get statistics about decision trees"""
        return {
            'total_trees': len(self.decision_trees),
            'available_trees': list(self.decision_trees.keys())
        }
