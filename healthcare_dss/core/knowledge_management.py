"""
Main Knowledge Manager for Healthcare DSS
=========================================

This module contains the main KnowledgeManager class that coordinates all knowledge management components:
- Clinical rules and guidelines management
- Decision support and analysis
- Knowledge base operations
- Intelligent recommendations
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from healthcare_dss.core.knowledge_models import ClinicalRule, ClinicalGuideline, RuleType, SeverityLevel
from healthcare_dss.core.knowledge_database import KnowledgeDatabaseManager
from healthcare_dss.core.rule_engine import ClinicalRuleEngine
from healthcare_dss.core.guidelines_manager import ClinicalGuidelinesManager
from healthcare_dss.core.decision_analysis import DecisionAnalysisEngine

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Main Knowledge-Based Management Subsystem for Healthcare DSS
    
    Implements comprehensive clinical knowledge management following Healthcare_DSS_Architecture.md:
    - Clinical Guidelines Integration and Evidence-Based Protocols
    - Expert System Architecture and Rule Management
    - Multi-Criteria Decision Analysis Framework
    - Evidence-based decision support with clinical validation
    - Knowledge representation and reasoning for healthcare
    
    Manages clinical knowledge, rules, guidelines, and decision support logic
    for healthcare applications.
    """
    
    def __init__(self, data_manager, model_manager, 
                 knowledge_db_path: str = "knowledge_base.db"):
        """
        Initialize Knowledge Manager
        
        Args:
            data_manager: Instance of DataManager for data access
            model_manager: Instance of ModelManager for model access
            knowledge_db_path: Path to knowledge base database
        """
        self.data_manager = data_manager
        self.model_manager = model_manager
        
        # Initialize components
        self.db_manager = KnowledgeDatabaseManager(knowledge_db_path)
        self.rule_engine = ClinicalRuleEngine()
        self.guidelines_manager = ClinicalGuidelinesManager(self.rule_engine)
        self.decision_engine = DecisionAnalysisEngine()
        
        # Knowledge storage
        self.clinical_rules = {}
        self.clinical_guidelines = {}
        self.knowledge_graph = {}
        
        # Load knowledge from database
        self._load_knowledge_from_database()
        self._build_knowledge_graph()
    
    def _load_knowledge_from_database(self):
        """Load knowledge from database into memory"""
        try:
            # Load clinical rules
            db_rules = self.db_manager.get_clinical_rules()
            for rule_data in db_rules:
                rule = ClinicalRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    rule_type=RuleType(rule_data['rule_type']),
                    conditions=rule_data['conditions'],
                    actions=rule_data['actions'],
                    severity=SeverityLevel(rule_data['severity']),
                    evidence_level=rule_data['evidence_level'],
                    created_at=datetime.fromisoformat(rule_data['created_at']),
                    updated_at=datetime.fromisoformat(rule_data['updated_at']),
                    active=rule_data['active']
                )
                self.clinical_rules[rule.rule_id] = rule
            
            # Load clinical guidelines
            db_guidelines = self.db_manager.get_clinical_guidelines()
            for guideline_data in db_guidelines:
                guideline = ClinicalGuideline(
                    guideline_id=guideline_data['guideline_id'],
                    title=guideline_data['title'],
                    description=guideline_data['description'],
                    category=guideline_data['category'],
                    conditions=guideline_data['conditions'],
                    recommendations=guideline_data['recommendations'],
                    evidence_level=guideline_data['evidence_level'],
                    source=guideline_data['source'],
                    version=guideline_data['version'],
                    created_at=datetime.fromisoformat(guideline_data['created_at'])
                )
                self.clinical_guidelines[guideline.guideline_id] = guideline
            
            # Load rules into rule engine
            self.rule_engine.load_rules(self.clinical_rules)
            
            logger.info(f"Loaded {len(self.clinical_rules)} rules and {len(self.clinical_guidelines)} guidelines")
            
        except Exception as e:
            logger.error(f"Error loading knowledge from database: {e}")
    
    def _build_knowledge_graph(self):
        """Build a knowledge graph of medical concepts and relationships"""
        knowledge_relationships = [
            ("diabetes", "causes", "high_blood_glucose", 0.9, "medical_literature"),
            ("diabetes", "increases_risk", "heart_disease", 0.8, "medical_literature"),
            ("diabetes", "increases_risk", "kidney_disease", 0.7, "medical_literature"),
            ("obesity", "increases_risk", "diabetes", 0.8, "medical_literature"),
            ("family_history", "increases_risk", "diabetes", 0.6, "medical_literature"),
            ("breast_cancer", "detected_by", "mammography", 0.9, "medical_literature"),
            ("breast_cancer", "increases_risk", "metastasis", 0.7, "medical_literature"),
            ("early_detection", "improves", "survival_rate", 0.9, "medical_literature"),
            ("metformin", "treats", "diabetes", 0.9, "medical_literature"),
            ("lifestyle_modification", "prevents", "diabetes", 0.7, "medical_literature")
        ]
        
        for source, relationship, target, confidence, source_ref in knowledge_relationships:
            self.add_knowledge_relationship(source, relationship, target, confidence, source_ref)
    
    def add_clinical_rule(self, rule: ClinicalRule):
        """Add a clinical rule to the knowledge base"""
        self.clinical_rules[rule.rule_id] = rule
        self.db_manager.add_clinical_rule(rule)
        self.rule_engine.load_rules(self.clinical_rules)
        logger.info(f"Added clinical rule: {rule.name}")
    
    def add_clinical_guideline(self, guideline: ClinicalGuideline):
        """Add a clinical guideline to the knowledge base"""
        self.clinical_guidelines[guideline.guideline_id] = guideline
        self.db_manager.add_clinical_guideline(guideline)
        self.guidelines_manager.add_clinical_guideline(guideline)
        logger.info(f"Added clinical guideline: {guideline.title}")
    
    def add_knowledge_relationship(self, source: str, relationship: str, target: str, 
                                 confidence: float, source_ref: str):
        """Add a relationship to the knowledge graph"""
        self.knowledge_graph[f"{source}_{relationship}_{target}"] = {
            'source': source,
            'relationship': relationship,
            'target': target,
            'confidence': confidence,
            'source_ref': source_ref,
            'created_at': datetime.now()
        }
        self.db_manager.add_knowledge_relationship(source, relationship, target, confidence, source_ref)
    
    def evaluate_clinical_rules(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate clinical rules against patient data
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            List of triggered rules with recommendations
        """
        return self.rule_engine.evaluate_clinical_rules(patient_data)
    
    def get_clinical_recommendations(self, patient_data: Dict[str, Any], 
                                   category: str = None) -> List[Dict[str, Any]]:
        """
        Get clinical recommendations based on patient data and guidelines
        
        Args:
            patient_data: Dictionary containing patient information
            category: Optional category filter for guidelines
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Get recommendations from guidelines
        guideline_recommendations = self.guidelines_manager.get_clinical_recommendations(patient_data, category)
        recommendations.extend(guideline_recommendations)
        
        # Get recommendations from triggered rules
        triggered_rules = self.evaluate_clinical_rules(patient_data)
        for rule in triggered_rules:
            for action in rule['actions']:
                recommendations.append({
                    'rule_id': rule['rule_id'],
                    'title': rule['name'],
                    'recommendation': action,
                    'evidence_level': rule['evidence_level'],
                    'source': 'Clinical Decision Support System'
                })
        
        return recommendations
    
    def evaluate_decision_tree(self, tree_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a decision tree with patient data
        
        Args:
            tree_id: ID of the decision tree to evaluate
            patient_data: Dictionary containing patient information
            
        Returns:
            Decision tree evaluation result
        """
        return self.decision_engine.evaluate_decision_tree(tree_id, patient_data)
    
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
        return self.decision_engine.multi_criteria_decision_analysis(alternatives, criteria, weights)
    
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
        return self.decision_engine.perform_sensitivity_analysis(alternatives, criteria, weights, sensitivity_range)
    
    def get_decision_support_frameworks(self) -> Dict[str, Any]:
        """
        Get comprehensive decision support frameworks following DSS documentation
        
        Returns:
            Dictionary containing various decision support frameworks
        """
        return self.decision_engine.get_decision_support_frameworks()
    
    def get_clinical_guidelines_integration(self) -> Dict[str, Any]:
        """
        Get clinical guidelines integration following Healthcare_DSS_Architecture.md
        
        Returns:
            Dictionary containing clinical guidelines and evidence-based protocols
        """
        return self.guidelines_manager.get_clinical_guidelines_integration()
    
    def get_evidence_based_protocols(self, condition: str = "diabetes") -> Dict[str, Any]:
        """
        Get evidence-based protocols for specific conditions
        
        Args:
            condition: Medical condition (diabetes, hypertension, cardiac, etc.)
            
        Returns:
            Dictionary containing evidence-based protocols
        """
        return self.guidelines_manager.get_evidence_based_protocols(condition)
    
    def get_expert_system_architecture(self) -> Dict[str, Any]:
        """
        Get expert system architecture following Healthcare_DSS_Architecture.md
        
        Returns:
            Dictionary containing expert system components and architecture
        """
        expert_system = {
            'knowledge_base': {
                'clinical_rules': {
                    'diagnostic_rules': 'IF-THEN statements for diagnosis',
                    'treatment_rules': 'IF-THEN statements for treatment',
                    'monitoring_rules': 'IF-THEN statements for monitoring',
                    'alert_rules': 'IF-THEN statements for alerts'
                },
                'clinical_facts': {
                    'drug_interactions': 'Database of medication interactions',
                    'contraindications': 'Database of contraindications',
                    'dosage_guidelines': 'Standard dosing recommendations',
                    'side_effects': 'Known side effects and monitoring'
                },
                'clinical_protocols': {
                    'emergency_protocols': 'Critical care protocols',
                    'routine_protocols': 'Standard care protocols',
                    'specialty_protocols': 'Specialty-specific protocols'
                }
            },
            'inference_engine': {
                'forward_chaining': 'Data-driven reasoning',
                'backward_chaining': 'Goal-driven reasoning',
                'uncertainty_handling': 'Fuzzy logic for uncertain situations',
                'conflict_resolution': 'Rule priority and conflict resolution'
            },
            'explanation_facility': {
                'rule_tracing': 'Show which rules were applied',
                'reasoning_path': 'Display reasoning process',
                'evidence_citation': 'Cite clinical evidence',
                'confidence_scores': 'Show confidence levels'
            },
            'knowledge_acquisition': {
                'expert_interviews': 'Structured expert knowledge capture',
                'literature_mining': 'Automated literature analysis',
                'case_based_reasoning': 'Learn from clinical cases',
                'feedback_integration': 'Continuous learning from outcomes'
            }
        }
        
        return expert_system
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents"""
        return {
            'clinical_rules': len(self.clinical_rules),
            'clinical_guidelines': len(self.clinical_guidelines),
            'decision_trees': len(self.decision_engine.decision_trees),
            'knowledge_relationships': len(self.knowledge_graph),
            'active_rules': sum(1 for rule in self.clinical_rules.values() if rule.active)
        }
    
    def get_intelligent_recommendations(self, patient_data: Dict[str, Any], 
                                      context: str = None) -> List[Dict[str, Any]]:
        """
        Get intelligent, context-aware clinical recommendations
        
        Args:
            patient_data: Dictionary containing patient information
            context: Optional context (e.g., 'diagnosis', 'treatment', 'prevention')
            
        Returns:
            List of intelligent clinical recommendations with confidence scores
        """
        recommendations = []
        
        # Analyze patient data to determine relevant conditions
        relevant_conditions = self._analyze_patient_conditions(patient_data)
        
        # Get base recommendations
        base_recommendations = self.get_clinical_recommendations(patient_data)
        
        # Enhance with intelligent analysis
        for rec in base_recommendations:
            enhanced_rec = self._enhance_recommendation(rec, patient_data, relevant_conditions, context)
            recommendations.append(enhanced_rec)
        
        # Add intelligent insights based on data patterns
        insights = self._generate_intelligent_insights(patient_data, relevant_conditions)
        recommendations.extend(insights)
        
        # Sort by relevance and confidence
        recommendations.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
        
        return recommendations[:10]  # Return top 10 most relevant recommendations
    
    def search_knowledge(self, query: str, search_type: str = "all") -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant information
        
        Args:
            query: Search query
            search_type: Type of search ("rules", "guidelines", "all")
            
        Returns:
            List of matching knowledge items
        """
        results = []
        
        if search_type in ["rules", "all"]:
            rules_results = self.db_manager.search_rules(query)
            results.extend(rules_results)
        
        if search_type in ["guidelines", "all"]:
            guidelines_results = self.guidelines_manager.search_guidelines(query)
            results.extend(guidelines_results)
        
        return results
    
    def get_clinical_rules(self) -> List[Dict[str, Any]]:
        """
        Get all clinical rules from the knowledge base
        
        Returns:
            List of clinical rules with their details
        """
        rules = []
        for rule_id, rule in self.clinical_rules.items():
            rules.append({
                'rule_id': rule_id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type.value,
                'severity': rule.severity.value,
                'evidence_level': rule.evidence_level,
                'active': rule.active,
                'created_at': rule.created_at.isoformat(),
                'updated_at': rule.updated_at.isoformat()
            })
        return rules
    
    def _analyze_patient_conditions(self, patient_data: Dict[str, Any]) -> List[str]:
        """Analyze patient data to identify relevant medical conditions"""
        conditions = []
        
        # Age-based conditions
        age = patient_data.get('age', 0)
        if age >= 65:
            conditions.extend(['elderly_care', 'geriatric_assessment'])
        elif age >= 50:
            conditions.extend(['middle_age_screening', 'chronic_disease_monitoring'])
        elif age >= 18:
            conditions.extend(['adult_care', 'preventive_screening'])
        else:
            conditions.extend(['pediatric_care', 'developmental_monitoring'])
        
        # BMI-based conditions
        bmi = patient_data.get('bmi', 0)
        if bmi >= 30:
            conditions.extend(['obesity', 'metabolic_syndrome'])
        elif bmi >= 25:
            conditions.extend(['overweight', 'weight_management'])
        elif bmi < 18.5:
            conditions.extend(['underweight', 'nutritional_assessment'])
        
        # Blood pressure conditions
        systolic_bp = patient_data.get('systolic_bp', 0)
        diastolic_bp = patient_data.get('diastolic_bp', 0)
        if systolic_bp >= 140 or diastolic_bp >= 90:
            conditions.extend(['hypertension', 'cardiovascular_risk'])
        elif systolic_bp >= 120 or diastolic_bp >= 80:
            conditions.extend(['prehypertension', 'cardiovascular_monitoring'])
        
        # Diabetes-related conditions
        hba1c = patient_data.get('hba1c', 0)
        if hba1c >= 6.5:
            conditions.extend(['diabetes', 'diabetic_care'])
        elif hba1c >= 5.7:
            conditions.extend(['prediabetes', 'diabetes_prevention'])
        
        # Family history conditions
        if patient_data.get('family_history', False):
            conditions.extend(['genetic_predisposition', 'family_history_monitoring'])
        
        return list(set(conditions))  # Remove duplicates
    
    def _enhance_recommendation(self, recommendation: Dict[str, Any], 
                               patient_data: Dict[str, Any], 
                               conditions: List[str], 
                               context: str = None) -> Dict[str, Any]:
        """Enhance recommendation with intelligent analysis"""
        enhanced = recommendation.copy()
        
        # Calculate confidence score based on patient data relevance
        confidence = self._calculate_recommendation_confidence(recommendation, patient_data, conditions)
        enhanced['confidence'] = confidence
        
        # Add personalized reasoning
        reasoning = self._generate_recommendation_reasoning(recommendation, patient_data, conditions)
        enhanced['reasoning'] = reasoning
        
        # Add priority based on severity and urgency
        priority = self._calculate_recommendation_priority(recommendation, patient_data)
        enhanced['priority'] = priority
        
        return enhanced
    
    def _calculate_recommendation_confidence(self, recommendation: Dict[str, Any], 
                                           patient_data: Dict[str, Any], 
                                           conditions: List[str]) -> float:
        """Calculate confidence score for a recommendation"""
        try:
            confidence = 0.5  # Base confidence
            
            # Evidence level boost
            evidence_level = recommendation.get('evidence_level', 'C')
            evidence_scores = {'A': 0.3, 'B': 0.2, 'C': 0.1, 'D': 0.05}
            confidence += evidence_scores.get(evidence_level, 0.1)
            
            # Patient data relevance
            if 'age' in patient_data and 'age' in str(recommendation.get('recommendation', '')):
                confidence += 0.1
            
            if 'bmi' in patient_data and 'weight' in str(recommendation.get('recommendation', '')).lower():
                confidence += 0.1
            
            if 'hba1c' in patient_data and 'diabetes' in str(recommendation.get('recommendation', '')).lower():
                confidence += 0.15
            
            return min(1.0, max(0.0, confidence))  # Clamp between 0-1
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_recommendation_reasoning(self, recommendation: Dict[str, Any], 
                                         patient_data: Dict[str, Any], 
                                         conditions: List[str]) -> str:
        """Generate reasoning for why this recommendation is relevant"""
        try:
            reasoning_parts = []
            
            # Evidence-based reasoning
            evidence_level = recommendation.get('evidence_level', 'C')
            if evidence_level == 'A':
                reasoning_parts.append("Strong evidence supports this recommendation")
            elif evidence_level == 'B':
                reasoning_parts.append("Moderate evidence supports this recommendation")
            
            # Patient-specific reasoning
            age = patient_data.get('age', 0)
            if age >= 65 and 'geriatric' in str(recommendation.get('recommendation', '')).lower():
                reasoning_parts.append("Age-appropriate for elderly patients")
            
            bmi = patient_data.get('bmi', 0)
            if bmi >= 30 and 'weight' in str(recommendation.get('recommendation', '')).lower():
                reasoning_parts.append("Relevant for patients with obesity")
            
            return ". ".join(reasoning_parts) if reasoning_parts else "Based on clinical guidelines"
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Based on clinical guidelines"
    
    def _calculate_recommendation_priority(self, recommendation: Dict[str, Any], 
                                         patient_data: Dict[str, Any]) -> str:
        """Calculate priority level for recommendation"""
        try:
            # High priority indicators
            high_priority_keywords = ['urgent', 'immediate', 'critical', 'emergency', 'alert']
            rec_text = str(recommendation.get('recommendation', '')).lower()
            
            if any(keyword in rec_text for keyword in high_priority_keywords):
                return 'high'
            
            # Check patient data for urgency indicators
            systolic_bp = patient_data.get('systolic_bp', 0)
            diastolic_bp = patient_data.get('diastolic_bp', 0)
            if systolic_bp >= 180 or diastolic_bp >= 110:
                return 'high'
            
            hba1c = patient_data.get('hba1c', 0)
            if hba1c >= 10:
                return 'high'
            
            return 'medium'  # Default priority
            
        except Exception as e:
            logger.error(f"Error calculating priority: {e}")
            return 'medium'
    
    def _generate_intelligent_insights(self, patient_data: Dict[str, Any], 
                                     conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate intelligent insights based on patient data patterns"""
        insights = []
        
        try:
            # Risk stratification insights
            risk_factors = self._identify_risk_factors(patient_data)
            if risk_factors:
                insights.append({
                    'type': 'risk_assessment',
                    'title': 'Risk Factor Analysis',
                    'recommendation': f"Patient has {len(risk_factors)} risk factors: {', '.join(risk_factors)}",
                    'confidence': 0.8,
                    'priority': 'medium',
                    'reasoning': 'Based on clinical data analysis',
                    'evidence_level': 'B'
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors from patient data"""
        risk_factors = []
        
        age = patient_data.get('age', 0)
        if age >= 65:
            risk_factors.append('advanced age')
        
        bmi = patient_data.get('bmi', 0)
        if bmi >= 30:
            risk_factors.append('obesity')
        elif bmi >= 25:
            risk_factors.append('overweight')
        
        systolic_bp = patient_data.get('systolic_bp', 0)
        diastolic_bp = patient_data.get('diastolic_bp', 0)
        if systolic_bp >= 140 or diastolic_bp >= 90:
            risk_factors.append('hypertension')
        
        hba1c = patient_data.get('hba1c', 0)
        if hba1c >= 6.5:
            risk_factors.append('diabetes')
        elif hba1c >= 5.7:
            risk_factors.append('prediabetes')
        
        if patient_data.get('family_history', False):
            risk_factors.append('family history')
        
        return risk_factors
    
    def close_connection(self):
        """Close knowledge base connection"""
        self.db_manager.close_connection()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close_connection()
        except Exception:
            pass
