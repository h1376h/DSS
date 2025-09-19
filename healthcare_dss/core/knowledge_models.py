"""
Knowledge Management Data Models for Healthcare DSS
==================================================

This module contains the data models and enums used in the knowledge management system:
- Rule types and severity levels
- Clinical rules and guidelines data structures
- Knowledge representation models
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any


class RuleType(Enum):
    """Types of clinical rules"""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    PREVENTIVE = "preventive"
    MONITORING = "monitoring"
    ALERT = "alert"


class SeverityLevel(Enum):
    """Severity levels for clinical conditions"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClinicalRule:
    """Clinical decision support rule"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    conditions: Dict[str, Any]
    actions: List[str]
    severity: SeverityLevel
    evidence_level: str
    created_at: datetime
    updated_at: datetime
    active: bool = True


@dataclass
class ClinicalGuideline:
    """Clinical guideline or protocol"""
    guideline_id: str
    title: str
    description: str
    category: str
    conditions: List[str]
    recommendations: List[str]
    evidence_level: str
    source: str
    version: str
    created_at: datetime


@dataclass
class DecisionTreeNode:
    """Node in a decision tree"""
    condition: str
    operator: str
    threshold: Any
    true_branch: Dict[str, Any] = None
    false_branch: Dict[str, Any] = None
    outcome: str = None
    recommendations: List[str] = None


@dataclass
class KnowledgeRelationship:
    """Relationship in the knowledge graph"""
    source_entity: str
    relationship: str
    target_entity: str
    confidence: float
    source: str
    created_at: datetime


@dataclass
class ClinicalRecommendation:
    """Clinical recommendation with metadata"""
    recommendation_id: str
    title: str
    recommendation: str
    evidence_level: str
    source: str
    confidence: float
    priority: str
    reasoning: str
    created_at: datetime
