"""
Knowledge Base Database Manager for Healthcare DSS
=================================================

This module handles all database operations for the knowledge management system:
- Database initialization and table creation
- CRUD operations for clinical rules and guidelines
- Knowledge graph management
- Database connection management
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from healthcare_dss.core.knowledge_models import ClinicalRule, ClinicalGuideline, KnowledgeRelationship

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeDatabaseManager:
    """
    Manages database operations for the knowledge management system
    """
    
    def __init__(self, knowledge_db_path: str = "knowledge_base.db"):
        """
        Initialize Knowledge Database Manager
        
        Args:
            knowledge_db_path: Path to knowledge base database
        """
        self.knowledge_db_path = knowledge_db_path
        self.connection = None
        self._init_knowledge_base()
    
    def _init_knowledge_base(self):
        """Initialize knowledge base database"""
        try:
            self.connection = sqlite3.connect(self.knowledge_db_path)
            self._create_knowledge_tables()
            logger.info(f"Knowledge base initialized at {self.knowledge_db_path}")
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            raise
    
    def _create_knowledge_tables(self):
        """Create knowledge base tables"""
        cursor = self.connection.cursor()
        
        # Clinical rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                rule_type TEXT,
                conditions TEXT,
                actions TEXT,
                severity TEXT,
                evidence_level TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                active BOOLEAN DEFAULT 1
            )
        """)
        
        # Clinical guidelines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_guidelines (
                guideline_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                conditions TEXT,
                recommendations TEXT,
                evidence_level TEXT,
                source TEXT,
                version TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Decision trees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_trees (
                tree_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tree_structure TEXT,
                conditions TEXT,
                outcomes TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Knowledge graph table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                relationship TEXT,
                target_entity TEXT,
                confidence REAL,
                source TEXT,
                created_at TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("Knowledge base tables created successfully")
    
    def add_clinical_rule(self, rule: ClinicalRule):
        """Add a clinical rule to the knowledge base"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO clinical_rules 
            (rule_id, name, description, rule_type, conditions, actions, severity, 
             evidence_level, created_at, updated_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.name, rule.description, rule.rule_type.value,
            json.dumps(rule.conditions), json.dumps(rule.actions), rule.severity.value,
            rule.evidence_level, rule.created_at.isoformat(), rule.updated_at.isoformat(),
            rule.active
        ))
        self.connection.commit()
        logger.info(f"Added clinical rule: {rule.name}")
    
    def add_clinical_guideline(self, guideline: ClinicalGuideline):
        """Add a clinical guideline to the knowledge base"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO clinical_guidelines 
            (guideline_id, title, description, category, conditions, recommendations, 
             evidence_level, source, version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            guideline.guideline_id, guideline.title, guideline.description,
            guideline.category, json.dumps(guideline.conditions),
            json.dumps(guideline.recommendations), guideline.evidence_level,
            guideline.source, guideline.version, guideline.created_at.isoformat()
        ))
        self.connection.commit()
        logger.info(f"Added clinical guideline: {guideline.title}")
    
    def add_knowledge_relationship(self, source: str, relationship: str, target: str, 
                                 confidence: float, source_ref: str):
        """Add a relationship to the knowledge graph"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO knowledge_graph 
            (source_entity, relationship, target_entity, confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source, relationship, target, confidence, source_ref, datetime.now().isoformat()))
        self.connection.commit()
    
    def get_clinical_rules(self) -> List[Dict[str, Any]]:
        """Get all clinical rules from database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM clinical_rules WHERE active = 1")
        rows = cursor.fetchall()
        
        rules = []
        for row in rows:
            rules.append({
                'rule_id': row[0],
                'name': row[1],
                'description': row[2],
                'rule_type': row[3],
                'conditions': json.loads(row[4]) if row[4] else {},
                'actions': json.loads(row[5]) if row[5] else [],
                'severity': row[6],
                'evidence_level': row[7],
                'created_at': row[8],
                'updated_at': row[9],
                'active': bool(row[10])
            })
        return rules
    
    def get_clinical_guidelines(self) -> List[Dict[str, Any]]:
        """Get all clinical guidelines from database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM clinical_guidelines")
        rows = cursor.fetchall()
        
        guidelines = []
        for row in rows:
            guidelines.append({
                'guideline_id': row[0],
                'title': row[1],
                'description': row[2],
                'category': row[3],
                'conditions': json.loads(row[4]) if row[4] else [],
                'recommendations': json.loads(row[5]) if row[5] else [],
                'evidence_level': row[6],
                'source': row[7],
                'version': row[8],
                'created_at': row[9]
            })
        return guidelines
    
    def get_knowledge_relationships(self) -> List[Dict[str, Any]]:
        """Get all knowledge relationships from database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM knowledge_graph")
        rows = cursor.fetchall()
        
        relationships = []
        for row in rows:
            relationships.append({
                'id': row[0],
                'source_entity': row[1],
                'relationship': row[2],
                'target_entity': row[3],
                'confidence': row[4],
                'source': row[5],
                'created_at': row[6]
            })
        return relationships
    
    def update_clinical_rule(self, rule_id: str, updates: Dict[str, Any]):
        """Update a clinical rule in the database"""
        cursor = self.connection.cursor()
        
        # Build update query dynamically
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ['conditions', 'actions']:
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == 'updated_at':
                set_clauses.append(f"{key} = ?")
                values.append(datetime.now().isoformat())
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        if set_clauses:
            query = f"UPDATE clinical_rules SET {', '.join(set_clauses)} WHERE rule_id = ?"
            values.append(rule_id)
            cursor.execute(query, values)
            self.connection.commit()
            logger.info(f"Updated clinical rule: {rule_id}")
    
    def delete_clinical_rule(self, rule_id: str):
        """Delete a clinical rule from the database"""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM clinical_rules WHERE rule_id = ?", (rule_id,))
        self.connection.commit()
        logger.info(f"Deleted clinical rule: {rule_id}")
    
    def search_rules(self, query: str) -> List[Dict[str, Any]]:
        """Search clinical rules by name or description"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM clinical_rules 
            WHERE (name LIKE ? OR description LIKE ?) AND active = 1
        """, (f"%{query}%", f"%{query}%"))
        
        rows = cursor.fetchall()
        rules = []
        for row in rows:
            rules.append({
                'rule_id': row[0],
                'name': row[1],
                'description': row[2],
                'rule_type': row[3],
                'severity': row[6],
                'evidence_level': row[7]
            })
        return rules
    
    def search_guidelines(self, query: str) -> List[Dict[str, Any]]:
        """Search clinical guidelines by title or description"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM clinical_guidelines 
            WHERE title LIKE ? OR description LIKE ?
        """, (f"%{query}%", f"%{query}%"))
        
        rows = cursor.fetchall()
        guidelines = []
        for row in rows:
            guidelines.append({
                'guideline_id': row[0],
                'title': row[1],
                'description': row[2],
                'category': row[3],
                'evidence_level': row[6],
                'source': row[7]
            })
        return guidelines
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        cursor = self.connection.cursor()
        
        stats = {}
        
        # Count clinical rules
        cursor.execute("SELECT COUNT(*) FROM clinical_rules WHERE active = 1")
        stats['active_rules'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM clinical_rules")
        stats['total_rules'] = cursor.fetchone()[0]
        
        # Count clinical guidelines
        cursor.execute("SELECT COUNT(*) FROM clinical_guidelines")
        stats['guidelines'] = cursor.fetchone()[0]
        
        # Count knowledge relationships
        cursor.execute("SELECT COUNT(*) FROM knowledge_graph")
        stats['relationships'] = cursor.fetchone()[0]
        
        return stats
    
    def close_connection(self):
        """Close knowledge base connection"""
        if self.connection:
            self.connection.close()
            if (hasattr(logger, 'handlers') and logger.handlers and 
                not getattr(self, '_shutdown', False)):
                try:
                    logger.info("Knowledge base connection closed")
                except Exception:
                    pass
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self._shutdown = True
            self.close_connection()
        except Exception:
            pass
