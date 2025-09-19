"""
Enhanced Debug System for Healthcare DSS
========================================

Provides comprehensive debugging capabilities with detailed system information,
performance metrics, database queries, model training logs, and real-time monitoring.
All configuration is externalized to avoid hardcoded values.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback
import sys
import os
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
import threading
from collections import deque

from healthcare_dss.config.dashboard_config import config_manager

logger = logging.getLogger(__name__)

class DebugManager:
    """Enhanced debug management with comprehensive system monitoring"""
    
    def __init__(self):
        self.debug_log = deque(maxlen=1000)  # Circular buffer for performance
        self.performance_metrics = {}
        self.query_log = deque(maxlen=500)
        self.model_training_log = deque(maxlen=200)
        self.start_time = time.time()
        
        # Load debug configuration
        self.debug_config = self._load_debug_config()
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is currently enabled"""
        return st.session_state.get('debug_mode', False)
        
    def _load_debug_config(self) -> Dict[str, Any]:
        """Load debug configuration from environment variables"""
        return {
            "show_performance_metrics": os.getenv('DSS_DEBUG_PERFORMANCE', 'true').lower() == 'true',
            "show_database_queries": os.getenv('DSS_DEBUG_DB_QUERIES', 'true').lower() == 'true',
            "show_model_training": os.getenv('DSS_DEBUG_MODEL_TRAINING', 'true').lower() == 'true',
            "show_system_monitoring": os.getenv('DSS_DEBUG_SYSTEM_MONITOR', 'true').lower() == 'true',
            "show_session_state": os.getenv('DSS_DEBUG_SESSION', 'true').lower() == 'true',
            "show_configuration": os.getenv('DSS_DEBUG_CONFIG', 'true').lower() == 'true',
            "max_log_entries": int(os.getenv('DSS_DEBUG_MAX_LOG', '1000')),
            "refresh_interval": int(os.getenv('DSS_DEBUG_REFRESH', '5')),  # seconds
            "detailed_errors": os.getenv('DSS_DEBUG_DETAILED_ERRORS', 'true').lower() == 'true'
        }
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        return st.session_state.get('debug_mode', False)
    
    def enable_debug(self):
        """Enable debug mode"""
        st.session_state.debug_mode = True
        self.log_debug("Debug mode enabled", "SYSTEM")
    
    def disable_debug(self):
        """Disable debug mode"""
        st.session_state.debug_mode = False
        self.log_debug("Debug mode disabled", "SYSTEM")
    
    def log_debug(self, message: str, category: str = "INFO", data: Any = None):
        """Log debug message with optional data"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "category": category,
            "message": message,
            "data": data,
            "thread": threading.current_thread().name
        }
        self.debug_log.append(log_entry)
        
        if self.is_debug_enabled():
            # Color coding for different categories
            color_map = {
                "ERROR": "🔴",
                "WARNING": "🟡", 
                "INFO": "🔵",
                "SYSTEM": "🟢",
                "PERFORMANCE": "🟣",
                "DATABASE": "🔵",
                "MODEL": "🟠"
            }
            icon = color_map.get(category, "ℹ️")
            st.write(f"{icon} **DEBUG [{category}]**: {message}")
            
            if data and self.debug_config["detailed_errors"]:
                with st.expander("Debug Data", expanded=False):
                    st.json(data)
    
    def log_database_query(self, query: str, execution_time: float, result_count: int = None):
        """Log database query with performance metrics"""
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "execution_time": execution_time,
            "result_count": result_count
        }
        self.query_log.append(query_entry)
        self.log_debug(f"DB Query executed in {execution_time:.3f}s", "DATABASE", query_entry)
    
    def log_model_training(self, model_name: str, dataset: str, metrics: Dict[str, float], training_time: float):
        """Log model training information"""
        training_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "dataset": dataset,
            "metrics": metrics,
            "training_time": training_time
        }
        self.model_training_log.append(training_entry)
        self.log_debug(f"Model {model_name} trained in {training_time:.2f}s", "MODEL", training_entry)
    
    def update_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Update performance metric"""
        self.performance_metrics[metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
        self.log_debug(f"Performance metric updated: {metric_name} = {value} {unit}", "PERFORMANCE")
    
    def _render_system_status(self):
        """Render system status overview"""
        with st.sidebar.expander("System Status", expanded=True):
            # Initialization status
            init_status = st.session_state.get('initialized', False)
            st.sidebar.write(f"**Status:** {'✅ Ready' if init_status else '❌ Not Ready'}")
            
            # Component status
            components = {
                'Data Manager': st.session_state.get('data_manager'),
                'Model Manager': st.session_state.get('model_manager'),
                'Knowledge Manager': st.session_state.get('knowledge_manager'),
                'Preprocessing Engine': st.session_state.get('preprocessing_engine')
            }
            
            for component_name, component in components.items():
                status = "✅" if component else "❌"
                st.sidebar.write(f"{status} {component_name}")
            
            # Uptime
            uptime = time.time() - self.start_time
            st.sidebar.write(f"**Uptime:** {uptime:.1f}s")
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        with st.sidebar.expander("Performance Metrics"):
            if self.performance_metrics:
                for metric_name, metric_data in self.performance_metrics.items():
                    st.sidebar.metric(
                        metric_name,
                        f"{metric_data['value']:.2f}",
                        help=f"Last updated: {metric_data['timestamp']}"
                    )
            else:
                st.sidebar.info("No performance metrics recorded")
    
    def _render_database_queries(self):
        """Render database query log"""
        with st.sidebar.expander("Database Queries"):
            if self.query_log:
                # Show recent queries
                recent_queries = list(self.query_log)[-5:]  # Last 5 queries
                for query_entry in recent_queries:
                    st.sidebar.write(f"**{query_entry['timestamp'][:8]}**")
                    st.sidebar.write(f"Time: {query_entry['execution_time']:.3f}s")
                    if query_entry['result_count']:
                        st.sidebar.write(f"Results: {query_entry['result_count']}")
                    st.sidebar.write(f"Query: {query_entry['query'][:50]}...")
                    st.sidebar.write("---")
            else:
                st.sidebar.info("No database queries logged")
    
    def _render_model_training_log(self):
        """Render model training log"""
        with st.sidebar.expander("Model Training"):
            if self.model_training_log:
                recent_training = list(self.model_training_log)[-3:]  # Last 3 training sessions
                for training_entry in recent_training:
                    st.sidebar.write(f"**{training_entry['model_name']}**")
                    st.sidebar.write(f"Dataset: {training_entry['dataset']}")
                    st.sidebar.write(f"Time: {training_entry['training_time']:.2f}s")
                    if training_entry['metrics']:
                        for metric, value in training_entry['metrics'].items():
                            st.sidebar.write(f"{metric}: {value:.3f}")
                    st.sidebar.write("---")
            else:
                st.sidebar.info("No model training logged")
    
    def _render_system_monitoring(self):
        """Render system monitoring information"""
        with st.sidebar.expander("System Monitoring"):
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                st.sidebar.metric("Memory Usage", f"{memory.percent:.1f}%")
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                st.sidebar.metric("CPU Usage", f"{cpu_percent:.1f}%")
                
                # Disk usage
                disk = psutil.disk_usage('/')
                st.sidebar.metric("Disk Usage", f"{disk.percent:.1f}%")
                
                # Process info
                process = psutil.Process()
                st.sidebar.metric("Process Memory", f"{process.memory_info().rss / 1024 / 1024:.1f} MB")
                
            except Exception as e:
                st.sidebar.error(f"Monitoring error: {str(e)}")
    
    def _render_session_state(self):
        """Render session state information"""
        with st.sidebar.expander("Session State"):
            session_data = {}
            for key, value in st.session_state.items():
                session_data[key] = {
                    "type": type(value).__name__,
                    "size": len(str(value)) if hasattr(value, '__len__') else "N/A"
                }
            
            st.sidebar.json(session_data)
    
    def _render_configuration(self):
        """Render configuration information"""
        with st.sidebar.expander("Configuration"):
            system_config = config_manager.get_system_config()
            debug_config = self.debug_config
            
            config_data = {
                "system": system_config,
                "debug": debug_config
            }
            
            st.sidebar.json(config_data)
    
    def _render_debug_controls(self):
        """Render debug controls"""
        with st.sidebar.expander("Debug Controls"):
            if st.sidebar.button("Clear Logs"):
                self.debug_log.clear()
                self.query_log.clear()
                self.model_training_log.clear()
                st.sidebar.success("Logs cleared")
            
            if st.sidebar.button("Export Logs"):
                self._export_logs()
            
            if st.sidebar.button("Refresh Metrics"):
                self._refresh_metrics()
                st.sidebar.success("Metrics refreshed")
    
    def _export_logs(self):
        """Export debug logs to file"""
        try:
            export_data = {
                "debug_log": list(self.debug_log),
                "query_log": list(self.query_log),
                "model_training_log": list(self.model_training_log),
                "performance_metrics": self.performance_metrics,
                "export_time": datetime.now().isoformat()
            }
            
            export_path = f"debug_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            st.sidebar.success(f"Logs exported to {export_path}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")
    
    def _refresh_metrics(self):
        """Refresh performance metrics"""
        try:
            # Update system metrics
            memory = psutil.virtual_memory()
            self.update_performance_metric("Memory Usage", memory.percent, "%")
            
            cpu_percent = psutil.cpu_percent()
            self.update_performance_metric("CPU Usage", cpu_percent, "%")
            
            # Update application metrics
            if st.session_state.get('data_manager'):
                total_datasets = len(st.session_state.data_manager.datasets)
                self.update_performance_metric("Total Datasets", total_datasets)
                
                total_records = sum(len(df) for df in st.session_state.data_manager.datasets.values())
                self.update_performance_metric("Total Records", total_records)
            
        except Exception as e:
            self.log_debug(f"Error refreshing metrics: {str(e)}", "ERROR")
    
    def render_page_debug_info(self, page_name: str, additional_data: Dict[str, Any] = None):
        """Render page-specific debug information"""
        if not self.is_debug_enabled():
            return
            
        st.markdown("---")
        st.subheader("🔍 Page Debug Information")
        
        debug_data = {
            "Page Name": page_name,
            "Function Called": f"show_{page_name.lower().replace(' ', '_')}()",
            "Session State Initialized": st.session_state.get('initialized', False),
            "Debug Mode": self.is_debug_enabled(),
            "Current Time": datetime.now().isoformat(),
            "User Role": st.session_state.get('user_role', 'Not Set')
        }
        
        if additional_data:
            debug_data.update(additional_data)
        
        with st.expander("Page Debug Details", expanded=False):
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
    
    def get_page_debug_data(self, page_name: str, additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get page debug data without rendering (for use inside custom expanders)"""
        debug_data = {
            "Page Name": page_name,
            "Function Called": f"show_{page_name.lower().replace(' ', '_')}()",
            "Session State Initialized": st.session_state.get('initialized', False),
            "Debug Mode": self.is_debug_enabled(),
            "Current Time": datetime.now().isoformat(),
            "User Role": st.session_state.get('user_role', 'Not Set')
        }
        
        if additional_data:
            debug_data.update(additional_data)
            
        return debug_data
    
    def render_error_debug(self, error: Exception, context: str = ""):
        """Render comprehensive error debugging information"""
        if not self.is_debug_enabled():
            return
            
        st.error("🚨 Error Detected")
        
        error_data = {
            "Error Type": type(error).__name__,
            "Error Message": str(error),
            "Context": context,
            "Timestamp": datetime.now().isoformat(),
            "Traceback": traceback.format_exc()
        }
        
        with st.expander("Error Debug Information", expanded=True):
            for key, value in error_data.items():
                if key == "Traceback":
                    st.code(value, language="python")
                else:
                    st.write(f"**{key}:** {value}")
        
        # Log the error
        self.log_debug(f"Error in {context}: {str(error)}", "ERROR", error_data)
    
    def show_system_status(self, title="System Status"):
        """Show comprehensive system status"""
        if not self.is_debug_enabled():
            return
            
        debug_data = {
            "Session State Initialized": st.session_state.get('initialized', False),
            "Debug Mode": self.is_debug_enabled(),
            "Has Data Manager": hasattr(st.session_state, 'data_manager'),
            "Has Model Manager": hasattr(st.session_state, 'model_manager'),
            "Has Knowledge Manager": hasattr(st.session_state, 'knowledge_manager'),
            "Python Version": sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "Current Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager is not None:
            debug_data["Data Manager Datasets"] = len(st.session_state.data_manager.datasets)
        else:
            debug_data["Data Manager"] = "Not available"
        
        if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager is not None:
            try:
                models_df = st.session_state.model_manager.list_models()
                debug_data["Model Manager Models"] = len(models_df)
                debug_data["Models DataFrame Empty"] = models_df.empty
            except Exception as e:
                debug_data["Model Manager Error"] = str(e)
        else:
            debug_data["Model Manager"] = "Not available"
        
        self.show_debug_info(title, debug_data, expanded=True)
    
    def show_page_debug(self, page_name: str, additional_data: Dict[str, Any] = None):
        """Show page-specific debug information"""
        if not self.is_debug_enabled():
            return
            
        debug_data = {
            "Page Name": page_name,
            "Function Called": f"show_{page_name.lower().replace(' ', '_')}()",
            "Session State Initialized": st.session_state.get('initialized', False),
            "Debug Mode": self.is_debug_enabled()
        }
        
        if additional_data:
            debug_data.update(additional_data)
        
        self.show_debug_info(f"{page_name} Debug", debug_data, expanded=True)
    
    def show_error_debug(self, error: Exception, context: str = ""):
        """Show error debugging information"""
        if not self.is_debug_enabled():
            return
            
        error_data = {
            "Error Type": type(error).__name__,
            "Error Message": str(error),
            "Context": context,
            "Timestamp": datetime.now().isoformat(),
            "Traceback": traceback.format_exc()
        }
        
        with st.expander("Error Debug Information", expanded=True):
            for key, value in error_data.items():
                if key == "Traceback":
                    st.code(value, language="python")
                else:
                    st.write(f"**{key}:** {value}")
        
        # Log the error
        self.log_debug(f"Error in {context}: {str(error)}", "ERROR", error_data)
    
    def get_debug_log(self):
        """Get the debug log"""
        return list(self.debug_log)
    
    def clear_debug_log(self):
        """Clear the debug log"""
        self.debug_log.clear()
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debug summary for reporting"""
        return {
            "debug_mode": self.is_debug_enabled(),
            "log_entries": len(self.debug_log),
            "query_count": len(self.query_log),
            "training_sessions": len(self.model_training_log),
            "performance_metrics": len(self.performance_metrics),
            "uptime": time.time() - self.start_time,
            "configuration": self.debug_config
        }


# Global debug manager instance
debug_manager = DebugManager()


def debug_write(message: str, category: str = "INFO", data: Any = None):
    """Write debug message using debug manager"""
    debug_manager.log_debug(message, category, data)


def show_debug_info(title: str, debug_data: Dict[str, Any], expanded: bool = None):
    """Show debug information using debug manager"""
    if debug_manager.is_debug_enabled():
        if expanded is None:
            expanded = debug_manager.debug_mode
        
        with st.expander(f"{title}", expanded=expanded):
            for key, value in debug_data.items():
                st.write(f"**{key}:** {value}")
            
            st.write(f"**Debug Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_database_query(query: str, execution_time: float, result_count: int = None):
    """Log database query with performance metrics"""
    debug_manager.log_database_query(query, execution_time, result_count)


def log_model_training(model_name: str, dataset: str, metrics: Dict[str, float], training_time: float):
    """Log model training information"""
    debug_manager.log_model_training(model_name, dataset, metrics, training_time)


def update_performance_metric(metric_name: str, value: float, unit: str = ""):
    """Update performance metric"""
    debug_manager.update_performance_metric(metric_name, value, unit)
