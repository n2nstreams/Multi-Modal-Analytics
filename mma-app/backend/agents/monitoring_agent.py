"""
Monitoring Agent for MMA Application

This module provides functionality to:
1. Track execution of the multi-agent workflow
2. Collect performance metrics
3. Handle errors and exceptions
4. Log system events
5. Provide feedback on system health
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for monitoring
class AgentMetrics(BaseModel):
    """Metrics for an individual agent's performance"""
    agent_id: str = Field(..., description="Identifier for the agent")
    agent_type: str = Field(..., description="Type of agent (voice, query, data, etc.)")
    execution_time: float = Field(..., description="Time taken for execution in seconds")
    success: bool = Field(..., description="Whether the agent executed successfully")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    memory_usage: Optional[float] = Field(None, description="Memory used during execution (MB)")
    timestamp: str = Field(..., description="When the metrics were recorded")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens processed")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class WorkflowExecution(BaseModel):
    """Record of a full workflow execution"""
    workflow_id: str = Field(..., description="Unique identifier for the workflow execution")
    session_id: str = Field(..., description="Session identifier")
    start_time: str = Field(..., description="When the workflow started")
    end_time: Optional[str] = Field(None, description="When the workflow completed")
    total_execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    status: str = Field(..., description="Current status (running, completed, failed)")
    original_query: str = Field(..., description="Original user query that triggered the workflow")
    agent_metrics: List[AgentMetrics] = Field(default_factory=list, description="Metrics for individual agents")
    error: Optional[str] = Field(None, description="Error message if workflow failed")
    result_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of workflow result")
    
class SystemHealth(BaseModel):
    """System health information"""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_space: float = Field(..., description="Available disk space percentage")
    timestamp: str = Field(..., description="When the health check was performed")
    active_sessions: int = Field(..., description="Number of active sessions")
    error_rate: float = Field(..., description="Error rate in recent executions")
    average_response_time: float = Field(..., description="Average response time in seconds")
    component_status: Dict[str, str] = Field(..., description="Status of system components")

class MonitoringEvent(BaseModel):
    """Model for monitoring events"""
    event_type: str = Field(..., description="Type of event")
    timestamp: str = Field(..., description="When the event occurred")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    severity: str = Field("info", description="Event severity (info, warning, error, critical)")
    message: str = Field(..., description="Event message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional event details")

class ErrorReport(BaseModel):
    """Detailed error report"""
    error_id: str = Field(..., description="Unique identifier for the error")
    timestamp: str = Field(..., description="When the error occurred")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    agent_id: Optional[str] = Field(None, description="Agent where the error occurred")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    state_snapshot: Optional[Dict[str, Any]] = Field(None, description="Snapshot of system state when error occurred")
    recovery_action: Optional[str] = Field(None, description="Action taken to recover")
    is_resolved: bool = Field(False, description="Whether the error has been resolved")

# In-memory stores for monitoring data
# In production, this would be stored in a database
active_workflows: Dict[str, WorkflowExecution] = {}
recent_errors: List[ErrorReport] = []
system_events: List[MonitoringEvent] = []
performance_metrics: Dict[str, List[AgentMetrics]] = {}

# File paths for persistence
LOGS_DIR = Path("../logs")
METRICS_FILE = LOGS_DIR / "metrics.jsonl"
ERRORS_FILE = LOGS_DIR / "errors.jsonl"
EVENTS_FILE = LOGS_DIR / "events.jsonl"

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Helper functions
def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix"""
    timestamp = int(time.time() * 1000)
    random_suffix = os.urandom(4).hex()
    return f"{prefix}_{timestamp}_{random_suffix}"

async def persist_data(data: Union[BaseModel, List[BaseModel]], file_path: Path, append: bool = True) -> None:
    """Persist data to file"""
    mode = "a" if append and file_path.exists() else "w"
    
    try:
        # Fix: Don't use async with for asyncio.to_thread
        file = open(file_path, mode)
        try:
            if isinstance(data, list):
                for item in data:
                    file.write(f"{item.json()}\n")
            else:
                file.write(f"{data.json()}\n")
        finally:
            file.close()
    except Exception as e:
        logger.error(f"Failed to persist data to {file_path}: {e}")

# Core monitoring functions
async def log_event(
    event_type: str,
    message: str,
    severity: str = "info",
    workflow_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> MonitoringEvent:
    """
    Log a monitoring event
    
    Args:
        event_type: Type of event
        message: Event message
        severity: Event severity
        workflow_id: Associated workflow ID
        session_id: Associated session ID
        agent_id: Associated agent ID
        details: Additional event details
        
    Returns:
        Created MonitoringEvent
    """
    event = MonitoringEvent(
        event_type=event_type,
        timestamp=get_timestamp(),
        workflow_id=workflow_id,
        session_id=session_id,
        agent_id=agent_id,
        severity=severity,
        message=message,
        details=details
    )
    
    # Log based on severity
    log_message = f"[{event.event_type}] {event.message}"
    if severity == "info":
        logger.info(log_message)
    elif severity == "warning":
        logger.warning(log_message)
    elif severity == "error":
        logger.error(log_message)
    elif severity == "critical":
        logger.critical(log_message)
    
    # Store in memory
    system_events.append(event)
    
    # Persist to file
    await persist_data(event, EVENTS_FILE)
    
    return event

async def record_agent_metrics(
    agent_id: str,
    agent_type: str,
    execution_time: float,
    success: bool,
    error: Optional[str] = None,
    workflow_id: Optional[str] = None,
    session_id: Optional[str] = None,
    memory_usage: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentMetrics:
    """
    Record metrics for an agent execution
    
    Args:
        agent_id: Identifier for the agent
        agent_type: Type of agent
        execution_time: Time taken for execution
        success: Whether execution was successful
        error: Error message if failed
        workflow_id: Associated workflow ID
        session_id: Associated session ID
        memory_usage: Memory used during execution
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        metadata: Additional metadata
        
    Returns:
        Created AgentMetrics
    """
    metrics = AgentMetrics(
        agent_id=agent_id,
        agent_type=agent_type,
        execution_time=execution_time,
        success=success,
        error=error,
        memory_usage=memory_usage,
        timestamp=get_timestamp(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata or {}
    )
    
    # Store in memory
    if agent_type not in performance_metrics:
        performance_metrics[agent_type] = []
    performance_metrics[agent_type].append(metrics)
    
    # Update workflow if workflow_id is provided
    if workflow_id and workflow_id in active_workflows:
        active_workflows[workflow_id].agent_metrics.append(metrics)
    
    # Persist to file
    await persist_data(metrics, METRICS_FILE)
    
    # Log event
    severity = "info" if success else "error"
    message = f"Agent {agent_id} ({agent_type}) execution completed in {execution_time:.2f}s"
    if not success:
        message += f" with error: {error}"
    
    await log_event(
        event_type="agent_execution",
        message=message,
        severity=severity,
        workflow_id=workflow_id,
        session_id=session_id,
        agent_id=agent_id
    )
    
    return metrics

async def start_workflow_execution(
    session_id: str,
    original_query: str,
    workflow_id: Optional[str] = None
) -> WorkflowExecution:
    """
    Start tracking a new workflow execution
    
    Args:
        session_id: Session identifier
        original_query: Original user query
        workflow_id: Optional workflow ID (generated if not provided)
        
    Returns:
        Created WorkflowExecution
    """
    if not workflow_id:
        workflow_id = generate_id("wf")
    
    execution = WorkflowExecution(
        workflow_id=workflow_id,
        session_id=session_id,
        start_time=get_timestamp(),
        status="running",
        original_query=original_query
    )
    
    # Store in memory
    active_workflows[workflow_id] = execution
    
    # Log event
    await log_event(
        event_type="workflow_start",
        message=f"Started workflow execution for query: {original_query}",
        workflow_id=workflow_id,
        session_id=session_id
    )
    
    return execution

async def end_workflow_execution(
    workflow_id: str,
    status: str = "completed",
    error: Optional[str] = None,
    result_summary: Optional[Dict[str, Any]] = None
) -> Optional[WorkflowExecution]:
    """
    End tracking of a workflow execution
    
    Args:
        workflow_id: Workflow identifier
        status: Final status (completed, failed)
        error: Error message if failed
        result_summary: Summary of results
        
    Returns:
        Updated WorkflowExecution or None if not found
    """
    if workflow_id not in active_workflows:
        await log_event(
            event_type="workflow_error",
            message=f"Attempted to end unknown workflow: {workflow_id}",
            severity="warning",
            workflow_id=workflow_id
        )
        return None
    
    execution = active_workflows[workflow_id]
    end_time = get_timestamp()
    execution.end_time = end_time
    execution.status = status
    execution.error = error
    execution.result_summary = result_summary
    
    # Calculate total execution time
    start = datetime.fromisoformat(execution.start_time)
    end = datetime.fromisoformat(end_time)
    execution.total_execution_time = (end - start).total_seconds()
    
    # Log event
    severity = "info" if status == "completed" else "error"
    message = f"Workflow {workflow_id} {status} in {execution.total_execution_time:.2f}s"
    if error:
        message += f" with error: {error}"
    
    await log_event(
        event_type="workflow_end",
        message=message,
        severity=severity,
        workflow_id=workflow_id,
        session_id=execution.session_id
    )
    
    return execution

async def report_error(
    error_type: str,
    error_message: str,
    workflow_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    stack_trace: Optional[str] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
    recovery_action: Optional[str] = None
) -> ErrorReport:
    """
    Report an error in the system
    
    Args:
        error_type: Type of error
        error_message: Error message
        workflow_id: Associated workflow ID
        session_id: Associated session ID
        agent_id: Agent where the error occurred
        stack_trace: Stack trace
        state_snapshot: Snapshot of system state
        recovery_action: Action taken to recover
        
    Returns:
        Created ErrorReport
    """
    error_id = generate_id("err")
    
    error_report = ErrorReport(
        error_id=error_id,
        timestamp=get_timestamp(),
        workflow_id=workflow_id,
        session_id=session_id,
        agent_id=agent_id,
        error_type=error_type,
        error_message=error_message,
        stack_trace=stack_trace,
        state_snapshot=state_snapshot,
        recovery_action=recovery_action,
        is_resolved=recovery_action is not None
    )
    
    # Store in memory
    recent_errors.append(error_report)
    
    # Update workflow if workflow_id is provided
    if workflow_id and workflow_id in active_workflows:
        if active_workflows[workflow_id].status == "running":
            active_workflows[workflow_id].status = "failed"
            active_workflows[workflow_id].error = error_message
    
    # Persist to file
    await persist_data(error_report, ERRORS_FILE)
    
    # Log event
    await log_event(
        event_type="system_error",
        message=f"Error: {error_message}",
        severity="error",
        workflow_id=workflow_id,
        session_id=session_id,
        agent_id=agent_id,
        details={"error_id": error_id, "error_type": error_type}
    )
    
    return error_report

async def check_system_health() -> SystemHealth:
    """
    Check current system health
    
    Returns:
        SystemHealth object with current health metrics
    """
    # In a real implementation, this would check actual system resources
    # Here we provide mock data for demonstration
    
    # Count active workflows
    active_count = sum(1 for exec in active_workflows.values() if exec.status == "running")
    
    # Calculate error rate from recent executions
    recent_metrics = []
    for metrics_list in performance_metrics.values():
        recent_metrics.extend(metrics_list[-20:] if len(metrics_list) > 20 else metrics_list)
    
    if recent_metrics:
        error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
        avg_response_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
    else:
        error_rate = 0.0
        avg_response_time = 0.0
    
    health = SystemHealth(
        cpu_usage=50.0,  # Mock value
        memory_usage=30.0,  # Mock value
        disk_space=80.0,  # Mock value
        timestamp=get_timestamp(),
        active_sessions=active_count,
        error_rate=error_rate,
        average_response_time=avg_response_time,
        component_status={
            "voice_agent": "operational",
            "query_agent": "operational",
            "data_agent": "operational",
            "visualization_agent": "operational",
            "database": "operational"
        }
    )
    
    return health

# LangGraph node functions
async def monitor_start(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to start monitoring a workflow
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state with workflow ID
    """
    session_id = state.get("session_id", generate_id("session"))
    query = state.get("natural_language_query", "")
    
    if not query:
        # If no query is provided, generate a generic workflow ID
        workflow_id = generate_id("wf")
        return {
            **state,
            "workflow_id": workflow_id,
            "session_id": session_id,
            "monitoring": {
                "start_time": get_timestamp(),
                "status": "running"
            }
        }
    
    # Start workflow tracking
    execution = await start_workflow_execution(
        session_id=session_id,
        original_query=query
    )
    
    return {
        **state,
        "workflow_id": execution.workflow_id,
        "session_id": session_id,
        "monitoring": {
            "start_time": execution.start_time,
            "status": "running"
        }
    }

async def monitor_agent_execution(state: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
    """
    LangGraph node function to monitor an agent's execution
    
    Args:
        state: Current state dictionary
        agent_type: Type of agent being monitored
        
    Returns:
        Updated state
    """
    # Extract required information
    workflow_id = state.get("workflow_id")
    session_id = state.get("session_id")
    agent_id = f"{agent_type}_{generate_id('')}"
    
    # Check if agent execution was successful
    agent_result = state.get(f"{agent_type}_result", {})
    success = agent_result.get("success", False) if isinstance(agent_result, dict) else False
    error = agent_result.get("error") if isinstance(agent_result, dict) and not success else None
    
    # Placeholder for execution time (in a real implementation, would be measured)
    execution_time = state.get("_execution_times", {}).get(agent_type, 1.0)
    
    # Record metrics
    await record_agent_metrics(
        agent_id=agent_id,
        agent_type=agent_type,
        execution_time=execution_time,
        success=success,
        error=error,
        workflow_id=workflow_id,
        session_id=session_id
    )
    
    # Update state
    return {
        **state,
        f"{agent_type}_monitoring": {
            "agent_id": agent_id,
            "success": success,
            "execution_time": execution_time,
            "timestamp": get_timestamp()
        }
    }

async def monitor_end(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to end workflow monitoring
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state
    """
    workflow_id = state.get("workflow_id")
    if not workflow_id:
        return state
    
    # Determine final status
    final_status = "completed"
    error = None
    
    # Check for errors in state
    if state.get("error"):
        final_status = "failed"
        error = state.get("error")
    
    # Get result summary
    result_summary = None
    if "visualization_result" in state:
        result_summary = {
            "visualization_type": state.get("visualization_result", {}).get("type"),
            "data_points": len(state.get("data_result", {}).get("data", [])) if state.get("data_result", {}).get("data") else 0
        }
    
    # End workflow tracking
    execution = await end_workflow_execution(
        workflow_id=workflow_id,
        status=final_status,
        error=error,
        result_summary=result_summary
    )
    
    if not execution:
        return state
    
    # Update state
    return {
        **state,
        "monitoring": {
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "total_execution_time": execution.total_execution_time,
            "status": execution.status
        }
    }

async def handle_error(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """
    LangGraph node function to handle errors during workflow execution
    
    Args:
        state: Current state dictionary
        error: Exception that occurred
        
    Returns:
        Updated state with error information
    """
    workflow_id = state.get("workflow_id")
    session_id = state.get("session_id")
    current_agent = state.get("current_agent", "unknown")
    
    # Get stack trace
    stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    
    # Report error
    error_report = await report_error(
        error_type=type(error).__name__,
        error_message=str(error),
        workflow_id=workflow_id,
        session_id=session_id,
        agent_id=current_agent,
        stack_trace=stack_trace,
        state_snapshot=state
    )
    
    # Update state
    return {
        **state,
        "error": str(error),
        "error_id": error_report.error_id,
        "error_timestamp": error_report.timestamp,
        "recovery_needed": True
    }

async def attempt_recovery(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to attempt recovery from errors
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state after recovery attempt
    """
    if not state.get("recovery_needed"):
        return state
    
    error = state.get("error", "Unknown error")
    error_id = state.get("error_id")
    current_agent = state.get("current_agent", "unknown")
    
    # Log recovery attempt
    await log_event(
        event_type="recovery_attempt",
        message=f"Attempting to recover from error in {current_agent}: {error}",
        severity="warning",
        workflow_id=state.get("workflow_id"),
        session_id=state.get("session_id"),
        agent_id=current_agent
    )
    
    # Simple recovery strategy - graceful degradation
    recovery_action = "Skipping failed step and continuing with limited functionality"
    
    # Update error report
    if error_id:
        for report in recent_errors:
            if report.error_id == error_id:
                report.recovery_action = recovery_action
                report.is_resolved = True
                await persist_data(report, ERRORS_FILE, append=False)
                break
    
    # Return updated state
    return {
        **state,
        "recovery_needed": False,
        "recovery_action": recovery_action,
        "has_degraded_functionality": True
    }

# Utility functions
def get_recent_errors(limit: int = 10) -> List[ErrorReport]:
    """
    Get recent errors
    
    Args:
        limit: Maximum number of errors to return
        
    Returns:
        List of recent ErrorReport objects
    """
    return sorted(
        recent_errors,
        key=lambda err: datetime.fromisoformat(err.timestamp),
        reverse=True
    )[:limit]

def get_active_workflows() -> List[WorkflowExecution]:
    """
    Get currently active workflows
    
    Returns:
        List of active WorkflowExecution objects
    """
    return [
        workflow for workflow in active_workflows.values()
        if workflow.status == "running"
    ]

def get_agent_performance_summary() -> Dict[str, Dict[str, float]]:
    """
    Get performance summary for all agents
    
    Returns:
        Dictionary with performance metrics by agent type
    """
    summary = {}
    
    for agent_type, metrics_list in performance_metrics.items():
        if not metrics_list:
            continue
            
        total_executions = len(metrics_list)
        successful_executions = sum(1 for m in metrics_list if m.success)
        total_time = sum(m.execution_time for m in metrics_list)
        
        summary[agent_type] = {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time": total_time / total_executions if total_executions > 0 else 0,
            "error_rate": (total_executions - successful_executions) / total_executions if total_executions > 0 else 0
        }
    
    return summary
