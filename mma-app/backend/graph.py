"""
LangGraph Workflow for MMA Application

This module defines the LangGraph workflow that orchestrates
the different agents in the MMA application.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable, cast, TypeVar
from langchain_core.callbacks.base import Callbacks
from langgraph.graph import StateGraph, END

# Import agent functions
from agents.voice_agent import process_voice_input, validate_voice_input
from agents.inference_agent import process_natural_language_query, handle_clarification, extract_query_for_data_agent
from agents.data_agent import query_data
from agents.visualization_agent import generate_visualization
from agents.monitoring_agent import (
    monitor_start, 
    monitor_agent_execution, 
    monitor_end, 
    handle_error, 
    attempt_recovery
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
StateType = Dict[str, Any]
AgentFunction = Callable[[StateType], Awaitable[StateType]]

def create_workflow_graph() -> StateGraph:
    """
    Create the LangGraph workflow for the MMA application
    
    Returns:
        StateGraph object with defined agent workflow
    """
    # Create the workflow graph
    workflow = StateGraph(StateType)
    
    # Define node functions with monitoring
    
    # Start monitoring
    workflow.add_node("start_monitoring", monitor_start)
    
    # Voice processing
    async def monitored_voice_process(state: StateType) -> StateType:
        """Voice processing with monitoring"""
        state = await process_voice_input(state)
        return await monitor_agent_execution(state, "voice_agent")
    
    workflow.add_node("process_voice", monitored_voice_process)
    workflow.add_node("validate_voice", validate_voice_input)
    
    # Query interpretation
    async def monitored_query_interpretation(state: StateType) -> StateType:
        """Query interpretation with monitoring"""
        state = await process_natural_language_query(state)
        return await monitor_agent_execution(state, "query_agent")
    
    workflow.add_node("interpret_query", monitored_query_interpretation)
    workflow.add_node("handle_clarification", handle_clarification)
    workflow.add_node("extract_query", extract_query_for_data_agent)
    
    # Data querying
    async def monitored_data_query(state: StateType) -> StateType:
        """Data querying with monitoring"""
        state = await query_data(state)
        return await monitor_agent_execution(state, "data_agent")
    
    workflow.add_node("query_data", monitored_data_query)
    
    # Visualization generation
    async def monitored_visualization(state: StateType) -> StateType:
        """Visualization generation with monitoring"""
        state = await generate_visualization(state)
        return await monitor_agent_execution(state, "visualization_agent")
    
    workflow.add_node("generate_visualization", monitored_visualization)
    
    # End monitoring
    workflow.add_node("end_monitoring", monitor_end)
    
    # Error handling
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("attempt_recovery", attempt_recovery)
    
    # Define the workflow
    
    # Start with monitoring
    workflow.set_entry_point("start_monitoring")
    
    # Conditional routing based on input type
    def route_by_input(state: StateType) -> str:
        """Route based on input type"""
        if state.get("voice_input"):
            return "process_voice"
        elif state.get("natural_language_query"):
            return "interpret_query"
        else:
            return "handle_error"
    
    workflow.add_conditional_edges(
        "start_monitoring",
        route_by_input,
        {
            "process_voice": "process_voice",
            "interpret_query": "interpret_query",
            "handle_error": "handle_error"
        }
    )
    
    # Voice processing flow
    workflow.add_edge("process_voice", "validate_voice")
    
    def needs_clarification_voice(state: StateType) -> str:
        """Check if voice input needs clarification"""
        if state.get("needs_clarification", False):
            return "needs_clarification"
        else:
            return "proceed"
    
    workflow.add_conditional_edges(
        "validate_voice", 
        needs_clarification_voice,
        {
            "needs_clarification": "handle_clarification",
            "proceed": "interpret_query"
        }
    )
    
    # Query interpretation flow
    def needs_clarification_query(state: StateType) -> str:
        """Check if query needs clarification"""
        if state.get("needs_clarification", False):
            return "needs_clarification"
        else:
            return "proceed"
    
    workflow.add_conditional_edges(
        "interpret_query", 
        needs_clarification_query,
        {
            "needs_clarification": "handle_clarification",
            "proceed": "extract_query"
        }
    )
    
    # After clarification, go back to query interpretation
    workflow.add_edge("handle_clarification", "interpret_query")
    
    # Data querying
    workflow.add_edge("extract_query", "query_data")
    
    # Visualization
    workflow.add_edge("query_data", "generate_visualization")
    
    # End monitoring
    workflow.add_edge("generate_visualization", "end_monitoring")
    
    # Error handling routes
    workflow.add_edge("handle_error", "attempt_recovery")
    
    def check_recovery(state: StateType) -> str:
        """Check if recovery was successful"""
        if state.get("recovery_needed", False):
            return "failed"
        else:
            return "succeeded"
    
    workflow.add_conditional_edges(
        "attempt_recovery",
        check_recovery,
        {
            "failed": END,  # End workflow if recovery failed
            "succeeded": "end_monitoring"  # Continue to end monitoring if recovered
        }
    )
    
    # End the workflow
    workflow.add_edge("end_monitoring", END)
    
    return workflow

async def run_workflow(
    initial_state: Dict[str, Any],
    callbacks: Optional[Callbacks] = None
) -> Dict[str, Any]:
    """
    Run the MMA workflow with the provided initial state
    
    Args:
        initial_state: Initial state for the workflow
        callbacks: Optional callbacks for monitoring
        
    Returns:
        Final state after workflow execution
    """
    # Create the workflow graph
    workflow = create_workflow_graph()
    
    # Run the workflow
    try:
        app = workflow.compile()
        result = await app.ainvoke(initial_state, config={"callbacks": callbacks})
        return result
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        # Try to run error handling directly
        try:
            error_state = await handle_error(initial_state, e)
            recovery_state = await attempt_recovery(error_state)
            return recovery_state
        except Exception as recovery_error:
            logger.critical(f"Failed to recover from error: {recovery_error}")
            return {
                **initial_state,
                "error": f"Critical error: {str(e)}",
                "recovery_error": f"Recovery failed: {str(recovery_error)}"
            }

# Main entry point for running a workflow from voice input
async def process_voice_command(
    audio_path: str,
    session_id: Optional[str] = None,
    callbacks: Optional[Callbacks] = None
) -> Dict[str, Any]:
    """
    Process a voice command
    
    Args:
        audio_path: Path to audio file
        session_id: Optional session ID
        callbacks: Optional callbacks for monitoring
        
    Returns:
        Result of workflow execution
    """
    # Create initial state
    initial_state = {
        "session_id": session_id or os.urandom(4).hex(),
        "voice_input": {
            "audio_path": audio_path,
            "session_id": session_id or os.urandom(4).hex(),
            "timestamp": "",  # Will be filled by voice agent
            "duration": 0.0,  # Will be filled by voice agent
            "language": "en"
        }
    }
    
    # Run workflow
    return await run_workflow(initial_state, callbacks)

# Main entry point for running a workflow from text input
async def process_text_query(
    query: str,
    session_id: Optional[str] = None,
    callbacks: Optional[Callbacks] = None
) -> Dict[str, Any]:
    """
    Process a text query
    
    Args:
        query: Natural language query
        session_id: Optional session ID
        callbacks: Optional callbacks for monitoring
        
    Returns:
        Result of workflow execution
    """
    # Create initial state
    initial_state = {
        "session_id": session_id or os.urandom(4).hex(),
        "natural_language_query": query
    }
    
    # Run workflow
    return await run_workflow(initial_state, callbacks)
