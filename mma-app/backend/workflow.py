"""
Main Workflow Orchestrator for MMA Application

This module creates and configures the LangGraph workflow that coordinates
all agents in the vulnerability analysis system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, TypedDict, Annotated, NotRequired
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from langgraph.graph import StateGraph, END

# Import our agent modules
from agents.voice_agent import process_voice_input as process_voice_command
from agents.inference_agent import (
    process_natural_language_query, 
    handle_clarification,
    extract_query_for_data_agent
)
from agents.data_agent import query_data
from agents.visualization_agent import generate_visualization
from agents.monitoring_agent import (
    monitor_start, 
    monitor_agent_execution, 
    monitor_end, 
    handle_error
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definition for state
class StateType(TypedDict):
    query_input: str
    query_result: NotRequired[Annotated[dict, "query"]]  
    data_result: NotRequired[Annotated[dict, "data"]]
    visualization_result: NotRequired[Annotated[dict, "visualization"]]
    clarification_needed: NotRequired[bool]
    clarification_response: NotRequired[str]
    workflow_id: str
    session_id: str
    status: str
    error: NotRequired[str]

# Create the workflow graph
def create_workflow() -> StateGraph:
    """
    Create and configure the main workflow graph
    
    Returns:
        Configured StateGraph for the MMA workflow
    """
    # Create a new graph
    workflow = StateGraph(StateType)
    
    # Add nodes to the graph
    workflow.add_node("start_monitoring", monitor_start)
    workflow.add_node("process_voice", process_voice_command)
    workflow.add_node("interpret_query", process_natural_language_query)
    workflow.add_node("handle_clarification", handle_clarification)
    workflow.add_node("query_data", query_data)
    workflow.add_node("generate_visualization", generate_visualization)
    workflow.add_node("monitor_voice", lambda state: asyncio.run(monitor_agent_execution(state, "voice")))
    workflow.add_node("monitor_query", lambda state: asyncio.run(monitor_agent_execution(state, "query")))
    workflow.add_node("monitor_data", lambda state: asyncio.run(monitor_agent_execution(state, "data")))
    workflow.add_node("monitor_visualization", lambda state: asyncio.run(monitor_agent_execution(state, "visualization")))
    workflow.add_node("end_monitoring", monitor_end)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("extract_query", extract_query_for_data_agent)
    
    # Define the workflow edges (the flow between nodes)
    
    # Start with monitoring
    workflow.set_entry_point("start_monitoring")
    
    # Router for input type
    def route_input(state: StateType) -> str:
        if state.get("query_result", {}).get("needs_clarification", False):
            return "needs_clarification"
        return "process_data"

    workflow.add_conditional_edges(
        "start_monitoring",
        route_input,
        {
            "process_voice": "process_voice",
            "interpret_query": "interpret_query",
            "handle_error": "handle_error"
        }
    )
    
    # Voice processing flow
    workflow.add_edge("process_voice", "monitor_voice")
    workflow.add_edge("monitor_voice", "interpret_query")
    
    # Query interpretation flow
    workflow.add_edge("interpret_query", "monitor_query")
    
    # Router for clarification needs after query interpretation
    def needs_clarification_query(state: StateType) -> str:
        if state.get("needs_clarification", False) or state.get("needs_human_input", False):
            return "needs_clarification"
        else:
            return "proceed"
    
    workflow.add_conditional_edges(
        "monitor_query", 
        needs_clarification_query,
        {
            "needs_clarification": "handle_clarification",
            "proceed": "query_data"
        }
    )
    
    # Router for after clarification handling
    def after_clarification(state: StateType) -> str:
        if state.get("needs_human_input", False):
            return "end"
        else:
            return "query_data"
    
    workflow.add_conditional_edges(
        "handle_clarification",
        after_clarification,
        {
            "end": END,
            "query_data": "query_data"
        }
    )
    
    # Data querying and visualization flow
    workflow.add_edge("query_data", "monitor_data")
    workflow.add_edge("monitor_data", "generate_visualization")
    workflow.add_edge("generate_visualization", "monitor_visualization")
    workflow.add_edge("monitor_visualization", "end_monitoring")
    
    # End the workflow
    workflow.add_edge("end_monitoring", END)
    
    # Error handling routes
    workflow.add_edge("handle_error", "end_monitoring")
    
    # Add a node to explicitly extract query info before data query
    workflow.add_edge("monitor_query", "extract_query")
    workflow.add_edge("extract_query", "query_data")
    
    return workflow

# Initialize the workflow
vulnerability_workflow = create_workflow()

# Function to process a new request
async def process_request(
    natural_language_query: Optional[str] = None,
    voice_input: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a new vulnerability analysis request
    
    Args:
        natural_language_query: Natural language query text (if direct text input)
        voice_input: Voice input data (if voice command)
        session_id: Session identifier (generated if not provided)
        
    Returns:
        Dictionary with workflow results
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid4())
    
    # Initialize state
    state = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "natural_language_query": natural_language_query,
        "voice_input": voice_input,
        "status": "initialized"
    }
    
    # Execute the workflow
    try:
        # Create a new workflow instance
        workflow_instance = vulnerability_workflow.compile()
        
        # Run the workflow
        result = await workflow_instance.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return {
            **state,
            "error": str(e),
            "status": "failed"
        }

# Main function for testing
async def test_workflow():
    """
    Test the workflow with sample queries
    """
    # Add to the top of test_workflow function
    def ensure_test_data():
        data_dir = Path("../database/csv")
        data_dir.mkdir(parents=True, exist_ok=True)
        # Notify user about test data requirements
        if not (data_dir / "known_exploited_vulnerabilities.csv").exists():
            print("\n⚠️ WARNING: Test data file not found!")
            print("Please download known_exploited_vulnerabilities.csv from:")
            print("https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv")
            print(f"And place it at: {data_dir}/known_exploited_vulnerabilities.csv\n")

    ensure_test_data()

    # Test queries
    test_queries = [
        "Show me the latest vulnerabilities",
        "What are the critical vulnerabilities in Microsoft products?",
        "Show me vulnerability trends over the past year",
        "List all Adobe vulnerabilities",
        "Tell me about CVE-2021-44228"  # Log4Shell vulnerability
    ]
    
    for query in test_queries:
        print(f"\n\n=== Testing query: '{query}' ===\n")
        
        result = await process_request(natural_language_query=query)
        
        # Display results
        print(f"Status: {result.get('status')}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        data_result = result.get('data_result', {})
        if data_result and data_result.get('success'):
            print(f"Data retrieved: {len(data_result.get('data', []))} records")
        
        viz_result = result.get('visualization_result', {})
        if viz_result and viz_result.get('success'):
            print(f"Visualization type: {viz_result.get('viz_type')}")
            print(f"Visualization title: {viz_result.get('title')}")
        
        print("\n=== End of test ===\n")

async def process_query(state: StateType) -> Annotated[dict, "query"]:
    # Process and return only the query part
    result = await process_natural_language_query(state["query_input"])
    return {"query_result": result}

async def process_data(state: StateType) -> Annotated[dict, "data"]:
    # Process and return only the data part
    query_info = state.get("query_result", {})
    result = await query_data(query_info)
    return {"data_result": result}

async def generate_visualization(state: StateType) -> Annotated[dict, "visualization"]:
    # Process and return only the visualization part
    data = state.get("data_result", {})
    result = await generate_visualization(data)
    return {"visualization_result": result}

async def monitor_start_sync(state: StateType) -> dict:
    # Use a sync wrapper that calls the async function
    await monitor_start(state)
    return {}  # Empty update to avoid modifying state

async def monitor_agent_execution_sync(state: StateType, agent_type: str) -> dict:
    # Use a sync wrapper that calls the async function
    await monitor_agent_execution(state, agent_type)
    return {}  # Empty update to avoid modifying state

if __name__ == "__main__":
    # Run the test workflow
    asyncio.run(test_workflow()) 