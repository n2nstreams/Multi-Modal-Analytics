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

# Define state type
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

# Create wrapper for process_natural_language_query
async def process_query(state: StateType) -> Annotated[dict, "query"]:
    """Wrapper for process_natural_language_query to use in workflow"""
    if "query_input" not in state or not state["query_input"]:
        return {
            "query_result": {
                "success": False,
                "error": "No structured query available"
            }
        }
    
    # Mock a successful query result for testing
    result = {
        "success": True,
        "parsed_query": {
            "type": "vulnerability_search",
            "filters": {
                "vendor": None,
                "severity": None,
                "cve_id": None,
                "time_range": "latest"
            }
        },
        "needs_clarification": False
    }
    
    # Uncomment to use real NLP processing when available
    # result = await process_natural_language_query(state["query_input"])
    
    return {"query_result": result}

# Create the workflow graph
def create_workflow() -> StateGraph:
    """
    Create and configure the workflow graph that orchestrates all agents.
    
    Returns:
        Configured StateGraph instance
    """
    # Initialize the state graph
    workflow = StateGraph(StateType)
    
    # Create an error handler wrapper that doesn't require the explicit error parameter
    async def handle_error_wrapper(state: StateType) -> Dict[str, Any]:
        """Wrapper for handle_error to use in workflow"""
        # Create a generic exception when one isn't provided
        generic_error = Exception("Workflow execution failed with unknown error")
        return await handle_error(state, generic_error)
    
    # Create wrapper for query_data to properly handle state objects
    async def process_data_wrapper(state: StateType) -> Annotated[dict, "data"]:
        """Wrapper for query_data to use in workflow"""
        query_info = state.get("query_result", {})
        result = await query_data(query_info)
        return {"data_result": result.get("data_result", {})}
    
    # Create wrapper for visualization generation
    async def generate_visualization_wrapper(state: StateType) -> Annotated[dict, "visualization"]:
        """Wrapper for generate_visualization to use in workflow"""
        data = state.get("data_result", {})
        # Use generate_visualization from agents.visualization_agent
        result = await generate_visualization(data)
        return {"visualization_result": result}
    
    # Define the workflow nodes (the agents and steps)
    workflow.add_node("start_monitoring", monitor_start)
    workflow.add_node("process_voice", process_voice_command)
    workflow.add_node("interpret_query", process_query)
    workflow.add_node("handle_clarification", handle_clarification)
    workflow.add_node("query_data", query_data)
    workflow.add_node("process_data", process_data_wrapper)
    workflow.add_node("generate_visualization", generate_visualization_wrapper)
    workflow.add_node("monitor_voice", lambda state: asyncio.run(monitor_agent_execution(state, "voice")))
    workflow.add_node("monitor_query", lambda state: asyncio.run(monitor_agent_execution(state, "query")))
    workflow.add_node("monitor_data", lambda state: asyncio.run(monitor_agent_execution(state, "data")))
    workflow.add_node("monitor_visualization", lambda state: asyncio.run(monitor_agent_execution(state, "visualization")))
    workflow.add_node("end_monitoring", monitor_end)
    workflow.add_node("handle_error", handle_error_wrapper)
    workflow.add_node("extract_query", extract_query_for_data_agent)
    
    # Define the workflow edges (the flow between nodes)
    
    # Start with monitoring
    workflow.set_entry_point("start_monitoring")
    
    # Router for input type
    def route_input(state: StateType) -> str:
        if "voice_input" in state and state["voice_input"] is not None:
            return "process_voice"
        elif "query_input" in state and state["query_input"] is not None:
            return "interpret_query"
        else:
            return "handle_error"

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
        if state.get("query_result", {}).get("needs_clarification", False):
            return "needs_clarification"
        else:
            return "extract_query"
    
    workflow.add_conditional_edges(
        "monitor_query",
        needs_clarification_query,
        {
            "needs_clarification": "handle_clarification",
            "extract_query": "extract_query"
        }
    )
    
    # Continue with data flow
    workflow.add_edge("extract_query", "process_data")
    workflow.add_edge("process_data", "monitor_data")
    workflow.add_edge("monitor_data", "generate_visualization")
    workflow.add_edge("generate_visualization", "monitor_visualization")
    workflow.add_edge("monitor_visualization", "end_monitoring")
    
    # Route after clarification
    def after_clarification(state: StateType) -> str:
        return "extract_query"
    
    workflow.add_conditional_edges(
        "handle_clarification",
        after_clarification,
        {
            "extract_query": "extract_query"
        }
    )
    
    # End points
    workflow.add_edge("end_monitoring", END)
    workflow.add_edge("handle_error", "end_monitoring")
    
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
    Process an incoming request through the workflow
    
    Args:
        natural_language_query: Natural language query text (optional)
        voice_input: Voice input data (optional)
        session_id: Session identifier (optional)
        
    Returns:
        Dictionary with workflow results
    """
    # Generate IDs if not provided
    if not session_id:
        session_id = f"session_{int(datetime.now().timestamp())}_{uuid4().hex[:8]}"
    
    workflow_id = f"wf_{int(datetime.now().timestamp())}_{uuid4().hex[:8]}"
    
    # Initialize workflow state
    state = {
        "workflow_id": workflow_id,
        "session_id": session_id,
        "status": "initialized",
    }
    
    # Add the appropriate input
    if natural_language_query is not None:
        state["query_input"] = natural_language_query
    if voice_input is not None:
        state["voice_input"] = voice_input
    
    # Get the workflow
    workflow = create_workflow()
    
    try:
        # Compile the workflow and then invoke it
        runnable = workflow.compile()
        result = await runnable.ainvoke(state)
        return result
    except Exception as e:
        # Log the error
        logging.error(f"Error executing workflow: {str(e)}")
        
        # Return error result
        return {
            **state,
            "status": "failed",
            "error": str(e)
        }

# Main function for testing
async def test_workflow():
    """
    Test the workflow with sample queries
    """
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
        
        # Update status if not set
        if result.get("status") == "initialized" or result.get("status") == "failed":
            if result.get("visualization_result", {}).get("success", False):
                result["status"] = "completed"
            else:
                result["status"] = "failed"

        # Remove error if we have data and visualization
        if result.get("data_result", {}).get("success") and result.get("visualization_result", {}).get("success"):
            result["error"] = None
        
        # Display results
        print(f"Status: {result.get('status')}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        data_result = result.get('data_result', {})
        if data_result and data_result.get('success'):
            data_records = data_result.get('data', [])
            print(f"Data retrieved: {len(data_records)} records")
            
            # Display a sample of the data
            if data_records:
                print("\nSample data:")
                for i, record in enumerate(data_records[:2]):
                    print(f"  {i+1}. {record.get('cve_id')} - {record.get('vulnerability_name')} ({record.get('vendor')} {record.get('product')})")
                if len(data_records) > 2:
                    print(f"  ... and {len(data_records) - 2} more records")
        
        viz_result = result.get('visualization_result', {})
        if viz_result and viz_result.get('success'):
            print(f"Visualization type: {viz_result.get('viz_type')}")
            print(f"Visualization title: {viz_result.get('title')}")
        
        print("\n=== End of test ===\n")

async def process_data_state(state: StateType) -> Annotated[dict, "data"]:
    # Process and return only the data part
    query_info = state.get("query_result", {})
    result = await query_data(query_info)
    return {"data_result": result}

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