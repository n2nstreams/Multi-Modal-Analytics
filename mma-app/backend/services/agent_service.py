"""
Agent Service for MMA Application

This module provides service-layer functions to coordinate agent operations,
abstracting the implementation details from API endpoints.
"""

import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4

from workflow import process_request
from agents.monitoring_agent import log_event, get_recent_errors, check_system_health

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_query(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a natural language query through the agent workflow
    
    Args:
        query: Natural language query text
        session_id: Optional session identifier
        
    Returns:
        Dictionary with workflow results
    """
    logger.info(f"Processing query: {query}")
    return await process_request(natural_language_query=query, session_id=session_id)

async def process_voice_command(voice_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a voice command through the agent workflow
    
    Args:
        voice_data: Voice input data
        session_id: Optional session identifier
        
    Returns:
        Dictionary with workflow results
    """
    logger.info("Processing voice command")
    return await process_request(voice_input=voice_data, session_id=session_id)

async def get_system_status() -> Dict[str, Any]:
    """
    Get current system status including health metrics
    
    Returns:
        Dictionary with system status information
    """
    health = await check_system_health()
    errors = get_recent_errors(5)
    
    return {
        "health": health.dict(),
        "recent_errors": [error.dict() for error in errors],
        "status": "operational" if health.error_rate < 0.2 else "degraded"
    }
