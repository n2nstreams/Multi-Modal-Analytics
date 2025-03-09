"""
API Endpoints for MMA Application

This module defines the API endpoints for the MMA vulnerability analysis system.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

# Import the process_request function - make sure workflow.py is fixed first
from workflow import process_request

# Create router
router = APIRouter(prefix="/api", tags=["mma"])

# Request models
class QueryRequest(BaseModel):
    """Model for natural language query requests"""
    query: str = Field(..., description="Natural language query text")
    session_id: Optional[str] = Field(None, description="Session identifier")

class VoiceRequest(BaseModel):
    """Model for voice command requests"""
    audio_data: Dict[str, Any] = Field(..., description="Voice input data")
    session_id: Optional[str] = Field(None, description="Session identifier")

# Response models
class QueryResponse(BaseModel):
    """Model for query response"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Status of the request")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved data")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Visualization result")
    error: Optional[str] = Field(None, description="Error message if any")

# Endpoints
@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a natural language query
    
    Args:
        request: Query request
        
    Returns:
        Query response
    """
    try:
        result = await process_request(
            natural_language_query=request.query,
            session_id=request.session_id
        )
        
        return {
            "session_id": result.get("session_id", ""),
            "status": result.get("status", "failed"),
            "data": result.get("data_result", {}).get("data") if result.get("data_result") else None,
            "visualization": result.get("visualization_result"),
            "error": result.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice", response_model=QueryResponse)
async def handle_voice(request: VoiceRequest) -> Dict[str, Any]:
    """
    Process a voice command
    
    Args:
        request: Voice request
        
    Returns:
        Query response
    """
    try:
        result = await process_request(
            voice_input=request.audio_data,
            session_id=request.session_id
        )
        
        return {
            "session_id": result.get("session_id", ""),
            "status": result.get("status", "failed"),
            "data": result.get("data_result", {}).get("data") if result.get("data_result") else None,
            "visualization": result.get("visualization_result"),
            "error": result.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    return {"status": "healthy"} 