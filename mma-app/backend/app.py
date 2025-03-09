"""
Main Application Entry Point for MMA

This module provides the FastAPI application for the MMA system
with endpoints for voice and text-based queries.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our workflow functions
from workflow import process_request
from agents.monitoring_agent import check_system_health, get_recent_errors, get_agent_performance_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
UPLOAD_DIR = Path("../uploads")
STATIC_DIR = Path("../static")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR / "audio", exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(STATIC_DIR / "visualizations", exist_ok=True)

# Define Pydantic models for API
class TextQueryRequest(BaseModel):
    """Model for text query request"""
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Session identifier")

class VoiceQueryRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: str = Field("wav", description="Audio format")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    session_id: Optional[str] = Field(None, description="Session identifier")

class QueryResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for this request")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Status of the request")
    results: Optional[Dict[str, Any]] = Field(None, description="Query results")
    error: Optional[str] = Field(None, description="Error message if request failed")
    needs_clarification: Optional[bool] = Field(None, description="Whether clarification is needed")
    clarification_prompt: Optional[str] = Field(None, description="Prompt for clarification")

class HealthStatusResponse(BaseModel):
    """Model for system health status"""
    status: str = Field(..., description="Overall system status")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_sessions: int = Field(..., description="Number of active sessions")
    error_rate: float = Field(..., description="Error rate in recent executions")
    components: Dict[str, str] = Field(..., description="Status of system components")
    timestamp: str = Field(..., description="Timestamp of health check")

# Create FastAPI app
app = FastAPI(
    title="MMA - Multimodal Vulnerability Analysis",
    description="Voice-triggered vulnerability report generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Active workflows
active_jobs: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns basic HTML page"""
    return """
    <html>
        <head>
            <title>MMA - Multimodal Vulnerability Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                p { color: #666; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>MMA - Multimodal Vulnerability Analysis</h1>
            <p>API endpoints available:</p>
            <ul>
                <li><code>POST /api/query/text</code> - Submit a text query</li>
                <li><code>POST /api/query/voice</code> - Submit a voice query</li>
                <li><code>GET /api/health</code> - Get system health status</li>
                <li><code>GET /api/docs</code> - API documentation</li>
            </ul>
            <p>For full API documentation, visit <a href="/docs">/docs</a></p>
        </body>
    </html>
    """

@app.post("/api/query/text", response_model=QueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Process a text query for vulnerability analysis
    """
    try:
        # Process the request
        result = await process_request(
            natural_language_query=request.query,
            session_id=request.session_id
        )
        
        # Prepare the response
        request_id = str(uuid.uuid4())
        
        response = {
            "request_id": request_id,
            "session_id": result.get("session_id", ""),
            "status": result.get("status", "failed"),
            "results": {
                "data_result": result.get("data_result"),
                "visualization_result": result.get("visualization_result")
            },
            "error": result.get("error"),
            "needs_clarification": result.get("needs_human_input", False),
            "clarification_prompt": result.get("human_input_prompt")
        }
        
        return response
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/voice", response_model=QueryResponse)
async def voice_query(request: VoiceQueryRequest):
    """
    Process a voice query for vulnerability analysis
    """
    try:
        # Process the request
        voice_input = {
            "audio_data": request.audio_data,
            "format": request.format,
            "sample_rate": request.sample_rate
        }
        
        result = await process_request(
            voice_input=voice_input,
            session_id=request.session_id
        )
        
        # Prepare the response
        request_id = str(uuid.uuid4())
        
        response = {
            "request_id": request_id,
            "session_id": result.get("session_id", ""),
            "status": result.get("status", "failed"),
            "results": {
                "voice_result": result.get("voice_result"),
                "data_result": result.get("data_result"),
                "visualization_result": result.get("visualization_result")
            },
            "error": result.get("error"),
            "needs_clarification": result.get("needs_human_input", False),
            "clarification_prompt": result.get("human_input_prompt")
        }
        
        return response
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthStatusResponse)
async def get_health():
    """
    Get system health status
    
    Returns:
        HealthStatusResponse with system health information
    """
    try:
        health = await check_system_health()
        
        return HealthStatusResponse(
            status="operational" if health.error_rate < 0.1 else "degraded",
            cpu_usage=health.cpu_usage,
            memory_usage=health.memory_usage,
            active_sessions=health.active_sessions,
            error_rate=health.error_rate,
            components=health.component_status,
            timestamp=health.timestamp
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        return HealthStatusResponse(
            status="error",
            cpu_usage=0.0,
            memory_usage=0.0,
            active_sessions=0,
            error_rate=1.0,
            components={"error": "unavailable"},
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/errors")
async def get_errors():
    """
    Get recent system errors
    
    Returns:
        List of recent errors
    """
    try:
        errors = get_recent_errors(limit=10)
        return {"errors": [error.dict() for error in errors]}
    except Exception as e:
        logger.error(f"Error getting system errors: {e}")
        return {"errors": [], "error": str(e)}

@app.get("/api/performance")
async def get_performance():
    """
    Get agent performance metrics
    
    Returns:
        Performance metrics by agent type
    """
    try:
        performance = get_agent_performance_summary()
        return {"performance": performance}
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {"performance": {}, "error": str(e)}

@app.get("/api/status")
async def get_status():
    """
    Get the API status
    """
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
