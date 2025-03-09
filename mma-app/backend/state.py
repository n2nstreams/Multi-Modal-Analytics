"""
State Models for MMA Application

This module defines the state models that are passed between
different nodes in the LangGraph workflow.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class VoiceInput(BaseModel):
    """Model for voice input data"""
    audio_path: str = Field(..., description="Path to the audio file")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Timestamp of the recording")
    duration: float = Field(..., description="Duration of the audio in seconds")
    language: str = Field("en", description="Language of the audio")
    
class TranscriptionResult(BaseModel):
    """Model for voice transcription results"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score of transcription")
    audio_path: str = Field(..., description="Original audio file path")
    language_detected: Optional[str] = Field(None, description="Detected language")
    
class QueryInfo(BaseModel):
    """Model for structured query information"""
    source_type: Literal["sqlite", "csv"] = Field(..., description="Type of data source")
    source_path: str = Field(..., description="Path to the data source")
    query: Optional[str] = Field(None, description="SQL query or filter criteria")
    fields: Optional[List[str]] = Field(None, description="Fields to retrieve")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range for filtering")
    aggregation: Optional[str] = Field(None, description="Aggregation function")
    limit: Optional[int] = Field(None, description="Maximum number of records")
    
class DataResult(BaseModel):
    """Model for data query results"""
    success: bool = Field(..., description="Whether the query was successful")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved data")
    error: Optional[str] = Field(None, description="Error message if query failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
class VisualizationConfig(BaseModel):
    """Model for visualization configuration"""
    type: str = Field(..., description="Type of visualization")
    title: str = Field(..., description="Title for the visualization")
    x_axis: Optional[str] = Field(None, description="Field for x-axis")
    y_axis: Optional[List[str]] = Field(None, description="Fields for y-axis")
    color_by: Optional[str] = Field(None, description="Field to use for color")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    description: str = Field(..., description="Description of the visualization")
    
class VisualizationResult(BaseModel):
    """Model for visualization results"""
    success: bool = Field(..., description="Whether visualization was successful")
    type: str = Field(..., description="Type of visualization generated")
    image_path: Optional[str] = Field(None, description="Path to generated image file")
    html_content: Optional[str] = Field(None, description="HTML content for interactive visualization")
    error: Optional[str] = Field(None, description="Error message if visualization failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
class WorkflowState(BaseModel):
    """Model for the complete workflow state"""
    # Session information
    session_id: str = Field(..., description="Session identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")
    
    # Voice and query information
    voice_input: Optional[VoiceInput] = Field(None, description="Voice input data")
    transcription_result: Optional[TranscriptionResult] = Field(None, description="Transcription result")
    natural_language_query: Optional[str] = Field(None, description="Natural language query")
    previous_queries: List[str] = Field(default_factory=list, description="Previous queries in this session")
    
    # Query interpretation
    query_interpretation: Optional[Dict[str, Any]] = Field(None, description="Interpretation of the query")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    
    # Data query information
    query_info: Optional[Dict[str, Any]] = Field(None, description="Structured query information")
    data_result: Optional[Dict[str, Any]] = Field(None, description="Data query results")
    
    # Visualization
    visualization_config: Optional[Dict[str, Any]] = Field(None, description="Visualization configuration")
    visualization_result: Optional[Dict[str, Any]] = Field(None, description="Visualization results")
    
    # Monitoring and error handling
    current_agent: Optional[str] = Field(None, description="Currently executing agent")
    monitoring: Optional[Dict[str, Any]] = Field(None, description="Monitoring information")
    error: Optional[str] = Field(None, description="Error message if workflow failed")
    error_id: Optional[str] = Field(None, description="Error identifier")
    recovery_needed: bool = Field(False, description="Whether recovery is needed")
    recovery_action: Optional[str] = Field(None, description="Recovery action taken")
    
    # User preferences
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
