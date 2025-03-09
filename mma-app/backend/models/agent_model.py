"""
Agent Models for MMA Application

This module defines Pydantic models used by agents across the system to ensure
consistent data structures and validation.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    """User query request model"""
    query_text: str = Field(..., description="Original natural language query")
    query_type: Optional[str] = Field(None, description="Detected query type")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('query_text')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query text cannot be empty')
        return v

class QueryParameters(BaseModel):
    """Structured query parameters extracted from natural language"""
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    vendor: Optional[str] = Field(None, description="Vendor filter")
    cve_id: Optional[str] = Field(None, description="Specific CVE identifier")
    severity: Optional[str] = Field(None, description="Severity level filter")
    time_range: Optional[str] = Field(None, description="Time range for query")
    limit: int = Field(50, description="Maximum number of results to return")

class VulnerabilityData(BaseModel):
    """Model for vulnerability data"""
    cve_id: str = Field(..., description="CVE identifier")
    vendor: str = Field(..., description="Vendor name")
    product: str = Field(..., description="Product name")
    vulnerability_name: Optional[str] = Field(None, description="Vulnerability name")
    description: str = Field(..., description="Vulnerability description")
    severity: str = Field(..., description="Severity rating")
    published_date: str = Field(..., description="Date published")
    fixed_date: Optional[str] = Field(None, description="Date fixed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DataResult(BaseModel):
    """Data query result model"""
    success: bool = Field(..., description="Whether the query was successful")
    query_parameters: QueryParameters = Field(..., description="Parameters used for query")
    data: List[VulnerabilityData] = Field(default_factory=list, description="Retrieved vulnerability data")
    count: int = Field(0, description="Number of records retrieved")
    error: Optional[str] = Field(None, description="Error message if query failed")

class VisualizationType(str, Literal["bar_chart", "line_chart", "pie_chart", "table", "timeline"]):
    """Types of visualizations supported by the system"""
    pass

class VisualizationRequest(BaseModel):
    """Request for visualization generation"""
    viz_type: VisualizationType = Field(..., description="Type of visualization to generate")
    title: str = Field(..., description="Visualization title")
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    x_axis: Optional[str] = Field(None, description="X-axis field name")
    y_axis: Optional[str] = Field(None, description="Y-axis field name")
    group_by: Optional[str] = Field(None, description="Field to group by")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters to apply")

class VisualizationResult(BaseModel):
    """Visualization result model"""
    success: bool = Field(..., description="Whether visualization was successful")
    viz_type: VisualizationType = Field(..., description="Type of visualization generated")
    title: str = Field(..., description="Visualization title")
    chart_data: Any = Field(..., description="Visualization data structure")
    error: Optional[str] = Field(None, description="Error message if visualization failed")

class AgentState(BaseModel):
    """Common agent state model"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Type of agent")
    is_processing: bool = Field(False, description="Whether the agent is currently processing")
    last_execution: Optional[datetime] = Field(None, description="When agent last executed")
    execution_count: int = Field(0, description="Number of times agent has executed")
    last_error: Optional[str] = Field(None, description="Last error encountered")
