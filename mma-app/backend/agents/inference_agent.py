"""
Inference Agent for MMA Application

This module provides functionality to:
1. Interpret natural language queries about vulnerabilities
2. Transform voice commands into structured data queries
3. Determine appropriate data sources and visualization types
4. Handle ambiguities by requesting clarification
"""

import logging
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for query processing
class QueryContext(BaseModel):
    """Model for storing context about a query"""
    original_query: str = Field(..., description="Original natural language query")
    session_id: str = Field(..., description="Unique session identifier")
    timestamp: str = Field(..., description="Timestamp of the query")
    previous_queries: List[str] = Field(default_factory=list, description="Previous queries in this session")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for visualization")
    
class StructuredQuery(BaseModel):
    """Model representing the structured query for data sources"""
    source_type: Literal["sqlite", "csv"] = Field(..., description="Type of data source to query")
    source_path: str = Field(..., description="Path to the data source")
    query: Optional[str] = Field(None, description="SQL query for SQLite or filter for CSV")
    fields: Optional[List[str]] = Field(None, description="Fields to retrieve from data source")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range for filtering data")
    aggregation: Optional[str] = Field(None, description="Aggregation function to apply")
    limit: Optional[int] = Field(None, description="Maximum number of records to return")
    
class VisualizationSpec(BaseModel):
    """Model for visualization recommendations"""
    type: str = Field(..., description="Type of visualization (bar, line, pie, etc.)")
    title: str = Field(..., description="Title for the visualization")
    x_axis: Optional[str] = Field(None, description="Field for x-axis")
    y_axis: Optional[List[str]] = Field(None, description="Fields for y-axis")
    color_by: Optional[str] = Field(None, description="Field to use for color")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    description: str = Field(..., description="Description of what the visualization shows")
    
class QueryInterpretation(BaseModel):
    """Model for the complete interpretation of a natural language query"""
    query_type: str = Field(..., description="Type of query (vulnerability report, trend analysis, etc.)")
    confidence: float = Field(..., description="Confidence score of interpretation", ge=0.0, le=1.0)
    natural_language_query: str = Field(..., description="Original natural language query")
    structured_query: Optional[StructuredQuery] = Field(None, description="Structured query for data sources")
    visualization: Optional[VisualizationSpec] = Field(None, description="Visualization recommendation")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(None, description="Question to ask for clarification")
    error: Optional[str] = Field(None, description="Error message if interpretation failed")

class DataSourceInfo(BaseModel):
    """Model for information about available data sources"""
    source_type: str = Field(..., description="Type of data source")
    source_path: str = Field(..., description="Path to the data source")
    tables: Optional[List[str]] = Field(None, description="Tables in the database (for SQLite)")
    columns: Optional[Dict[str, List[Dict[str, str]]]] = Field(None, description="Column information")
    description: Optional[str] = Field(None, description="Description of the data source")

# Add a constant for the test data path
KNOWN_VULNERABILITIES_PATH = "../database/csv/known_exploited_vulnerabilities.csv"

# Query templates
QUERY_TEMPLATES = {
    "latest_vulnerabilities": {
        "description": "Get the most recent vulnerabilities",
        "sqlite": "SELECT * FROM vulnerabilities ORDER BY timestamp DESC LIMIT {limit}",
        "csv": "dateAdded.sort_values(ascending=False).head({limit})"  # Assuming dateAdded is the date field
    },
    "critical_vulnerabilities": {
        "description": "Get vulnerabilities with critical severity",
        "sqlite": "SELECT * FROM vulnerabilities WHERE severity = 'critical'",
        "csv": "requiredAction.str.contains('critical', case=False, na=False)"  # Adjust based on actual field
    },
    "vulnerability_trend": {
        "description": "Get vulnerability trends over time",
        "sqlite": "SELECT date(timestamp) as date, COUNT(*) as count FROM vulnerabilities GROUP BY date(timestamp) ORDER BY date",
        "csv": None  # Requires custom aggregation with pandas
    },
    "vendor_vulnerabilities": {
        "description": "Get vulnerabilities by vendor",
        "sqlite": "SELECT * FROM vulnerabilities WHERE vendor = '{vendor}'", 
        "csv": "vendorProject == '{vendor}'"  # Assuming vendorProject is the vendor field
    },
    "cve_search": {
        "description": "Search for specific CVE",
        "sqlite": "SELECT * FROM vulnerabilities WHERE cve_id LIKE '%{cve_id}%'",
        "csv": "cveID.str.contains('{cve_id}', case=False, na=False)"  # Assuming cveID is the field name
    },
}

# LLM prompt templates for query interpretation
INTERPRETATION_PROMPT = """
You are an expert cybersecurity analyst tasked with interpreting natural language queries about vulnerability reports.
Please analyze the following query and convert it into a structured format for data retrieval.

Query: {query}

Available data sources: {data_sources}

Based on the query and available data sources, please provide:
1. The type of query (e.g., latest vulnerabilities, critical issues, trend analysis)
2. The most appropriate data source(s) to use
3. The specific fields needed
4. Any filtering criteria
5. Time range if applicable
6. The most appropriate visualization type

If the query is ambiguous or lacks necessary information, indicate what clarification is needed.
"""

# Mock function for LLM interpretation - in production, this would use an actual LLM API
async def interpret_with_llm(query: str, data_sources: List[DataSourceInfo], context: QueryContext) -> Dict[str, Any]:
    """
    Use an LLM to interpret a natural language query
    
    Args:
        query: Natural language query
        data_sources: Information about available data sources
        context: Context of the current query session
        
    Returns:
        Dictionary with interpretation results
    """
    # Find our known vulnerabilities file in the data sources
    vuln_source = next(
        (s for s in data_sources if KNOWN_VULNERABILITIES_PATH in s.source_path), 
        None
    )
    
    if not vuln_source:
        return {
            "query_type": "unknown",
            "confidence": 0.3,
            "natural_language_query": query,
            "needs_clarification": True,
            "clarification_question": "I couldn't find the vulnerabilities database. Please ensure the known_exploited_vulnerabilities.csv file is available."
        }
    
    query_lower = query.lower()
    
    # Example simple interpretation logic updated for the known vulnerabilities file
    if "latest" in query_lower or "recent" in query_lower:
        query_type = "latest_vulnerabilities"
        confidence = 0.85
        
        template = QUERY_TEMPLATES[query_type]
        structured_query = {
            "source_type": "csv",
            "source_path": KNOWN_VULNERABILITIES_PATH,
            "query": template["csv"].format(limit=10) if template["csv"] else None,
            "limit": 10
        }
        
        visualization = {
            "type": "table",
            "title": "Latest Known Exploited Vulnerabilities",
            "description": "Displays the most recent vulnerability records from CISA's known exploited vulnerabilities catalog"
        }
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "natural_language_query": query,
            "structured_query": structured_query,
            "visualization": visualization,
            "needs_clarification": False
        }
        
    elif "critical" in query_lower or "severe" in query_lower or "high" in query_lower:
        query_type = "critical_vulnerabilities"
        confidence = 0.9
        
        template = QUERY_TEMPLATES[query_type]
        structured_query = {
            "source_type": "csv",
            "source_path": KNOWN_VULNERABILITIES_PATH,
            "query": template["csv"] if template["csv"] else None,
        }
        
        visualization = {
            "type": "pie",
            "title": "Critical Vulnerabilities by Vendor",
            "color_by": "vendorProject",  # Assuming vendorProject is the vendor field
            "description": "Distribution of critical vulnerabilities by vendor"
        }
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "natural_language_query": query,
            "structured_query": structured_query,
            "visualization": visualization,
            "needs_clarification": False
        }
        
    elif "trend" in query_lower or "over time" in query_lower or "historical" in query_lower:
        query_type = "vulnerability_trend"
        confidence = 0.8
        
        # For trend analysis, we need to create a custom query since it requires grouping
        structured_query = {
            "source_type": "csv",
            "source_path": KNOWN_VULNERABILITIES_PATH,
            "fields": ["dateAdded", "cveID", "vendorProject"],  # Assumed field names
        }
        
        visualization = {
            "type": "line",
            "title": "Vulnerability Trends Over Time",
            "x_axis": "date",
            "y_axis": ["count"],
            "description": "Trend of vulnerability reports over time"
        }
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "natural_language_query": query,
            "structured_query": structured_query,
            "visualization": visualization,
            "needs_clarification": False
        }
    
    elif "vendor" in query_lower or "microsoft" in query_lower or "cisco" in query_lower or "adobe" in query_lower:
        query_type = "vendor_vulnerabilities"
        confidence = 0.85
        
        # Extract vendor name from query (simplified approach)
        import re
        vendor_match = re.search(r'(microsoft|cisco|adobe|oracle|apple|google)', query_lower)
        vendor = vendor_match.group(1) if vendor_match else "microsoft"  # Default to Microsoft if no match
        
        template = QUERY_TEMPLATES[query_type]
        structured_query = {
            "source_type": "csv",
            "source_path": KNOWN_VULNERABILITIES_PATH,
            "query": template["csv"].format(vendor=vendor.capitalize()) if template["csv"] else None,
        }
        
        visualization = {
            "type": "bar",
            "title": f"{vendor.capitalize()} Vulnerabilities",
            "x_axis": "dateAdded",  # Assuming dateAdded is the date field
            "y_axis": ["count"],
            "description": f"Vulnerabilities affecting {vendor.capitalize()} products"
        }
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "natural_language_query": query,
            "structured_query": structured_query,
            "visualization": visualization,
            "needs_clarification": False
        }
    
    # Default case: We need clarification
    return {
        "query_type": "unknown",
        "confidence": 0.3,
        "natural_language_query": query,
        "needs_clarification": True,
        "clarification_question": "I'm not sure what kind of vulnerability information you're looking for. Could you please specify if you want recent vulnerabilities, critical issues, trend analysis, or vulnerabilities for a specific vendor?"
    }

# Error handling functions
def handle_interpretation_error(query: str, error: Exception) -> QueryInterpretation:
    """
    Handle errors in query interpretation
    
    Args:
        query: Original natural language query
        error: Exception that occurred
        
    Returns:
        QueryInterpretation with error information
    """
    logger.error(f"Error interpreting query '{query}': {error}")
    
    return QueryInterpretation(
        query_type="error",
        confidence=0.0,
        natural_language_query=query,
        needs_clarification=True,
        clarification_question="I encountered an error while processing your request. Could you please rephrase your question?",
        error=str(error)
    )

async def get_data_source_info() -> List[DataSourceInfo]:
    """
    Get information about available data sources
    
    Returns:
        List of DataSourceInfo objects
    """
    # In a production environment, this would query the system for available data sources
    # and retrieve schema information for each source
    
    # For this example, we'll return mock data
    return [
        DataSourceInfo(
            source_type="sqlite",
            source_path="../database/vulnerabilities.db",
            tables=["vulnerabilities", "assets", "scans"],
            columns={
                "vulnerabilities": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "title", "type": "TEXT"},
                    {"name": "description", "type": "TEXT"},
                    {"name": "severity", "type": "TEXT"},
                    {"name": "cvss_score", "type": "REAL"},
                    {"name": "timestamp", "type": "TEXT"},
                    {"name": "status", "type": "TEXT"},
                    {"name": "asset_id", "type": "INTEGER"}
                ]
            },
            description="Main vulnerability database"
        ),
        DataSourceInfo(
            source_type="csv",
            source_path="../database/csv/monthly_reports.csv",
            columns={
                "monthly_reports": [
                    {"name": "month", "type": "TEXT"},
                    {"name": "year", "type": "INTEGER"},
                    {"name": "critical_count", "type": "INTEGER"},
                    {"name": "high_count", "type": "INTEGER"},
                    {"name": "medium_count", "type": "INTEGER"},
                    {"name": "low_count", "type": "INTEGER"}
                ]
            },
            description="Monthly vulnerability summary reports"
        )
    ]

async def interpret_query(
    query: str, 
    session_id: str, 
    timestamp: str, 
    previous_queries: Optional[List[str]] = None,
    user_preferences: Optional[Dict[str, Any]] = None
) -> QueryInterpretation:
    """
    Interpret a natural language query and convert it to a structured query
    
    Args:
        query: Natural language query
        session_id: Unique session identifier
        timestamp: Timestamp of the query
        previous_queries: Previous queries in this session
        user_preferences: User preferences for visualization
        
    Returns:
        QueryInterpretation object with structured query and visualization recommendation
    """
    try:
        # Construct query context
        context = QueryContext(
            original_query=query,
            session_id=session_id,
            timestamp=timestamp,
            previous_queries=previous_queries or [],
            user_preferences=user_preferences or {}
        )
        
        # Get information about available data sources
        data_sources = await get_data_source_info()
        
        # Interpret the query using LLM
        interpretation = await interpret_with_llm(query, data_sources, context)
        
        # Validate and return the interpretation
        return QueryInterpretation(**interpretation)
    
    except Exception as e:
        return handle_interpretation_error(query, e)

# LangGraph node function
async def process_natural_language_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to process natural language queries
    
    Args:
        state: Current state dictionary containing query information
        
    Returns:
        Updated state with query interpretation
    """
    # Extract query information from state
    query = state.get("natural_language_query")
    if not query:
        return {
            **state,
            "query_interpretation": {
                "query_type": "error",
                "confidence": 0.0,
                "natural_language_query": "",
                "needs_clarification": True,
                "clarification_question": "I didn't receive a query to process. Please provide a question about vulnerabilities.",
                "error": "No query provided"
            }
        }
    
    # Get session information from state
    session_id = state.get("session_id", "default_session")
    timestamp = state.get("timestamp", "")
    previous_queries = state.get("previous_queries", [])
    user_preferences = state.get("user_preferences", {})
    
    # Process the query
    interpretation = await interpret_query(
        query=query,
        session_id=session_id,
        timestamp=timestamp,
        previous_queries=previous_queries,
        user_preferences=user_preferences
    )
    
    # Update state with interpretation results
    return {
        **state,
        "query_interpretation": interpretation.dict(),
        "previous_queries": previous_queries + [query]
    }

async def handle_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to handle cases where clarification is needed
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state with clarification request
    """
    interpretation = state.get("query_interpretation", {})
    
    # If clarification is needed, prepare the request
    if interpretation.get("needs_clarification", False):
        return {
            **state,
            "needs_human_input": True,
            "human_input_prompt": interpretation.get("clarification_question", 
                                  "Could you please provide more details about your request?")
        }
    
    # No clarification needed, proceed with the workflow
    return state

# Function to extract structured query for data agent
def extract_query_for_data_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the structured query from the state for the data agent
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state with query info for data agent
    """
    interpretation = state.get("query_interpretation", {})
    structured_query = interpretation.get("structured_query")
    
    if not structured_query:
        return {
            **state,
            "query_info": None,
            "error": "No structured query available"
        }
    
    return {
        **state,
        "query_info": structured_query
    }
