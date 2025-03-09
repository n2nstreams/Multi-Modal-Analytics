"""
Visualization Agent for MMA Application

This module provides functionality to:
1. Generate appropriate visualizations based on vulnerability data
2. Render charts, graphs, and tables
3. Format visualizations for display
4. Optimize visual presentation of vulnerability data
"""


import base64
import logging
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_image
from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field, ValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")

# Constants
KNOWN_VULNERABILITIES_PATH = "../database/csv/known_exploited_vulnerabilities.csv"

# Define Pydantic models
class VisualizationRequest(BaseModel):
    """Model for visualization request"""
    viz_type: str = Field(..., description="Type of visualization (bar, line, pie, table, etc.)")
    title: str = Field(..., description="Title for the visualization")
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    x_axis: Optional[str] = Field(None, description="Field for x-axis")
    y_axis: Optional[Union[str, List[str]]] = Field(None, description="Field(s) for y-axis")
    color_by: Optional[str] = Field(None, description="Field to use for color differentiation")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to data")
    limit: Optional[int] = Field(None, description="Limit number of records to display")
    description: Optional[str] = Field(None, description="Description of what the visualization shows")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")
    
class VisualizationResult(BaseModel):
    """Model for visualization result"""
    success: bool = Field(..., description="Whether the visualization was created successfully")
    image_data: Optional[str] = Field(None, description="Base64 encoded visualization image")
    html: Optional[str] = Field(None, description="HTML representation of the visualization")
    error: Optional[str] = Field(None, description="Error message if visualization failed")
    viz_type: str = Field(..., description="Type of visualization that was created")
    title: str = Field(..., description="Title of the visualization")
    description: Optional[str] = Field(None, description="Description of the visualization")

# Core visualization functions
async def create_table_visualization(
    data: List[Dict[str, Any]],
    title: str,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a table visualization
        
        Args:
        data: Data to visualize
        title: Title for the visualization
        sort_by: Column to sort by
        limit: Maximum number of rows to display
        
    Returns:
        Dictionary with visualization result
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Apply sorting if specified
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)
        
        # Apply limit if specified
        if limit:
            df = df.head(limit)
        
        # Create a simple HTML table
        html_table = df.to_html(classes=["table", "table-striped", "table-hover"], index=False)
        styled_html = f"""
        <div class="table-container">
            <h3>{title}</h3>
            {html_table}
        </div>
        """
        
        return {
            "success": True,
            "html": styled_html,
            "viz_type": "table",
            "title": title,
            "description": kwargs.get("description", "Tabular display of vulnerability data")
        }
    except Exception as e:
        logger.error(f"Error creating table visualization: {e}")
        return {
            "success": False,
            "error": str(e),
            "viz_type": "table",
            "title": title
        }

async def create_bar_visualization(
    data: List[Dict[str, Any]],
    title: str,
    x_axis: str,
    y_axis: Optional[Union[str, List[str]]] = None,
    color_by: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a bar chart visualization
        
        Args:
        data: Data to visualize
        title: Title for the visualization
        x_axis: Field for x-axis
        y_axis: Field(s) for y-axis
        color_by: Field to use for color
            
        Returns:
        Dictionary with visualization result
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # For bar charts with vulnerability data, we often need to group and count
        if y_axis is None:
            # Count occurrences of x_axis values
            counts = df[x_axis].value_counts().reset_index()
            counts.columns = [x_axis, 'count']
            df = counts
            y_axis = 'count'
        
        # Create Plotly figure
        if color_by and color_by in df.columns:
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, title=title)
        else:
            fig = px.bar(df, x=x_axis, y=y_axis, title=title)
        
        # Customize layout
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis if isinstance(y_axis, str) else "Count",
            template="plotly_white"
        )
        
        # Convert to image
        img_bytes = to_image(fig, format="png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Also create HTML for interactive display
        html = fig.to_html(full_html=False)
        
        return {
            "success": True,
            "image_data": img_base64,
            "html": html,
            "viz_type": "bar",
            "title": title,
            "description": kwargs.get("description", f"Bar chart showing {x_axis} distribution")
        }
    except Exception as e:
        logger.error(f"Error creating bar visualization: {e}")
        return {
            "success": False,
            "error": str(e),
            "viz_type": "bar",
            "title": title
        }

async def create_pie_visualization(
    data: List[Dict[str, Any]],
    title: str,
    color_by: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a pie chart visualization
    
    Args:
        data: Data to visualize
        title: Title for the visualization
        color_by: Field to use for segments
        
    Returns:
        Dictionary with visualization result
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Count occurrences of the color_by field
        counts = df[color_by].value_counts().reset_index()
        counts.columns = [color_by, 'count']
        
        # Create Plotly figure
        fig = px.pie(counts, values='count', names=color_by, title=title)
        
        # Customize layout
        fig.update_layout(template="plotly_white")
        
        # Convert to image
        img_bytes = to_image(fig, format="png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Also create HTML for interactive display
        html = fig.to_html(full_html=False)
        
        return {
            "success": True,
            "image_data": img_base64,
            "html": html,
            "viz_type": "pie",
            "title": title,
            "description": kwargs.get("description", f"Pie chart showing distribution by {color_by}")
        }
    except Exception as e:
        logger.error(f"Error creating pie visualization: {e}")
        return {
            "success": False,
            "error": str(e),
            "viz_type": "pie",
            "title": title
        }

async def create_line_visualization(
    data: List[Dict[str, Any]],
    title: str,
    x_axis: str,
    y_axis: Optional[Union[str, List[str]]] = None,
    color_by: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a line chart visualization
    
    Args:
        data: Data to visualize
        title: Title for the visualization
        x_axis: Field for x-axis (typically a date)
        y_axis: Field(s) for y-axis
        color_by: Field to use for multiple lines
        
    Returns:
        Dictionary with visualization result
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # If x_axis is a date field, ensure it's properly parsed
        if x_axis.lower().endswith('date') or x_axis.lower().endswith('added') or x_axis.lower().endswith('published'):
            df[x_axis] = pd.to_datetime(df[x_axis])
        
        # For trend analysis, we typically need to group by date and count
        if y_axis is None or y_axis == 'count':
            # Group by date and count occurrences
            df = df.groupby(pd.Grouper(key=x_axis, freq='M')).size().reset_index(name='count')
            y_axis = 'count'
        
        # Create Plotly figure
        if color_by and color_by in df.columns:
            fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=title)
        else:
            fig = px.line(df, x=x_axis, y=y_axis, title=title)
        
        # Customize layout
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis if isinstance(y_axis, str) else "Count",
            template="plotly_white"
        )
        
        # Convert to image
        img_bytes = to_image(fig, format="png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Also create HTML for interactive display
        html = fig.to_html(full_html=False)
        
        return {
            "success": True,
            "image_data": img_base64,
            "html": html,
            "viz_type": "line",
            "title": title,
            "description": kwargs.get("description", f"Line chart showing trend of {y_axis} over time")
        }
    except Exception as e:
        logger.error(f"Error creating line visualization: {e}")
        return {
            "success": False,
            "error": str(e),
            "viz_type": "line",
            "title": title
        }

# Main function to create visualizations
async def create_visualization(request: VisualizationRequest) -> VisualizationResult:
    """
    Create a visualization based on the request
    
    Args:
        request: Visualization request details
        
    Returns:
        VisualizationResult with visualization data or error
    """
    # Extract request parameters
    viz_type = request.viz_type.lower()
    data = request.data
    title = request.title
    
    # Prepare additional parameters
    params = {
        "x_axis": request.x_axis,
        "y_axis": request.y_axis,
        "color_by": request.color_by,
        "sort_by": request.sort_by,
        "limit": request.limit,
        "description": request.description,
        "options": request.options
    }
    
    # Call appropriate visualization function based on type
    if viz_type == "table":
        result = await create_table_visualization(data, title, **params)
    elif viz_type == "bar":
        result = await create_bar_visualization(data, title, **params)
    elif viz_type == "pie":
        result = await create_pie_visualization(data, title, **params)
    elif viz_type == "line":
        result = await create_line_visualization(data, title, **params)
    else:
        result = {
            "success": False,
            "error": f"Unsupported visualization type: {viz_type}",
            "viz_type": viz_type,
            "title": title
        }
    
    # Convert result to VisualizationResult model
    return VisualizationResult(**result)

# LangGraph node function
async def generate_visualization(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to generate visualizations
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state with visualization results
    """
    # Extract data result and query interpretation from state
    data_result = state.get("data_result", {})
    query_interpretation = state.get("query_interpretation", {})
    
    # Check if we have data to visualize
    if not data_result.get("success", False) or not data_result.get("data"):
        return {
            **state,
            "visualization_result": {
                "success": False,
                "error": "No data available for visualization",
                "viz_type": "none",
                "title": "Visualization Error"
            },
            "status": "failed"
        }
    
    # Get visualization specs from query interpretation
    visualization_spec = query_interpretation.get("visualization", {})
    if not visualization_spec:
        # Default to table if no visualization specified
        visualization_spec = {
            "type": "table",
            "title": "Vulnerability Data",
            "description": "Tabular display of vulnerability data"
        }
    
    # Create visualization request
    request = VisualizationRequest(
        viz_type=visualization_spec.get("type", "table"),
        title=visualization_spec.get("title", "Vulnerability Data"),
        data=data_result.get("data", []),
        x_axis=visualization_spec.get("x_axis"),
        y_axis=visualization_spec.get("y_axis"),
        color_by=visualization_spec.get("color_by"),
        sort_by=visualization_spec.get("sort_by"),
        description=visualization_spec.get("description", "Visualization of vulnerability data")
    )
    
    # Update status
    state_with_status = {
        **state,
        "status": "generating_visualization"
    }
    
    try:
        # Generate the visualization
        result = await create_visualization(request)
        
        # Update state with visualization result
        return {
            **state_with_status,
            "visualization_result": result.dict(),
            "status": "completed" if result.success else "failed"
        }
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return {
            **state_with_status,
            "visualization_result": {
                "success": False,
                "error": str(e),
                "viz_type": request.viz_type,
                "title": request.title
            },
            "status": "failed"
        } 