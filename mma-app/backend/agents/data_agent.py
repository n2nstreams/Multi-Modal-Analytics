"""
Data Query Agent for MMA Application

This module provides functionality to:
1. Connect to SQLite and CSV data sources
2. Execute queries
3. Retrieve and format data for further processing
"""

import os
import sqlite3
from typing import Dict, List, Union, Optional, Literal, Any
import pandas as pd
from pathlib import Path
import logging
from functools import wraps
from pydantic import BaseModel, Field, ValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for data validation
class DataQuery(BaseModel):
    """Model representing a data query request"""
    source_type: Literal["sqlite", "csv"] = Field(..., description="Type of data source")
    source_path: str = Field(..., description="Path to the data source")
    query: Optional[str] = Field(None, description="SQL query for SQLite or filter criteria for CSV")
    fields: Optional[List[str]] = Field(None, description="Fields to retrieve from CSV")

class DataResult(BaseModel):
    """Model representing data query results"""
    success: bool = Field(..., description="Whether the query was successful")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved data")
    error: Optional[str] = Field(None, description="Error message if query failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the query results")

# Error handling decorator
def handle_data_errors(func):
    """Decorator to handle errors in data operations"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return DataResult(success=False, error=f"Database error: {str(e)}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return DataResult(success=False, error=f"Data source not found: {str(e)}")
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return DataResult(success=False, error=f"Data validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return DataResult(success=False, error=f"Unexpected error: {str(e)}")
    return wrapper

# SQLite operations
async def connect_to_sqlite(db_path: str) -> Union[sqlite3.Connection, None]:
    """
    Create an asynchronous connection to SQLite database
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        SQLite connection object or None if connection fails
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return None
    
    try:
        # SQLite doesn't support true async, but we can use this pattern
        # for consistency with other async functions
        return sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

@handle_data_errors
async def query_sqlite(request: DataQuery) -> DataResult:
    """
    Execute a query on SQLite database
    
    Args:
        request: DataQuery object containing query details
        
    Returns:
        DataResult object with query results or error
    """
    if not request.query:
        return DataResult(
            success=False, 
            error="Query is required for SQLite data source"
        )
    
    connection = await connect_to_sqlite(request.source_path)
    if not connection:
        return DataResult(
            success=False, 
            error=f"Failed to connect to database: {request.source_path}"
        )
    
    try:
        # Make the connection return rows as dictionaries
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(request.query)
        rows = cursor.fetchall()
        
        # Convert rows to list of dictionaries
        result_data = [dict(row) for row in rows]
        
        # Get column names for metadata
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        return DataResult(
            success=True,
            data=result_data,
            metadata={
                "row_count": len(result_data),
                "columns": column_names,
                "source_type": "sqlite",
                "source_path": request.source_path
            }
        )
    finally:
        connection.close()

# CSV operations
@handle_data_errors
async def query_csv(request: DataQuery) -> DataResult:
    """
    Query data from a CSV file
    
    Args:
        request: DataQuery object containing query details
        
    Returns:
        DataResult object with query results or error
    """
    if not os.path.exists(request.source_path):
        return DataResult(
            success=False, 
            error=f"CSV file not found: {request.source_path}"
        )
    
    try:
        # Use pandas for efficient CSV processing
        df = pd.read_csv(request.source_path)
        
        # Apply field filtering if specified
        if request.fields:
            valid_fields = [field for field in request.fields if field in df.columns]
            if not valid_fields:
                return DataResult(
                    success=False,
                    error=f"None of the requested fields {request.fields} exist in the CSV"
                )
            df = df[valid_fields]
        
        # Apply query filtering if provided (simple query string as Python expression)
        if request.query:
            try:
                filtered_df = df.query(request.query)
                if len(filtered_df) == 0:
                    logger.warning(f"Query '{request.query}' returned no results")
                df = filtered_df
            except Exception as e:
                return DataResult(
                    success=False,
                    error=f"Failed to apply filter query: {str(e)}"
                )
        
        # Convert to list of dictionaries
        result_data = df.to_dict(orient='records')
        
        return DataResult(
            success=True,
            data=result_data,
            metadata={
                "row_count": len(result_data),
                "columns": df.columns.tolist(),
                "source_type": "csv",
                "source_path": request.source_path
            }
        )
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return DataResult(success=False, error=f"Error processing CSV: {str(e)}")

# Main handler function for data queries
async def process_data_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a data query request
    
    Args:
        query_data: Dictionary containing query parameters
        
    Returns:
        Dictionary with query results
    """
    try:
        # Validate the input data
        request = DataQuery(**query_data)
        
        # Route to appropriate handler based on source type
        if request.source_type == "sqlite":
            result = await query_sqlite(request)
        elif request.source_type == "csv":
            result = await query_csv(request)
        else:
            result = DataResult(
                success=False,
                error=f"Unsupported data source type: {request.source_type}"
            )
        
        return result.dict()
    except ValidationError as e:
        logger.error(f"Invalid query data: {e}")
        return DataResult(
            success=False,
            error=f"Invalid query data: {str(e)}"
        ).dict()
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        return DataResult(
            success=False,
            error=f"Unexpected error: {str(e)}"
        ).dict()

# LangGraph node function
async def query_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to query data sources
    
    Args:
        state: Current state dictionary containing query information
        
    Returns:
        Updated state with query results
    """
    # Extract query information from state
    query_info = state.get("query_result", {})
    
    # Always include mock data for testing purposes
    logger.warning("Using mock data for testing.")
    mock_data = [
        {
            "cve_id": "CVE-2021-44228",
            "vendor": "Apache",
            "product": "Log4j",
            "vulnerability_name": "Log4Shell",
            "description": "Apache Log4j2 JNDI features do not protect against attacker controlled LDAP and other JNDI related endpoints",
            "severity": "Critical",
            "published_date": "2021-12-10",
            "fixed_date": "2021-12-15"
        },
        {
            "cve_id": "CVE-2022-22965",
            "vendor": "Microsoft",
            "product": "Windows",
            "vulnerability_name": "Spring4Shell",
            "description": "A Spring MVC or Spring WebFlux application running on JDK 9+ may be vulnerable to remote code execution",
            "severity": "Critical",
            "published_date": "2022-03-31",
            "fixed_date": "2022-04-01"
        },
        {
            "cve_id": "CVE-2023-12345",
            "vendor": "Adobe",
            "product": "Acrobat Reader",
            "vulnerability_name": "PDF RCE",
            "description": "Remote code execution vulnerability in Adobe Acrobat Reader",
            "severity": "High",
            "published_date": "2023-05-15",
            "fixed_date": "2023-05-20"
        }
    ]
    
    # Filter mock data based on query if available
    filtered_data = mock_data
    if query_info.get("success") and query_info.get("parsed_query"):
        parsed_query = query_info.get("parsed_query", {})
        filters = parsed_query.get("filters", {})
        
        # Apply filters
        if filters.get("vendor"):
            vendor = filters.get("vendor").lower()
            filtered_data = [v for v in filtered_data if v.get("vendor", "").lower() == vendor]
            
        if filters.get("severity"):
            severity = filters.get("severity").lower()
            filtered_data = [v for v in filtered_data if v.get("severity", "").lower() == severity]
            
        if filters.get("cve_id"):
            cve_id = filters.get("cve_id").upper()
            filtered_data = [v for v in filtered_data if v.get("cve_id", "").upper() == cve_id]
    
    return {
        **state,
        "data_result": {
            "success": True,
            "data": filtered_data,
            "count": len(filtered_data),
            "query_info": query_info.get("parsed_query", {})
        }
    }

# Add a constant for the test data path
KNOWN_VULNERABILITIES_PATH = "../database/csv/known_exploited_vulnerabilities.csv"

# Update the get_available_data_sources function
def get_available_data_sources() -> Dict[str, List[str]]:
    """
    Get available data sources in the system
    
    Returns:
        Dictionary mapping source types to lists of available sources
    """
    # For testing, we'll explicitly include our known vulnerabilities file
    database_dir = Path("../database")
    csv_dir = Path("../database/csv")
    
    # Find SQLite databases
    sqlite_files = []
    if database_dir.exists():
        sqlite_files = [str(f) for f in database_dir.glob("*.db") + database_dir.glob("*.sqlite")]
    
    # Find CSV files, ensuring our test file is included
    csv_files = [KNOWN_VULNERABILITIES_PATH]
    if csv_dir.exists():
        additional_csvs = [str(f) for f in csv_dir.glob("*.csv") if str(f) != KNOWN_VULNERABILITIES_PATH]
        csv_files.extend(additional_csvs)
    
    return {
        "sqlite": sqlite_files,
        "csv": csv_files
    }

async def get_schema_info(data_source: str, source_type: str) -> Dict[str, Any]:
    """
    Get schema information for a data source
    
    Args:
        data_source: Path to the data source
        source_type: Type of data source (sqlite or csv)
        
    Returns:
        Dictionary with schema information
    """
    if source_type == "sqlite":
        connection = await connect_to_sqlite(data_source)
        if not connection:
            return {"error": f"Failed to connect to database: {data_source}"}
        
        try:
            cursor = connection.cursor()
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {}
            # Get column info for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
                schema_info[table] = columns
                
            return {
                "success": True,
                "source_type": "sqlite",
                "source_path": data_source,
                "tables": tables,
                "schema": schema_info
            }
        finally:
            connection.close()
    
    elif source_type == "csv":
        if not os.path.exists(data_source):
            return {"error": f"CSV file not found: {data_source}"}
        
        try:
            # Read CSV header to get column names
            df = pd.read_csv(data_source, nrows=0)
            columns = df.columns.tolist()
            
            # Get column types by sampling a few rows
            df_sample = pd.read_csv(data_source, nrows=5)
            column_types = {col: str(df_sample[col].dtype) for col in columns}
            
            return {
                "success": True,
                "source_type": "csv",
                "source_path": data_source,
                "columns": columns,
                "column_types": column_types
            }
        except Exception as e:
            return {"error": f"Error reading CSV schema: {str(e)}"}
    
    else:
        return {"error": f"Unsupported data source type: {source_type}"}
