"""
Test script for the MMA Vulnerability Analysis API
"""
import asyncio
import httpx
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

# Test queries
TEST_QUERIES = [
    "Show me the latest vulnerabilities",
    "What are the critical vulnerabilities in Microsoft products?",
    "Show me vulnerability trends over the past year",
    "List all Adobe vulnerabilities",
    "Tell me about CVE-2021-44228"  # Log4Shell vulnerability
]

# Ensure the known_vulnerabilities.csv exists
def check_data_file():
    data_path = Path("../database/csv/known_exploited_vulnerabilities.csv")
    if not data_path.exists():
        print(f"⚠️ Warning: Data file not found at {data_path}")
        # Create directories if they don't exist
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Provide instruction to download the data
        print("Please download the known_exploited_vulnerabilities.csv file from:")
        print("https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv")
        print(f"And place it at: {data_path}")
        return False
    
    print(f"✅ Data file found at {data_path}")
    return True

async def test_text_query(query):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/query/text",
                json={"query": query}
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"\n==== Results for: '{query}' ====")
            print(f"Status: {result.get('status')}")
            
            if result.get("needs_clarification"):
                print(f"Needs clarification: {result.get('clarification_prompt')}")
            
            if result.get("error"):
                print(f"Error: {result.get('error')}")
            
            # Print data results if available
            if "results" in result and "data_result" in result["results"]:
                data_result = result["results"]["data_result"]
                if data_result and data_result.get("success"):
                    data = data_result.get("data", [])
                    print(f"Data retrieved: {len(data)} records")
                    if data and len(data) > 0:
                        print("Sample data (first record):")
                        print(json.dumps(data[0], indent=2))
            
            # Show visualization type if available
            if "results" in result and "visualization_result" in result["results"]:
                viz_result = result["results"]["visualization_result"]
                if viz_result and viz_result.get("success"):
                    print(f"Visualization type: {viz_result.get('viz_type')}")
                    print(f"Visualization title: {viz_result.get('title')}")
            
            return result
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
            return None

async def test_status():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/api/status")
            response.raise_for_status()
            result = response.json()
            print(f"\n==== API Status ====")
            print(f"Status: {result.get('status')}")
            print(f"Version: {result.get('version')}")
            print(f"Timestamp: {result.get('timestamp')}")
            return result
        except Exception as e:
            print(f"❌ Error checking API status: {e}")
            return None

async def main():
    print("🔍 Testing MMA Vulnerability Analysis API")
    
    # Check if data file exists
    if not check_data_file():
        print("Cannot proceed with testing without the data file.")
        return
    
    # Check API status
    status = await test_status()
    if not status or status.get("status") != "operational":
        print("API not operational. Please start the API server with 'python app.py'")
        return
    
    # Test each query
    for query in TEST_QUERIES:
        await test_text_query(query)
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 