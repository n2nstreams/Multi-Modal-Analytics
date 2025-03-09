# MMA - Multimodal Vulnerability Analysis

A multi-agent system using LangGraph to generate real-time vulnerability reports via voice commands.

## Overview

MMA (Multimodal Vulnerability Analysis) is a sophisticated system designed to provide security analysts with an intuitive, voice-controlled interface for generating vulnerability reports. Built with LangGraph for agent orchestration, it accepts voice commands, processes them with OpenAI's voice models, and generates appropriate visualizations based on the query.

## Key Features

- **Voice Command Interface**: Trigger analytical workflows using natural language voice commands
- **Multi-Agent Architecture**: Coordinated agent system using LangGraph for robust workflow orchestration
- **Data Flexibility**: Process data from SQLite databases and CSV files
- **Dynamic Visualizations**: Automatically select and generate the most appropriate visualization based on query content
- **Error Handling & Monitoring**: Comprehensive error handling and performance monitoring throughout the agent workflow

## System Architecture

MMA uses a multi-agent architecture with the following components:

1. **Voice Agent**: Captures voice commands and converts them to text using OpenAI's voice model
2. **Query Interpretation Agent**: Analyzes the natural language query and determines the appropriate data sources and visualization
3. **Data Query Agent**: Executes queries against SQLite and CSV data sources
4. **Visualization Agent**: Generates dynamic visualizations based on the data and query type
5. **Monitoring Agent**: Provides error handling, performance metrics, and system health information

## Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- OpenAI API key (for voice transcription)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mma.git
   cd mma
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

### Running the Application

1. Start the backend server:
   ```
   cd mma-app/backend
   python app.py
   ```

2. The API will be available at `http://localhost:8000`
3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

- `POST /api/query/text`: Submit a text query
- `POST /api/query/voice`: Submit a voice recording for processing
- `GET /api/health`: Get system health status
- `GET /api/errors`: Get recent system errors
- `GET /api/performance`: Get agent performance metrics

## Example Usage

### Voice Query

```bash
curl -X POST "http://localhost:8000/api/query/voice" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@recording.wav" \
  -F "session_id=optional-session-id"
```

### Text Query

```bash
curl -X POST "http://localhost:8000/api/query/text" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the latest vulnerability report",
    "session_id": "optional-session-id"
  }'
```

## Development

### Project Structure

```
mma-app/
├── backend/
│   ├── agents/
│   │   ├── data_agent.py       # Data query agent
│   │   ├── inference_agent.py  # Query interpretation agent
│   │   ├── monitoring_agent.py # System monitoring agent
│   │   └── voice_agent.py      # Voice processing agent
│   ├── api/                    # API-related modules
│   ├── database/               # Data storage modules
│   ├── static/                 # Static files (visualizations, etc.)
│   ├── graph.py                # LangGraph workflow definition
│   ├── state.py                # State models
│   └── app.py                  # FastAPI application
├── frontend/                   # Frontend code (not included in this version)
├── docs/                       # Documentation
└── requirements.txt            # Python dependencies
```

### Adding Custom Data Sources

1. Place SQLite databases in `backend/database/`
2. Place CSV files in `backend/database/csv/`
3. The system will automatically discover and make these available for querying

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [OpenAI API](https://openai.com/blog/openai-api) for voice transcription
- Visualizations with [Plotly](https://plotly.com/python/) and [Matplotlib](https://matplotlib.org/)
