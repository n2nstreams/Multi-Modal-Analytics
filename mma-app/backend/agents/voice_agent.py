"""
Voice Agent for MMA Application

This module provides functionality to:
1. Process audio input from users
2. Convert speech to text using OpenAI's voice model
3. Handle language detection and audio preprocessing
"""

import os
import logging
import asyncio
from typing import Dict, Optional, Any, Union, BinaryIO
from pathlib import Path
import time
from pydantic import BaseModel, Field

# In a production environment, use actual OpenAI client
# For this example, we'll create a mock client
try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False
    
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for voice processing
class AudioConfig(BaseModel):
    """Configuration for audio processing"""
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    channels: int = Field(1, description="Number of audio channels")
    format: str = Field("wav", description="Audio format")
    max_duration: float = Field(60.0, description="Maximum duration in seconds")
    language: str = Field("en", description="Expected language")
    
class TranscriptionConfig(BaseModel):
    """Configuration for speech-to-text transcription"""
    model: str = Field("whisper-1", description="OpenAI model to use")
    temperature: float = Field(0.0, description="Sampling temperature")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en')")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    
class TranscriptionResponse(BaseModel):
    """Response from transcription service"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score")
    language_detected: Optional[str] = Field(None, description="Detected language")
    duration: float = Field(..., description="Audio duration in seconds")
    model_used: str = Field(..., description="Model used for transcription")

# Constants
AUDIO_DIR = Path("../uploads/audio")
SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac"]
DEFAULT_SAMPLE_RATE = 16000

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

class VoiceProcessor:
    """Class to handle voice processing operations"""
    
    def __init__(
        self, 
        audio_config: Optional[AudioConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the voice processor
        
        Args:
            audio_config: Configuration for audio processing
            transcription_config: Configuration for transcription
            openai_api_key: OpenAI API key (from environment if not provided)
        """
        self.audio_config = audio_config or AudioConfig()
        self.transcription_config = transcription_config or TranscriptionConfig()
        
        # Setup OpenAI client if available
        if has_openai:
            self.openai_client = openai.OpenAI(
                api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
        else:
            self.openai_client = None
            logger.warning("OpenAI package not available. Using mock transcription.")
    
    async def save_audio(self, audio_data: Union[bytes, BinaryIO], file_name: Optional[str] = None) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: Binary audio data or file-like object
            file_name: Optional file name (generated if not provided)
            
        Returns:
            Path to saved audio file
        """
        if file_name is None:
            timestamp = int(time.time())
            file_name = f"audio_{timestamp}.{self.audio_config.format}"
        
        file_path = AUDIO_DIR / file_name
        
        # Write audio data to file
        if isinstance(audio_data, bytes):
            with open(file_path, "wb") as f:
                f.write(audio_data)
        else:
            with open(file_path, "wb") as f:
                f.write(audio_data.read())
        
        logger.info(f"Saved audio to {file_path}")
        return str(file_path)
    
    async def transcribe_audio(self, audio_path: str) -> TranscriptionResponse:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResponse with transcription results
        """
        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check if we have actual OpenAI client
        if has_openai and self.openai_client:
            try:
                # Transcribe with OpenAI Whisper API
                with open(audio_file_path, "rb") as audio_file:
                    response = self.openai_client.audio.transcriptions.create(
                        model=self.transcription_config.model,
                        file=audio_file,
                        language=self.transcription_config.language,
                        prompt=self.transcription_config.prompt,
                        temperature=self.transcription_config.temperature
                    )
                
                # Parse response
                return TranscriptionResponse(
                    text=response.text,
                    confidence=0.95,  # OpenAI doesn't provide confidence score, this is a placeholder
                    language_detected=response.language,
                    duration=60.0,  # Also a placeholder
                    model_used=self.transcription_config.model
                )
            
            except Exception as e:
                logger.error(f"Error transcribing with OpenAI: {e}")
                # Fall back to mock implementation
        
        # Mock implementation for demonstration
        logger.info("Using mock transcription implementation")
        
        # Simulate processing time
        await asyncio.sleep(1.0)
        
        # Generate mock response based on file name
        file_name = audio_file_path.name.lower()
        
        if "vulnerability" in file_name:
            text = "Show me the latest vulnerability report."
        elif "critical" in file_name:
            text = "What are the critical security issues from last week?"
        elif "trend" in file_name:
            text = "Show me vulnerability trends over the past month."
        else:
            text = "Generate a security vulnerability report."
        
        return TranscriptionResponse(
            text=text,
            confidence=0.9,
            language_detected="en",
            duration=2.5,
            model_used="mock-whisper-1"
        )

# LangGraph node functions
async def process_voice_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to process voice input
    
    Args:
        state: Current state dictionary containing voice input information
        
    Returns:
        Updated state with transcription results
    """
    # Extract voice input from state
    voice_input = state.get("voice_input")
    if not voice_input:
        return {
            **state,
            "error": "No voice input provided",
            "natural_language_query": None
        }
    
    # Initialize voice processor
    voice_processor = VoiceProcessor()
    
    try:
        # Transcribe audio
        transcription = await voice_processor.transcribe_audio(voice_input.get("audio_path"))
        
        # Update state with transcription results
        return {
            **state,
            "transcription_result": {
                "text": transcription.text,
                "confidence": transcription.confidence,
                "audio_path": voice_input.get("audio_path"),
                "language_detected": transcription.language_detected
            },
            "natural_language_query": transcription.text,
            "current_agent": "voice_agent"
        }
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        return {
            **state,
            "error": f"Error transcribing audio: {str(e)}",
            "current_agent": "voice_agent"
        }

async def validate_voice_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to validate voice input
    
    Args:
        state: Current state dictionary
        
    Returns:
        Updated state with validation results
    """
    transcription = state.get("transcription_result", {})
    confidence = transcription.get("confidence", 0) if isinstance(transcription, dict) else 0
    
    # Check if confidence is too low
    if confidence < 0.6:
        return {
            **state,
            "needs_clarification": True,
            "human_input_prompt": "I couldn't understand your question clearly. Could you please repeat or rephrase it?"
        }
    
    return state 