# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup Audio recorder (ffmpeg & portaudio)
# ffmpeg, portaudio, pyaudio
import logging
import speech_recognition as sr
import os
from groq import Groq
import numpy as np
import tempfile
import soundfile as sf
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to get API key from environment variables first
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq client initialized successfully")
else:
    client = None
    print("❌ Cannot initialize Groq client - no API key available")

def get_api_key():
    """Get API key from Streamlit secrets or environment variables"""
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except (ImportError, KeyError, AttributeError):
        return os.getenv("GROQ_API_KEY")

def record_audio(file_path):
    """
    Record audio using Streamlit's built-in audio recorder
    """
    try:
        import streamlit as st
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        # Use Streamlit's audio recorder
        audio_bytes = st.audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            sample_rate=16000
        )

        if audio_bytes is not None:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Save the audio using soundfile
            sf.write(temp_path, audio_array, 16000)
            
            # Move the temporary file to the desired location
            os.replace(temp_path, file_path)
            logger.info(f"Audio recorded and saved to {file_path}")
            return True
        else:
            logger.warning("No audio was recorded")
            return False

    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

# Step2: Setup Speech to text–STT–model for transcription
stt_model="whisper-large-v3"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY=None):
    """
    Transcribe audio using Groq API with Whisper model
    """
    try:
        # Get API key if not provided
        if not GROQ_API_KEY:
            GROQ_API_KEY = get_api_key()
        
        if not GROQ_API_KEY:
            raise ValueError("No API key provided for transcription")
        
        # Read the audio file
        with open(audio_filepath, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Create Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # Transcribe using Whisper model
        response = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_data,
            response_format="text"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error transcribing with Groq: {str(e)}")
        if "401" in str(e) or "invalid_api_key" in str(e).lower():
            logger.error("API key is invalid or expired")
        elif "403" in str(e):
            logger.error("Insufficient permissions or quota")
        elif "429" in str(e):
            logger.error("Rate limit exceeded")
        return None

def transcribe_audio(file_path):
    """
    Transcribe audio using Google Speech Recognition
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None

def main():
    try:
        import streamlit as st
        st.title("Voice of the Patient")
        
        # Create a directory for audio files if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        # Record audio
        if st.button("Start Recording"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("recordings", f"recording_{timestamp}.wav")
            
            if record_audio(file_path):
                st.success("Recording completed!")
                
                # Transcribe the audio
                text = transcribe_audio(file_path)
                if text:
                    st.write("Transcription:", text)
                    
                    # Process with Groq
                    response = transcribe_with_groq(stt_model, file_path)
                    if response:
                        st.write("AI Analysis:", response)
                    else:
                        st.error("Failed to process with AI")
                else:
                    st.error("Failed to transcribe the audio")
            else:
                st.error("Failed to record audio")
    except ImportError:
        print("Streamlit not available for main function")

if __name__ == "__main__":
    main()
