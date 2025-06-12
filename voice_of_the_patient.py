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
import streamlit as st
import tempfile
import soundfile as sf
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def record_audio(file_path):
    """
    Record audio using Streamlit's built-in audio recorder
    """
    try:
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
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

# Step2: Setup Speech to text–STT–model for transcription
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
stt_model="whisper-large-v3"

def transcribe_with_groq(text):
    """
    Process the transcribed text using Groq API
    """
    try:
        # Create a chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant. Analyze the patient's symptoms and provide a detailed response."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing with Groq: {str(e)}")
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
                response = transcribe_with_groq(text)
                if response:
                    st.write("AI Analysis:", response)
                else:
                    st.error("Failed to process with AI")
            else:
                st.error("Failed to transcribe the audio")
        else:
            st.error("Failed to record audio")

if __name__ == "__main__":
    main()
