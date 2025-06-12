# AI Doctor - Final Working Version
import os
os.environ["GRADIO_TEMP_DIR"] = "D:/gradio_temp"
if not os.path.exists("D:/gradio_temp"):
    os.makedirs("D:/gradio_temp")

import gradio as gr
from gtts import gTTS
import numpy as np
from PIL import Image
import base64
import shutil
import tempfile
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from brain_of_the_doctor import (
    encode_image, 
    analyze_image_with_query, 
    generate_prescription,
    analyze_text_query
)
from voice_of_the_patient import record_audio, transcribe_with_groq

# Supported languages mapping with proper language codes
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

def image_to_base64(image_path):
    """Convert an image file to a base64 string, using a true temp file for short path safety."""
    try:
        # Create a temporary directory with a short path
        temp_dir = tempfile.mkdtemp(dir="D:/gradio_temp")
        _, ext = os.path.splitext(image_path)
        new_path = os.path.join(temp_dir, f"temp{ext}")
        
        # Copy the file to the new location
        shutil.copy2(image_path, new_path)
        
        # Read and encode the file
        with open(new_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        
        # Clean up
        try:
            os.remove(new_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")
            
        return encoded
    except Exception as e:
        logger.error(f"Error in image_to_base64: {str(e)}")
        return None

def text_to_speech_bytes(text, language='en'):
    """Convert text to speech bytes with proper error handling"""
    try:
        # Get the correct language code
        lang_code = LANGUAGE_CODES.get(language, 'en')
        print(f"gTTS input text: {text[:100]}...")  # Print first 100 chars
        print(f"gTTS language code: {lang_code}")
        tts = gTTS(text=text, lang=lang_code)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        print("gTTS audio generated successfully")
        return audio_bytes.getvalue()
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

def save_audio_to_temp_file(audio_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name

def process_inputs(text_input, audio_input, image_input, response_language):
    """Process all inputs and return diagnosis results"""
    try:
        # Validate inputs
        if not any([text_input, audio_input, image_input]):
            error_msg = "Please provide at least one input (text, voice, or image)."
            logger.error(error_msg)
            return error_msg, error_msg, None, error_msg, None
        
        # Get language code for text-to-speech
        language_code = LANGUAGE_CODES.get(response_language, "en")
        
        # Process audio input if provided
        if audio_input:
            try:
                audio_text = transcribe_with_groq(
                    stt_model="whisper-large-v3",
                    audio_filepath=audio_input,
                    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
                )
                if not audio_text:
                    error_msg = "Could not transcribe audio. Please try again or use text input."
                    logger.error(error_msg)
                    return error_msg, error_msg, None, error_msg, None
                text_input = audio_text
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                error_msg = f"Error processing audio: {str(e)}"
                return error_msg, error_msg, None, error_msg, None
        
        # Process text input
        if text_input:
            try:
                diagnosis = analyze_text_query(text_input, response_language)
                if not diagnosis:
                    error_msg = "Could not analyze text. Please try again."
                    logger.error(error_msg)
                    return error_msg, error_msg, None, error_msg, None
                prescription = generate_prescription(diagnosis, response_language)
                # Only use the first sentence for audio
                short_diagnosis = diagnosis.split('.')[0] + '.' if '.' in diagnosis else diagnosis
                audio_bytes = text_to_speech_bytes(short_diagnosis, language_code)
                audio_filepath = None
                if audio_bytes:
                    audio_filepath = save_audio_to_temp_file(audio_bytes)
                return text_input, diagnosis, audio_filepath, prescription, None
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                error_msg = f"Error processing text: {str(e)}"
                return text_input or error_msg, error_msg, None, error_msg, None
        
        # Process image input
        if image_input:
            try:
                image_base64 = image_to_base64(image_input)
                if not image_base64:
                    error_msg = "Could not process image. Please try again."
                    logger.error(error_msg)
                    return error_msg, error_msg, None, error_msg, None
                diagnosis = analyze_image_with_query(text_input or "Analyze this skin condition", image_base64, response_language)
                if not diagnosis:
                    error_msg = "Could not analyze image. Please try again."
                    logger.error(error_msg)
                    return error_msg, error_msg, None, error_msg, None
                prescription = generate_prescription(diagnosis, response_language)
                # Only use the first sentence for audio
                short_diagnosis = diagnosis.split('.')[0] + '.' if '.' in diagnosis else diagnosis
                audio_bytes = text_to_speech_bytes(short_diagnosis, language_code)
                audio_filepath = None
                if audio_bytes:
                    audio_filepath = save_audio_to_temp_file(audio_bytes)
                return text_input or "Image analysis", diagnosis, audio_filepath, prescription, None
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                error_msg = f"Error processing image: {str(e)}"
                return error_msg, error_msg, None, error_msg, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"An error occurred: {str(e)}\n{tb}")
        error_msg = f"An error occurred: {str(e)}"
        return error_msg, error_msg, None, error_msg, None

# Create the Gradio interface
with gr.Blocks(title="VAIDYA - Medical Diagnosis") as app:
    gr.Markdown("""
    # 🩺 AI Doctor - Medical Diagnosis System
    *Professional medical diagnosis powered by AI*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎤 Voice Input"):
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your symptoms",
                        interactive=True
                    )
                with gr.TabItem("✍️ Text Input"):
                    text_input = gr.Textbox(
                        label="Describe your symptoms",
                        placeholder="Type your symptoms here...",
                        lines=3,
                        interactive=True
                    )
            
            with gr.Accordion("🖼️ Upload Medical Image (Optional)", open=False):
                image_input = gr.Image(
                    type="filepath",
                    label="Medical Image",
                    interactive=True
                )
            
            with gr.Row():
                language = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="English",
                    label="Response Language",
                    scale=2,
                    interactive=True
                )
                submit_btn = gr.Button("🔍 Get Diagnosis", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Image(
                value="portrait-3d-female-doctor[1].jpg",
                label="Your Doctor",
                height=400,
                width=300,
                show_label=True,
                elem_classes="doctor-avatar"
            )
    
    # Output Section
    with gr.Column(elem_classes="output-section"):
        gr.Markdown("## 📋 Diagnosis Results")
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Your Input Summary",
                    interactive=False
                )
                with gr.Accordion("🩺 Detailed Diagnosis", open=True):
                    diagnosis = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False,
                        elem_classes="diagnosis-box"
                    )
                audio_output = gr.Audio(
                    label="🎧 Audio Diagnosis",
                    interactive=False,
                    type="filepath"
                )
            with gr.Column(scale=1):
                with gr.Accordion("💊 Prescription", open=True):
                    prescription = gr.Textbox(
                        label="",
                        lines=10,
                        interactive=True,
                        elem_classes="prescription-box"
                    )

    # Set up the submit button click event
    submit_btn.click(
        fn=process_inputs,
        inputs=[text_input, audio_input, image_input, language],
        outputs=[input_text, diagnosis, audio_output, prescription]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
