# AI Doctor - Production Ready Version
import io
import gradio as gr
from gtts import gTTS
import numpy as np
from brain_of_the_doctor import encode_image, analyze_image_with_query, generate_prescription
from voice_of_the_patient import record_audio, transcribe_with_groq

# Server configuration
SERVER_NAME = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 7860
SHARE = True  # Enable public sharing

# Language support
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

def text_to_speech_bytes(text, language='en'):
    """Convert text to speech bytes with proper error handling"""
    try:
        lang_code = LANGUAGE_CODES.get(language, 'en')
        tts = gTTS(text=text, lang=lang_code)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

SYSTEM_PROMPTS = {
    "English": "You are a professional doctor providing medical advice in English...",
    "Hindi": "आप एक पेशेवर डॉक्टर हैं जो हिंदी में चिकित्सा सलाह दे रहे हैं...",
    "Marathi": "तुम्ही एक व्यावसायिक डॉक्टर आहात जे मराठीत वैद्यकीय सल्ला देत आहात..."
}

def process_inputs(audio, text, image, language, progress=gr.Progress()):
    try:
        progress(0.1, desc="Processing...")
        
        # Input handling
        input_text = transcribe_with_groq(audio) if audio else text
        
        # Image analysis
        if image:
            progress(0.3, desc="Analyzing image...")
            encoded_image = encode_image(image)
            diagnosis = analyze_image_with_query(
                query=SYSTEM_PROMPTS[language],
                encoded_image=encoded_image
            )
        else:
            diagnosis = f"Response to: {input_text}"

        # Generate audio in memory
        audio_bytes = text_to_speech_bytes(diagnosis, language)
        if not audio_bytes:
            raise ValueError("Failed to generate audio response")
            
        # Generate prescription
        prescription = generate_prescription(diagnosis, language)
            
        return (input_text, diagnosis, (16000, np.frombuffer(audio_bytes, dtype=np.int16)), prescription, 
                gr.DownloadButton(visible=True))
        
    except Exception as e:
        return (f"Error: {str(e)}",) * 4 + (gr.DownloadButton(visible=False),)

with gr.Blocks(title="AI Doctor") as app:
    with gr.Tabs():
        with gr.TabItem("Voice Input"):
            audio_input = gr.Audio(sources=["microphone"], type="filepath")
        with gr.TabItem("Text Input"):
            text_input = gr.Textbox(label="Describe symptoms")
    
    image_input = gr.Image(type="filepath", label="Upload medical image")
    language = gr.Dropdown(
        choices=list(LANGUAGE_CODES.keys()), 
        value="English", 
        label="Response Language"
    )
    
    submit_btn = gr.Button("Get Diagnosis", variant="primary")
    
    # Outputs
    input_text = gr.Textbox(label="Your Input")
    diagnosis = gr.Textbox(label="Doctor's Diagnosis") 
    audio_output = gr.Audio(label="Audio Response")
    prescription = gr.Textbox(label="Prescription", interactive=True)
    download_btn = gr.DownloadButton("Download Prescription", visible=False)

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, text_input, image_input, language],
        outputs=[input_text, diagnosis, audio_output, prescription, download_btn]
    )

    download_btn.click(
        lambda text: (text, "prescription.txt"),
        inputs=[prescription],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE
    )
