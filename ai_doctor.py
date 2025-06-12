# AI Doctor - Final Working Version
import os
import gradio as gr
import tempfile
import time
import random
from brain_of_the_doctor import encode_image, analyze_image_with_query, generate_prescription
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

# System prompts in multiple languages
SYSTEM_PROMPTS = {
    "English": "You are a professional doctor providing medical advice in English...",
    "Hindi": "आप एक पेशेवर डॉक्टर हैं जो हिंदी में चिकित्सा सलाह दे रहे हैं...",
    "Marathi": "तुम्ही एक व्यावसायिक डॉक्टर आहात जे मराठीत वैद्यकीय सल्ला देत आहात..."
}

def get_unique_filename(extension):
    """Generate unique filename with timestamp and random number"""
    timestamp = int(time.time() * 1000)  # Milliseconds precision
    random_num = random.randint(1000, 9999)
    return f"temp_{timestamp}_{random_num}{extension}"

def process_inputs(audio, text, image, language, progress=gr.Progress()):
    try:
        progress(0.1, desc="Processing inputs...")
        
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

        # Generate audio with unique filename
        audio_file = get_unique_filename('.wav')
        text_to_speech_with_gtts(diagnosis, audio_file, language)
        
        # Generate prescription
        prescription = generate_prescription(diagnosis, language)
        prescription_file = get_unique_filename('.txt')
        with open(prescription_file, 'w', encoding='utf-8') as f:
            f.write(prescription)
            
        return (input_text, diagnosis, audio_file, prescription, 
                prescription_file, gr.DownloadButton(visible=True))
        
    except Exception as e:
        return (f"Error: {str(e)}",) * 4 + (None, gr.DownloadButton(visible=False))

# Gradio Interface
with gr.Blocks(title="AI Doctor") as app:
    with gr.Tabs():
        with gr.TabItem("Voice Input"):
            audio_input = gr.Audio(sources=["microphone"], type="filepath")
        with gr.TabItem("Text Input"):
            text_input = gr.Textbox(label="Describe symptoms")
    
    image_input = gr.Image(type="filepath", label="Upload medical image")
    language = gr.Dropdown(choices=["English", "Hindi", "Marathi"], value="English", label="Response Language")
    
    submit_btn = gr.Button("Get Diagnosis", variant="primary")
    
    # Outputs
    input_text = gr.Textbox(label="Your Input")
    diagnosis = gr.Textbox(label="Doctor's Diagnosis") 
    audio_output = gr.Audio(label="Audio Response")
    prescription = gr.Textbox(label="Prescription", interactive=True)
    download_btn = gr.DownloadButton("Download Prescription", visible=False)
    prescription_path = gr.State()

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, text_input, image_input, language],
        outputs=[input_text, diagnosis, audio_output, prescription, prescription_path, download_btn]
    )

    download_btn.click(
        lambda path: path,
        inputs=[prescription_path],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    app.launch()
