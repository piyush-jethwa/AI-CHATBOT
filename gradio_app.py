#VoiceBot UI with Gradio
import os
import gradio as gr
import numpy as np
from pydub import AudioSegment

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
from custom_avatar import SpeakingAvatar

system_prompt="""You are a professional doctor providing medical advice. 
            Analyze this image and identify any medical issues. 
            Provide your diagnosis and suggested remedies in {language}.
            Respond conversationally as if speaking directly to the patient.
            Use phrases like 'With what I see, I think you have...'
            Keep your response concise (2-3 sentences maximum).
            Important: Respond in {language} only."""

def check_browser_permissions():
    """Check if browser has microphone permissions"""
    try:
        # More robust check that actually verifies microphone access
        import pyaudio
        p = pyaudio.PyAudio()
        default_input = p.get_default_input_device_info()
        if not default_input:
            raise ValueError("No microphone found")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Microphone check failed: {str(e)}")
        return False

import concurrent.futures

def process_inputs(input_data, image_filepath, language="English", voice_pack="default", progress=gr.Progress()):
    # Handle both audio and text input cases
    if isinstance(input_data, dict):  # Audio input case from Gradio
        # Extract the actual file path from Gradio's audio dict
        audio_filepath = input_data.get("name") if isinstance(input_data, dict) else None
        if not audio_filepath or not os.path.exists(audio_filepath):
            raise ValueError("Recorded audio file not found. Please try recording again.")
        text_input = None
    else:  # Text input case
        audio_filepath = None
        text_input = input_data
    progress(0.1, desc="Initializing...")
    # Check browser permissions first
    if not check_browser_permissions():
        raise ValueError("Please allow microphone access in your browser")
    try:
        if not audio_filepath and not text_input:
            raise ValueError("No input detected. Please record or type your query.")
            
        if audio_filepath and not os.path.exists(audio_filepath):
            raise ValueError("""
            Recording failed. Possible causes:
            1. Microphone access not granted
            2. No microphone detected
            3. Audio driver issues
            4. Background application using microphone
            
            Please check your microphone settings and try again.
            """)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Handle input based on type
            if audio_filepath:
                # Submit speech-to-text task
                stt_future = executor.submit(
                    transcribe_with_groq,
                    GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                    audio_filepath=audio_filepath,
                    stt_model="whisper-large-v3"
                )
            else:
                # Use text input directly
                stt_future = executor.submit(lambda: text_input)
            
            # Submit image analysis task if image provided
            image_future = None
            if image_filepath and os.path.exists(image_filepath):
                encoded_image = encode_image(image_filepath)
                image_future = executor.submit(
                    analyze_image_with_query,
                    query=system_prompt.format(language=language),
                    encoded_image=encoded_image,
                    model="llama-3.2-11b-vision-preview"
                )
            
            # Get speech-to-text result
            progress(0.4, desc="Processing speech...")
            try:
                speech_to_text_output = stt_future.result()
                if not speech_to_text_output.strip():
                    raise ValueError("Speech recognition returned empty result")
            except Exception as e:
                raise ValueError(f"Speech recognition failed: {str(e)}")

            # Get image analysis result if submitted
            doctor_response = "No image provided for analysis"
            if image_future:
                progress(0.6, desc="Analyzing image...")
                try:
                    doctor_response = image_future.result()
                    if not doctor_response.strip():
                        doctor_response = "Received empty analysis response"
                except Exception as e:
                    doctor_response = f"⚠️ Image analysis error: {str(e)}"

            # Generate voice response
            progress(0.8, desc="Generating response...")
            try:
                # Generate unique filename for each response
                output_file = f"response_{hash(doctor_response)}.wav"
                
                try:
                    if voice_pack == "Human Male":
                        text_to_speech_with_gtts(
                            input_text=doctor_response,
                            output_filepath=output_file,
                            language=language,
                            voice_pack="human_male"
                        )
                    else:  # AI Voice options
                        voice_map = {
                            "Professional (AI)": "professional",
                            "Friendly (AI)": "friendly",
                            "Serious (AI)": "serious",
                            "Compassionate (AI)": "compassionate"
                        }
                        text_to_speech_with_elevenlabs(
                            input_text=doctor_response,
                            output_filepath=output_file,
                            voice=voice_map.get(voice_pack, "professional")
                        )
                    
                    if not os.path.exists(output_file):
                        raise ValueError("Audio file was not generated")
                        
                    # Ensure file is readable
                    AudioSegment.from_wav(output_file)
                except Exception as e:
                    print(f"Voice generation failed, falling back to gTTS: {str(e)}")
                    text_to_speech_with_gtts(
                        input_text=doctor_response,
                        output_filepath=output_file,
                        language=language
                    )
            except Exception as e:
                print(f"Voice generation error: {str(e)}")
                raise

            # Generate avatar
            try:
                avatar = SpeakingAvatar()
                speaking_avatar = avatar.get_avatar(doctor_response)
                if not isinstance(speaking_avatar, np.ndarray):
                    raise ValueError("Avatar image not generated properly")
                return speaking_avatar, speech_to_text_output, doctor_response, output_file
            except Exception as e:
                print(f"Avatar error: {str(e)}")
                default_img = np.array(Image.new('RGB', (300, 300), (255,255,255)))
                return default_img, speech_to_text_output, doctor_response, output_file

    except Exception as e:
        error_msg = f"""
        🚨 Processing error: {str(e)}
        
        Troubleshooting steps:
        1. Refresh the page and allow microphone access
        2. Check if another app is using the microphone
        3. Test your microphone in another application
        4. Restart your browser if issues persist
        """
        avatar = SpeakingAvatar()
        error_avatar = avatar.get_avatar(error_msg)
        return error_avatar, error_msg, error_msg, None

# # Create the interface with image, voice, and text input options
# custom_css = """
# :root {
#     --primary: #3b82f6;
#     --primary-hover: #2563eb;
#     --bg-dark: #0f172a;
#     --bg-surface: #1e293b;
#     --text-light: #e2e8f0;
#     --border: #334155;
#     --radius: 10px;
#     --shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
#     --font: 'Inter', sans-serif;
# }

# /* Base styles */
# body, .gradio-container {
#     background-color: var(--bg-dark) !important;
#     color: var(--text-light) !important;
#     font-family: var(--font);
#     padding: 20px;
# }

# /* Buttons */
# button {
#     background-color: var(--primary) !important;
#     color: white !important;
#     border: none !important;
#     padding: 12px 20px !important;
#     border-radius: var(--radius) !important;
#     font-weight: 600 !important;
#     font-size: 15px !important;
#     box-shadow: var(--shadow);
#     transition: background-color 0.2s ease;
# }

# button:hover {
#     background-color: var(--primary-hover) !important;
# }

# /* Inputs and dropdowns */
# textarea, input, select {
#     background-color: var(--bg-surface) !important;
#     border: 1px solid var(--border) !important;
#     color: var(--text-light) !important;
#     border-radius: var(--radius) !important;
#     padding: 10px !important;
#     font-size: 15px !important;
#     width: 100%;
# }

# /* Component blocks */
# .gr-box, .gr-textbox, .gr-audio, .gr-image, .gr-dropdown {
#     background-color: var(--bg-surface) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: var(--radius) !important;
#     padding: 12px;
#     box-shadow: var(--shadow);
#     margin-top: 10px;
# }

# /* Labels */
# label {
#     font-weight: 600;
#     font-size: 14px;
#     margin-bottom: 6px;
#     display: block;
# }

# /* Image & Audio players */
# .gr-image img {
#     border-radius: var(--radius) !important;
#     border: 1px solid var(--border);
# }

# audio {
#     width: 100% !important;
#     margin-top: 10px;
# }

# /* Tabs and layout */
# .gr-tabs {
#     margin-bottom: 16px;
# }

# @media (max-width: 768px) {
#     button {
#         width: 100% !important;
#     }
# }
# """




with gr.Blocks(
    title="AI Doctor with Vision and Voice",
    css=custom_css,
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="gray"
    )
) as app:
    with gr.Tabs():
        with gr.TabItem("Voice Input"):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Press and Hold to Record",
                interactive=True
            )
        with gr.TabItem("Text Input"):
            text_input = gr.Textbox(
                label="Or Type Your Query",
                placeholder="Type your medical question here...",
                lines=3
            )
    
    image_input = gr.Image(type="filepath", label="Upload Medical Image")
    language = gr.Dropdown(
        choices=["English", "Hindi", "Marathi"],
        value="English",
        label="Response Language"
    )
    voice_pack = gr.Dropdown(
        choices=["Professional (AI)", "Friendly (AI)", "Serious (AI)", "Compassionate (AI)", "Human Male"],
        value="Professional (AI)", 
        label="Select Voice Type",
        info="Human voice requires pre-recorded samples"
    )
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        avatar_output = gr.Image(label="Doctor Avatar", visible=True)
    with gr.Row():
        stt_output = gr.Textbox(label="Speech to Text")
    with gr.Row():
        response_output = gr.Textbox(label="Doctor's Response")
    with gr.Row():
        audio_output = gr.Audio(label="Doctor Audio")
    with gr.Row():
        prescription_output = gr.Textbox(label="Prescription", visible=False)

    def process_combined_inputs(audio, text, image, lang, voice, progress=gr.Progress()):
        progress(0.1, desc="Initializing analysis...")
        
        try:
            # Handle image input completely independently
            if image is not None:
                progress(0.3, desc="Encoding image...")
                encoded_image = encode_image(image)
                
                progress(0.5, desc="Analyzing medical image...")
                doctor_response = analyze_image_with_query(
                    query=system_prompt.format(language=lang),
                    encoded_image=encoded_image,
                    model="llama-3.2-11b-vision-preview"
                )
                stt_output = "Automatic image analysis"
                
                # Generate response for image
                progress(0.7, desc="Generating response...")
                output_file = f"response_{hash(doctor_response)}.wav"
                
                # Fix for Hindi description
                if lang == "Hindi":
                    doctor_response = doctor_response.replace("Dandruff", "रूसी")
                    doctor_response = doctor_response.replace("dandruff", "रूसी")
                
                text_to_speech_with_gtts(
                    input_text=doctor_response,
                    output_filepath=output_file,
                    language=lang
                )
                
                return None, stt_output, doctor_response, output_file
            
            # Handle voice/text input independently
            elif audio is not None or text is not None:
                input_data = audio if audio is not None else text
                return process_inputs(input_data, image, lang, voice, progress)
                
            else:
                error_msg = "Please provide either an image, voice recording, or text input"
                avatar = SpeakingAvatar()
                error_avatar = avatar.get_avatar(error_msg)
                return error_avatar, error_msg, error_msg, None
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            avatar = SpeakingAvatar()
            error_avatar = avatar.get_avatar(error_msg)
            return error_avatar, error_msg, error_msg, None
        
    submit_btn.click(
        fn=process_combined_inputs,
        inputs=[audio_input, text_input, image_input, language, voice_pack],
        outputs=[avatar_output, stt_output, response_output, audio_output]
    )

app.launch(debug=True, share=True)
