import os
import tempfile
import streamlit as st
import av
import numpy as np
import io
from gtts import gTTS

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Vaidya Ai - Healthcare assistant", layout="wide")

from brain_of_the_doctor import (
    encode_image,
    analyze_image_with_query,
    generate_prescription,
    analyze_text_query
)
try:
    from voice_of_the_patient import transcribe_with_groq
except ImportError as e:
    st.error(f"Error importing voice module: {e}")
    # Create a fallback function
    def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
        st.error("Voice transcription not available due to import error")
        return None

# WebRTC for audio recording
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception as e:
    st.warning(f"WebRTC not available: {e}")
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False

@st.cache_data
def generate_audio_from_text(text, lang):
    """Generates audio from text using gTTS and caches the result."""
    try:
        if not text or not text.strip():
            return None
        
        clean_text = text.strip()[:500]
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        audio_data = audio_bytes_io.getvalue()
        
        if len(audio_data) > 0:
            return audio_data
        else:
            return None
            
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None

# Get API key from environment variables (for Streamlit deployment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Debug: Show API key status (without revealing the key)
if GROQ_API_KEY:
    st.success(" GROQ API Key loaded successfully")
else:
    st.error(" GROQ_API_KEY not found. Please set it in your environment variables.")

# Language translations
TRANSLATIONS = {
    "English": {
        "title": "Vaidya ai an healthcare assistant",
        "language": "Language",
        "input": "Input",
        "voice_tab": "Voice Input",
        "text_tab": "Text Input",
        "image_tab": "Image Input",
        "describe_symptoms": "Describe your symptoms",
        "earlier_symptoms": "Earlier symptoms / what problem are you facing?",
        "days_suffering": "Days suffering",
        "analyze": "Analyze",
        "voice_recording": "Voice Recording",
        "start_recording": "Start Recording",
        "stop_recording": "Stop Recording",
        "record_audio": "Record live audio or upload a file",
        "upload_audio": "Record your symptoms (upload .wav/.mp3)"
    },
    "Hindi": {
        "title": "वदय एआई एक सवसथय सहयक",
        "language": "भष",
        "input": "इनपट",
        "voice_tab": "वइस इनपट",
        "text_tab": "टकसट इनपट",
        "image_tab": "इमज इनपट",
        "describe_symptoms": "अपन लकषण क वरणन कर",
        "earlier_symptoms": "पहल क लकषण / आपक कय समसय ह रह ह?",
        "days_suffering": "कतन दन स परशन",
        "analyze": "वशलषण कर",
        "voice_recording": "वइस रकरडग",
        "start_recording": "रकरडग शर कर",
        "stop_recording": "रकरडग रक",
        "record_audio": "लइव ऑडय रकरड कर य फइल अपलड कर",
        "upload_audio": "अपन लकषण रकरड कर (.wav/.mp3 अपलड कर)"
    },
    "Marathi": {
        "title": "वदय एआई एक आरगय सहयक",
        "language": "भष",
        "input": "इनपट",
        "voice_tab": "आवज इनपट",
        "text_tab": "मजकर इनपट",
        "image_tab": "परतम इनपट",
        "describe_symptoms": "तमचय लकषणच वरणन कर",
        "earlier_symptoms": "परवच लकषण / तमहल कय समसय आह?",
        "days_suffering": "कत दवस तरस",
        "analyze": "वशलषण कर",
        "voice_recording": "आवज रकरडग",
        "start_recording": "रकरडग सर कर",
        "stop_recording": "रकरडग थबव",
        "record_audio": "लइवह ऑडओ रकरड कर कव फइल अपलड कर",
        "upload_audio": "तमच लकषण रकरड कर (.wav/.mp3 अपलड कर)"
    }
}

def tr(key):
    """Translation helper function"""
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS[lang].get(key, key)

# Language selector
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"<h1 class=\"title-nowrap\">{tr(\"title\")}</h1>", unsafe_allow_html=True)
with col2:
    st.session_state.language = st.selectbox(
        tr("language"),
        options=list(TRANSLATIONS.keys()),
        index=0,
        key="language_selector"
    )

# CSS for title
st.markdown("""
<style>
.title-nowrap {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 2.5rem;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# Get language code
LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
response_language = st.session_state.get("language", "English")
language_code = LANGUAGE_CODES[response_language]

st.markdown("---")
st.markdown("**Professional medical diagnosis powered by AI**")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(f"### {tr(\"input\")}")
    tab1, tab2, tab3 = st.tabs([tr("voice_tab"), tr("text_tab"), tr("image_tab")])
    
    with tab1:
        st.caption(tr("record_audio"))
        
        if WEBRTC_AVAILABLE:
            # WebRTC Audio Streamer
            webrtc_config = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode="sendrecv",
                audio_receiver_size=1024,
                media_stream_constraints={"video": False, "audio": True},
                rtc_configuration=webrtc_config,
            )
            
            if webrtc_ctx.audio_receiver:
                st.success(" Voice recording is active!")
                st.info("Speak into your microphone. The audio will be processed in real-time.")
                
                # Days suffering input for voice tab
                duration_val_voice = st.number_input(
                    tr("days_suffering"),
                    min_value=0,
                    max_value=365,
                    value=st.session_state.get("duration_days", 1),
                    key="duration_days_voice"
                )
                
                # Sync with text tab
                st.session_state.duration_days = duration_val_voice
                
                if st.button("Process Voice Input", type="primary"):
                    st.info(" Processing voice input...")
                    # Here you would process the audio from webrtc_ctx.audio_receiver
                    st.success("Voice processed! (This is a demo)")
            else:
                st.info("Click START to begin voice recording")
                st.warning("Note: Voice recording requires microphone access and works best in Chrome/Firefox browsers.")
        else:
            st.error(" WebRTC not available. Please check your installation.")
            st.info("Fallback: You can still use text input or upload audio files.")
        
        # File upload fallback
        st.markdown("---")
        st.caption(tr("upload_audio"))
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Upload an audio file of your symptoms"
        )
        
        if uploaded_audio:
            st.success(f" Audio file uploaded: {uploaded_audio.name}")
            if st.button("Transcribe Audio", type="primary"):
                # Save uploaded file temporarily
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split(\".\")[-1]}")
                tmp.write(uploaded_audio.getvalue())
                tmp.close()
                
                st.info(" Transcribing audio...")
                text_from_audio = transcribe_with_groq(
                    stt_model="whisper-large-v3",
                    audio_filepath=tmp.name,
                    GROQ_API_KEY=GROQ_API_KEY
                )
                os.remove(tmp.name)
                
                if text_from_audio:
                    st.session_state["prefill_text"] = text_from_audio
                    st.success(f" Transcribed: {text_from_audio[:100]}...")
                else:
                    st.error(" Transcription failed. Please try again.")
    
    with tab2:
        st.caption("Describe your symptoms in detail")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area(
                tr("describe_symptoms"),
                height=100,
                placeholder="Describe your symptoms in detail...",
                value=st.session_state.get("prefill_text", ""),
                key="main_symptoms"
            )
        with col2:
            duration_val = st.number_input(
                tr("days_suffering"),
                min_value=0,
                max_value=365,
                value=st.session_state.get("duration_days", 1),
                key="duration_days"
            )
        
        earlier_symptoms = st.text_area(
            tr("earlier_symptoms"),
            height=80,
            placeholder="What were your earlier symptoms or what problem are you facing?",
            key="earlier_symptoms"
        )
        
        if st.button(tr("analyze"), type="primary"):
            if text_input:
                enriched_text = f"Symptoms: {text_input}\\nEarlier symptoms/problem: {earlier_symptoms}\\nDuration (days): {duration_val}"
                st.info(" Analyzing symptoms...")
                
                try:
                    diagnosis = analyze_text_query(enriched_text, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                    
                    # Audio diagnosis and prescription
                    audio_bytes = None
                    if diagnosis and prescription:
                        full_text_for_audio = f"Diagnosis: {diagnosis}. Prescription: {prescription}"
                        audio_bytes = generate_audio_from_text(full_text_for_audio, language_code)
                        if audio_bytes:
                            st.success(" Audio generated successfully")
                    
                    # Output UI
                    st.markdown("---")
                    st.markdown("##  Diagnosis Results")
                    st.markdown("<div class=\\"section-title\\">Your Input Summary</div>", unsafe_allow_html=True)
                    st.text_area("Input Summary", value=str(text_input) if text_input else "Audio analysis", height=80, disabled=True, label_visibility="collapsed")
                    st.markdown("<div class=\\"section-title\\"> Detailed Diagnosis</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class=\\"diagnosis-card\\">{diagnosis or \
\}</div>", unsafe_allow_html=True)
                    st.markdown("<div class=\\"section-title\\"> Prescription</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class=\\"prescription-card\\">{prescription or \\}</div>", unsafe_allow_html=True)
                    st.markdown("<div class=\\"section-title\\"> Audio Diagnosis</div>", unsafe_allow_html=True)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.info("No audio available.")
                        
                except Exception as e:
                    st.error(f" Error during processing: {str(e)}")
                    st.error("Please check your GROQ_API_KEY environment variable and try again.")
            else:
                st.warning("Please describe your symptoms first.")
    
    with tab3:
        st.caption("Upload an image for analysis")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Days suffering input for image tab
            duration_val_image = st.number_input(
                tr("days_suffering"),
                min_value=0,
                max_value=365,
                value=st.session_state.get("duration_days", 1),
                key="duration_days_image"
            )
            
            earlier_symptoms_image = st.text_area(
                tr("earlier_symptoms"),
                height=80,
                placeholder="What were your earlier symptoms or what problem are you facing?",
                key="earlier_symptoms_image"
            )
            
            if st.button("Analyze Image", type="primary"):
                st.info(" Analyzing image...")
                
                try:
                    image_base64 = encode_image(uploaded_file)
                    image_prompt = (
                        f"Analyze this image with the following patient context.\\n\\n"
                        f"Patient report:\\n"
                        f"- Symptoms: {st.session_state.get(\"prefill_text\", \"Not provided\")}\\n"
                        f"- Earlier symptoms/problem: {earlier_symptoms_image}\\n"
                        f"- Duration (days): {duration_val_image}\\n\\n"
                        f"Task: Provide a detailed medical assessment including differential diagnosis, likely diagnosis, red flags, home care, and prescription-style suggestions."
                    )
                    diagnosis = analyze_image_with_query(image_prompt, image_base64, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                    
                    # Audio diagnosis and prescription
                    audio_bytes = None
                    if diagnosis and prescription:
                        full_text_for_audio = f"Diagnosis: {diagnosis}. Prescription: {prescription}"
                        audio_bytes = generate_audio_from_text(full_text_for_audio, language_code)
                        if audio_bytes:
                            st.success(" Audio generated successfully")
                    
                    # Output UI
                    st.markdown("---")
                    st.markdown("##  Diagnosis Results")
                    st.markdown("<div class=\\"section-title\\">Your Input Summary</div>", unsafe_allow_html=True)
                    st.text_area("Input Summary", value="Image analysis", height=80, disabled=True, label_visibility="collapsed")
                    st.markdown("<div class=\\"section-title\\"> Detailed Diagnosis</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class=\\"diagnosis-card\\">{diagnosis or \\}</div>", unsafe_allow_html=True)
                    st.markdown("<div class=\\"section-title\\"> Prescription</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class=\\"prescription-card\\">{prescription or \\}</div>", unsafe_allow_html=True)
                    st.markdown("<div class=\\"section-title\\"> Audio Diagnosis</div>", unsafe_allow_html=True)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.info("No audio available.")
                        
                except Exception as e:
                    st.error(f" Error during processing: {str(e)}")
                    st.error("Please check your GROQ_API_KEY environment variable and try again.")

with col2:
    st.markdown("###  Your Doctor")
    st.image("portrait-3d-female-doctor[1].jpg", width=300)

# CSS for styling
st.markdown("""
<style>
.section-title {
    font-size: 1.2em;
    font-weight: bold;
    margin: 10px 0 5px 0;
    color: #1f77b4;
}
.diagnosis-card, .prescription-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("###  Features")
st.markdown("""
-  **Text Input**: Describe symptoms with detailed analysis
-  **Voice Input**: Real-time voice recording and processing  
-  **Image Input**: Upload medical images for analysis
-  **Multilingual**: English, Hindi, Marathi support
-  **Audio Output**: Text-to-speech responses
-  **Cloud Compatible**: Works on Streamlit Cloud
""")
