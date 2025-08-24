import os
import tempfile
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="VAIDYA - Medical Diagnosis", layout="wide")

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

from gtts import gTTS
import base64
import io

@st.cache_data
def generate_audio_from_text(text, lang):
    """Generates audio from text using gTTS and caches the result."""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.getvalue()
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None

# Get API key from environment variables (for Streamlit deployment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Debug: Show API key status (without revealing the key)
if GROQ_API_KEY:
    st.info(f"🔑 API Key Status: Found (Length: {len(GROQ_API_KEY)} characters)")
    if GROQ_API_KEY.startswith("gsk_"):
        st.success("✅ API Key format looks correct (starts with 'gsk_')")
    else:
        st.warning("⚠️ API Key format may be incorrect (should start with 'gsk_')")
else:
    st.error("❌ No API key found! Please set GROQ_API_KEY environment variable in Streamlit deployment settings.")

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .stTextArea textarea {font-size: 1rem;}
    .diagnosis-card, .prescription-card {
        background: #22232b;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #fff;
        border: 1px solid #444;
    }
    .section-title {color: #ff9800; font-weight: bold; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🩺 AI Doctor - Medical Diagnosis System")
st.markdown("*Professional medical diagnosis powered by AI*")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### Input")
    tab1, tab2 = st.tabs(["🎤 Voice Input", "✍️ Text Input"])
    with tab1:
        audio_input = st.file_uploader("Record your symptoms (upload .wav/.mp3)", type=["wav", "mp3"])
        if audio_input is None:
            st.info("Please upload an audio file or use text input")
    with tab2:
        text_input = st.text_area("Describe your symptoms", placeholder="Type your symptoms here...", height=80)
    image_input = st.file_uploader("Upload Medical Image (Optional)", type=["jpg", "jpeg", "png", "webp"])
    response_language = st.selectbox("Response Language", list(LANGUAGE_CODES.keys()), index=0)
    submit_btn = st.button("🔍 Get Diagnosis", use_container_width=True)

with col2:
    st.markdown("### Your Doctor")
    st.image("portrait-3d-female-doctor[1].jpg", caption="Your Doctor", use_container_width=True)

# Output section
if submit_btn:
    if not GROQ_API_KEY:
        st.error("❌ Cannot proceed without API key. Please add GROQ_API_KEY to Streamlit secrets.")
    else:
        with st.spinner("Processing..."):
            try:
                # Audio input handling
                if audio_input is not None:
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_input.name)[-1])
                    temp_audio.write(audio_input.read())
                    temp_audio.close()
                    audio_path = temp_audio.name
                    st.info("🎤 Processing audio input...")
                    text_input = transcribe_with_groq(
                        stt_model="whisper-large-v3",
                        audio_filepath=audio_path,
                        GROQ_API_KEY=GROQ_API_KEY
                    )
                    os.remove(audio_path)
                    if text_input:
                        st.success(f"✅ Audio transcribed: {text_input[:100]}...")
                
                # Image input handling
                image_base64 = None
                if image_input is not None:
                    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_input.name)[-1])
                    temp_image.write(image_input.read())
                    temp_image.close()
                    image_base64 = encode_image(temp_image.name)
                    os.remove(temp_image.name)
                    st.success("🖼️ Image processed successfully")
                
                # Diagnosis logic
                diagnosis = None
                prescription = None
                audio_filepath = None
                language_code = LANGUAGE_CODES.get(response_language, "en")
                
                if text_input and not image_base64:
                    st.info("🧠 Analyzing text input...")
                    diagnosis = analyze_text_query(text_input, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                elif image_base64:
                    st.info("🧠 Analyzing image...")
                    diagnosis = analyze_image_with_query(text_input or "Analyze this skin condition", image_base64, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                
                # Audio diagnosis and prescription
                audio_bytes = None
                if diagnosis and prescription:
                    full_text_for_audio = f"Diagnosis: {diagnosis}. Prescription: {prescription}"
                    audio_bytes = generate_audio_from_text(full_text_for_audio, language_code)
                    if audio_bytes:
                        st.success("🎧 Audio generated successfully")
                
                # Output UI
                st.markdown("---")
                st.markdown("## 📋 Diagnosis Results")
                st.markdown("<div class='section-title'>Your Input Summary</div>", unsafe_allow_html=True)
                st.text_area("Input Summary", value=str(text_input) if text_input else "Image analysis", height=80, disabled=True, label_visibility="collapsed")
                st.markdown("<div class='section-title'>🩺 Detailed Diagnosis</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='diagnosis-card'>{diagnosis or ''}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>💊 Prescription</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prescription-card'>{prescription or ''}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🎧 Audio Diagnosis</div>", unsafe_allow_html=True)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.info("No audio available.")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.error("Please check your GROQ_API_KEY environment variable and try again.")
