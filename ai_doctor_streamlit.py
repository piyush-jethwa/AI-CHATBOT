import os
import tempfile
import streamlit as st
from brain_of_the_doctor import (
    encode_image,
    analyze_image_with_query,
    generate_prescription,
    analyze_text_query,
    GROQ_API_KEY
)
from voice_of_the_patient import transcribe_with_groq
from gtts import gTTS
import base64
import io

# Verify API key is available
if not GROQ_API_KEY:
    st.error("""
    Error: GROQ_API_KEY not found. Please make sure it's set in:
    1. Streamlit Secrets (for production)
    2. .env file (for local development)
    """)
    st.stop()

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

st.set_page_config(page_title="VAIDYA - Medical Diagnosis", layout="wide")
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
    st.image("portrait-3d-female-doctor[1].jpg", caption="Your Doctor", use_column_width=True)

# Output section
if submit_btn:
    with st.spinner("Processing..."):
        # Audio input handling
        if audio_input is not None:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_input.name)[-1])
            temp_audio.write(audio_input.read())
            temp_audio.close()
            audio_path = temp_audio.name
            text_input = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=audio_path,
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
            )
            os.remove(audio_path)
        # Image input handling
        image_base64 = None
        if image_input is not None:
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_input.name)[-1])
            temp_image.write(image_input.read())
            temp_image.close()
            image_base64 = encode_image(temp_image.name)
            os.remove(temp_image.name)
        # Diagnosis logic
        diagnosis = None
        prescription = None
        audio_filepath = None
        language_code = LANGUAGE_CODES.get(response_language, "en")
        if text_input and not image_base64:
            diagnosis = analyze_text_query(text_input, response_language)
            prescription = generate_prescription(diagnosis, response_language)
        elif image_base64:
            diagnosis = analyze_image_with_query(text_input or "Analyze this skin condition", image_base64, response_language)
            prescription = generate_prescription(diagnosis, response_language)
        # Audio diagnosis (full content)
        audio_bytes = None
        if diagnosis:
            try:
                tts = gTTS(text=diagnosis, lang=language_code)
                audio_bytes_io = io.BytesIO()
                tts.write_to_fp(audio_bytes_io)
                audio_bytes = audio_bytes_io.getvalue()
            except Exception as e:
                st.warning(f"Audio generation failed: {e}")
                # Fallback to English if language not supported
                if language_code != "en":
                    st.info("Falling back to English audio...")
                    try:
                        tts = gTTS(text=diagnosis, lang="en")
                        audio_bytes_io = io.BytesIO()
                        tts.write_to_fp(audio_bytes_io)
                        audio_bytes = audio_bytes_io.getvalue()
                    except Exception as e:
                        st.error(f"Audio generation failed in English too: {e}")
        # Output UI
        st.markdown("---")
        st.markdown("## 📋 Diagnosis Results")
        st.markdown("<div class='section-title'>Your Input Summary</div>", unsafe_allow_html=True)
        st.text_area("", value=text_input or "Image analysis", height=70, disabled=True)
        st.markdown("<div class='section-title'>🩺 Detailed Diagnosis</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='diagnosis-card'>{diagnosis or ''}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💊 Prescription</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prescription-card'>{prescription or ''}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎧 Audio Diagnosis</div>", unsafe_allow_html=True)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.info("No audio available.") 
