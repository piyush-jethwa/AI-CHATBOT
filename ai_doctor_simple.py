# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
from gtts import gTTS
import io

st.set_page_config(
    page_title="Vaidya AI - Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)

# Language translations
TRANSLATIONS = {
    "English": {
        "title": "Vaidya AI - Healthcare Assistant",
        "subtitle": "Your AI-powered medical diagnosis companion",
        "voice_recording": "Voice Recording",
        "text_input": "Text Input",
        "image_upload": "Image Upload",
        "language": "Language",
        "symptoms": "Describe your symptoms",
        "days_suffering": "How many days have you been suffering?",
        "submit": "Get Diagnosis",
        "diagnosis": "Diagnosis",
        "prescription": "Prescription",
        "recommendations": "Recommendations",
        "audio_output": "Audio Output"
    },
    "Hindi": {
        "title": "वैद्य AI - स्वास्थ्य सहायक",
        "subtitle": "आपका AI-संचालित चिकित्सा निदान साथी",
        "voice_recording": "आवाज रिकॉर्डिंग",
        "text_input": "टेक्स्ट इनपुट",
        "image_upload": "छवि अपलोड",
        "language": "भाषा",
        "symptoms": "अपने लक्षणों का वर्णन करें",
        "days_suffering": "आप कितने दिनों से पीड़ित हैं?",
        "submit": "निदान प्राप्त करें",
        "diagnosis": "निदान",
        "prescription": "पर्चे",
        "recommendations": "सिफारिशें",
        "audio_output": "ऑडियो आउटपुट"
    },
    "Marathi": {
        "title": "वैद्य AI - आरोग्य सहायक",
        "subtitle": "तुमचा AI-चालित वैद्यकीय निदान साथी",
        "voice_recording": "आवाज रेकॉर्डिंग",
        "text_input": "मजकूर इनपुट",
        "image_upload": "प्रतिमा अपलोड",
        "language": "भाषा",
        "symptoms": "तुमच्या लक्षणांचे वर्णन करा",
        "days_suffering": "तुम्ही किती दिवसांपासून त्रास होत आहात?",
        "submit": "निदान मिळवा",
        "diagnosis": "निदान",
        "prescription": "औषधपत्र",
        "recommendations": "शिफारसी",
        "audio_output": "ऑडिओ आउटपुट"
    }
}

def tr(key):
    """Get translation for current language"""
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

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

# Initialize session state
if "language" not in st.session_state:
    st.session_state["language"] = "English"

# Sidebar for language selection
with st.sidebar:
    st.header("Settings")
    language = st.selectbox(
        "Language / भाषा / भाषा",
        options=["English", "Hindi", "Marathi"],
        index=["English", "Hindi", "Marathi"].index(st.session_state["language"])
    )
    st.session_state["language"] = language

# Main title
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")

# Check for GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY environment variable")
    st.stop()

# Import the brain of the doctor
try:
    from brain_of_the_doctor import get_diagnosis_and_prescription
except ImportError:
    st.error("brain_of_the_doctor.py not found. Please ensure it's in the same directory.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs([tr("voice_recording"), tr("text_input"), tr("image_upload")])

with tab1:
    st.markdown("### " + tr("voice_recording"))
    
    # Audio file uploader for voice input
    audio_input = st.file_uploader(
        "Upload an audio file (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a"],
        key="audio_upload"
    )
    
    if audio_input is not None:
        st.success("Audio file uploaded successfully!")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_input.name.split(".")[-1]}') as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name
        
        try:
            # Days suffering input for voice tab
            duration_val_voice = st.number_input(
                tr("days_suffering"),
                min_value=0,
                max_value=365,
                value=st.session_state.get("duration_days", 1),
                key="duration_days_voice"
            )
            st.session_state["duration_days"] = duration_val_voice
            
            if st.button(tr("submit"), key="submit_voice"):
                # Get diagnosis
                diagnosis, prescription, recommendations = get_diagnosis_and_prescription(
                    symptoms="Voice input from uploaded audio",
                    duration_days=duration_val_voice,
                    language=language,
                    audio_file=tmp_path
                )
                
                # Display results
                st.markdown("### " + tr("diagnosis"))
                st.write(diagnosis)
                
                st.markdown("### " + tr("prescription"))
                st.write(prescription)
                
                st.markdown("### " + tr("recommendations"))
                st.write(recommendations)
                
                # Generate audio output
                if language == "English":
                    audio_text = f"Diagnosis: {diagnosis}. Prescription: {prescription}. Recommendations: {recommendations}"
                elif language == "Hindi":
                    audio_text = f"निदान: {diagnosis}. पर्चे: {prescription}. सिफारिशें: {recommendations}"
                else:  # Marathi
                    audio_text = f"निदान: {diagnosis}. औषधपत्र: {prescription}. शिफारसी: {recommendations}"
                
                audio_data = generate_audio_from_text(audio_text, "hi" if language == "Hindi" else "mr" if language == "Marathi" else "en")
                if audio_data:
                    st.markdown("### " + tr("audio_output"))
                    st.audio(audio_data, format="audio/mp3")
                
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        st.info("Please upload an audio file to get started with voice diagnosis.")

with tab2:
    st.markdown("### " + tr("text_input"))
    
    symptoms_text = st.text_area(
        tr("symptoms"),
        height=100,
        key="symptoms_text"
    )
    
    duration_val_text = st.number_input(
        tr("days_suffering"),
        min_value=0,
        max_value=365,
        value=st.session_state.get("duration_days", 1),
        key="duration_days_text"
    )
    st.session_state["duration_days"] = duration_val_text
    
    if st.button(tr("submit"), key="submit_text"):
        if symptoms_text.strip():
            try:
                diagnosis, prescription, recommendations = get_diagnosis_and_prescription(
                    symptoms=symptoms_text,
                    duration_days=duration_val_text,
                    language=language
                )
                
                st.markdown("### " + tr("diagnosis"))
                st.write(diagnosis)
                
                st.markdown("### " + tr("prescription"))
                st.write(prescription)
                
                st.markdown("### " + tr("recommendations"))
                st.write(recommendations)
                
                # Generate audio output
                if language == "English":
                    audio_text = f"Diagnosis: {diagnosis}. Prescription: {prescription}. Recommendations: {recommendations}"
                elif language == "Hindi":
                    audio_text = f"निदान: {diagnosis}. पर्चे: {prescription}. सिफारिशें: {recommendations}"
                else:  # Marathi
                    audio_text = f"निदान: {diagnosis}. औषधपत्र: {prescription}. शिफारसी: {recommendations}"
                
                audio_data = generate_audio_from_text(audio_text, "hi" if language == "Hindi" else "mr" if language == "Marathi" else "en")
                if audio_data:
                    st.markdown("### " + tr("audio_output"))
                    st.audio(audio_data, format="audio/mp3")
                
            except Exception as e:
                st.error(f"Error processing diagnosis: {e}")
        else:
            st.warning("Please describe your symptoms first.")

with tab3:
    st.markdown("### " + tr("image_upload"))
    
    uploaded_file = st.file_uploader(
        "Upload a medical image (skin condition, etc.)",
        type=['jpg', 'jpeg', 'png'],
        key="image_upload"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Get diagnosis from image
            diagnosis, prescription, recommendations = get_diagnosis_and_prescription(
                symptoms="Image analysis",
                duration_days=1,
                language=language,
                image_file=tmp_path
            )
            
            st.markdown("### " + tr("diagnosis"))
            st.write(diagnosis)
            
            st.markdown("### " + tr("prescription"))
            st.write(prescription)
            
            st.markdown("### " + tr("recommendations"))
            st.write(recommendations)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

# Footer
st.markdown("---")
st.caption("⚠️ This is for educational purposes only. Always consult a healthcare professional for medical advice.") 