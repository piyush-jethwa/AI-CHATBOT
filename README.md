# AI Doctor - Medical Diagnosis System

A comprehensive medical diagnosis system powered by AI that provides medical analysis, diagnosis, and prescription generation in multiple languages (English, Hindi, Marathi). The system is based on medical knowledge from Medical_book1.pdf.

## Features

- **Multi-language Support**: Diagnosis and prescriptions in English, Hindi, and Marathi
- **Voice Input**: Upload audio files for symptom description
- **Image Analysis**: Upload medical images for analysis (skin conditions, etc.)
- **Text-to-Speech**: Audio output of diagnosis and prescriptions
- **Detailed Prescriptions**: Medication instructions with dosage, frequency, and duration

## Deployment on Streamlit

### Prerequisites
- Streamlit account
- GROQ API key (get from https://console.groq.com/)

### Steps to Deploy

1. **Fork/Upload to GitHub**
   - Upload your project to a GitHub repository

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Set the main file path to `ai_doctor_streamlit.py`

3. **Set Environment Variables**
   - In Streamlit deployment settings, add environment variable:
     - `GROQ_API_KEY`: Your GROQ API key (starts with `gsk_`)

4. **Deploy**
   - Click "Deploy" and wait for the app to build

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable:**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   # or on Windows:
   set GROQ_API_KEY="your_groq_api_key_here"
   ```

3. **Run locally:**
   ```bash
   streamlit run ai_doctor_streamlit.py
   ```

## File Structure

```
├── ai_doctor_streamlit.py    # Main Streamlit application
├── brain_of_the_doctor.py    # Core AI logic and diagnosis functions
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .streamlit/
    └── secrets.toml         # Local development secrets (optional)
```

## Usage

1. **Input Methods:**
   - Text: Type symptoms in the text area
   - Voice: Upload audio file (.wav/.mp3) with symptom description
   - Image: Upload medical images for analysis

2. **Language Selection:**
   - Choose from English, Hindi, or Marathi for responses

3. **Get Diagnosis:**
   - Click "Get Diagnosis" to receive AI-powered medical analysis

4. **Prescription:**
   - View detailed medication instructions with dosage information

## Important Notes

- This is an AI-powered tool and should not replace professional medical advice
- Always consult healthcare professionals for serious medical conditions
- The system provides guidance based on symptom descriptions

## API Requirements

- GROQ API key is required for AI functionality
- The app uses the Llama 3 model for medical analysis
