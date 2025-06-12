import os
import sys
import base64
import time
import hashlib
import shutil
import tempfile
from functools import lru_cache
from groq import Groq, GroqError

try:
    from dotenv import load_dotenv
    import streamlit as st
    # Load variables from .env (only used in local development)
    load_dotenv()
    
    # First try to get from .env, then from Streamlit secrets
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}. Environment variables must be set manually.")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def test_api_key(api_key):
    """Test if the provided API key is valid by making a minimal request"""
    try:
        client = Groq(api_key=api_key)
        # Make a minimal request to list available models or similar
        models = client.models.list()
        if models:
            return True
        return False
    except Exception as e:
        print(f"API key test failed: {str(e)}")
        return False

def handle_long_path(file_path):
    """Handle long file paths by creating a shorter temporary path"""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Get the file extension
        _, ext = os.path.splitext(file_path)
        # Create a new shorter path
        new_path = os.path.join(temp_dir, f"temp{ext}")
        # Copy the file to the new location
        shutil.copy2(file_path, new_path)
        return new_path
    except Exception as e:
        print(f"Error handling long path: {str(e)}")
        return file_path

def encode_image(image_path, max_size=256):
    """Convert image to base64 string with optional resizing"""
    try:
        # Handle long paths
        image_path = handle_long_path(image_path)
        
        import cv2
        # Read and optionally resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            
        # Encode with lower quality
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
        
    except Exception:
        # Fallback to original method if OpenCV fails
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

PRESCRIPTION_TEMPLATE = """
PRESCRIPTION
Date: {date}
Patient: {patient_name}
Diagnosis: {diagnosis}

Medications:
{medications}

Instructions:
{instructions}

Doctor: AI Doctor
"""

def generate_prescription(diagnosis, language="English"):
    """Generate a prescription based on diagnosis"""
    from datetime import datetime
    
    # Validate diagnosis parameter
    if not diagnosis or not isinstance(diagnosis, str):
        raise ValueError("Diagnosis must be a non-empty string")
    
    # Expanded medication mapping with detailed instructions in multiple languages
    meds_map = {
        "Dandruff": {
            "English": {
                "medications": [
                    "Ketoconazole 2% shampoo",
                    "Selenium sulfide 2.5% shampoo",
                    "Zinc pyrithione 1% shampoo"
                ],
                "instructions": [
                    "Use medicated shampoo twice weekly",
                    "Leave shampoo on scalp for 5-10 minutes before rinsing",
                    "Avoid scratching the scalp",
                    "Use gentle, fragrance-free hair products"
                ],
                "follow_up": "Follow up in 2 weeks if condition persists"
            },
            "Hindi": {
                "medications": [
                    "कीटोकोनाज़ोल 2% शैम्पू",
                    "सेलेनियम सल्फाइड 2.5% शैम्पू",
                    "जिंक पायरिथियोन 1% शैम्पू"
                ],
                "instructions": [
                    "सप्ताह में दो बार मेडिकेटेड शैम्पू का उपयोग करें",
                    "रिंस करने से पहले 5-10 मिनट तक शैम्पू को स्कैल्प पर छोड़ दें",
                    "स्कैल्प को खरोंचने से बचें",
                    "हल्के, सुगंध-मुक्त हेयर प्रोडक्ट्स का उपयोग करें"
                ],
                "follow_up": "यदि स्थिति बनी रहती है तो 2 सप्ताह में फॉलो-अप करें"
            },
            "Marathi": {
                "medications": [
                    "कीटोकोनाज़ोल 2% शॅम्पू",
                    "सेलेनियम सल्फाइड 2.5% शॅम्पू",
                    "जिंक पायरिथियोन 1% शॅम्पू"
                ],
                "instructions": [
                    "आठवड्यातून दोनदा औषधी शॅम्पू वापरा",
                    "धुण्याआधी 5-10 मिनिटे शॅम्पू डोक्यावर ठेवा",
                    "डोक्यावर खाजवू नका",
                    "हलके, सुगंध-मुक्त केसांचे उत्पादने वापरा"
                ],
                "follow_up": "जर स्थिती टिकून राहिली तर 2 आठवड्यांनी पुन्हा तपासणी करा"
            }
        }
    }
    
    # Get current date
    date = datetime.now().strftime("%d/%m/%Y")
    
    # Get appropriate medication based on language
    treatment = meds_map.get(diagnosis, {}).get(language, {
        "medications": ["Consult doctor for proper medication"],
        "instructions": ["Follow doctor's advice"],
        "follow_up": "Follow up as recommended by doctor"
    })
    
    # Language-specific prescription templates
    templates = {
        "English": """
PRESCRIPTION
Date: {date}
Patient: [Patient Name]
Diagnosis: {diagnosis}

Medications:
{medications}

Instructions:
{instructions}

Follow-up: {follow_up}

Doctor: AI Doctor
""",
        "Hindi": """
नुस्खा
दिनांक: {date}
रोगी: [रोगी का नाम]
निदान: {diagnosis}

दवाइयां:
{medications}

निर्देश:
{instructions}

फॉलो-अप: {follow_up}

डॉक्टर: AI डॉक्टर
""",
        "Marathi": """
औषधोपचार
दिनांक: {date}
रुग्ण: [रुग्णाचे नाव]
निदान: {diagnosis}

औषधे:
{medications}

सूचना:
{instructions}

पुन्हा तपासणी: {follow_up}

डॉक्टर: AI डॉक्टर
"""
    }
    
    template = templates.get(language, templates["English"])
    
    return template.format(
        date=date,
        diagnosis=diagnosis,
        medications="\n".join(f"- {med}" for med in treatment["medications"]),
        instructions="\n".join(f"- {inst}" for inst in treatment["instructions"]),
        follow_up=treatment["follow_up"]
    )

@lru_cache(maxsize=100)
def analyze_image_with_query(query, encoded_image, language="English", model="llama3-8b-8192"):
    """Analyze image with query using Groq"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Language-specific prompts
        prompts = {
            "English": "You are a medical expert. Analyze this medical image and provide a detailed diagnosis: ",
            "Hindi": "आप एक चिकित्सा विशेषज्ञ हैं। इस चिकित्सा छवि का विश्लेषण करें और विस्तृत निदान प्रदान करें: ",
            "Marathi": "तुम्ही एक वैद्यकीय तज्ज्ञ आहात. या वैद्यकीय प्रतिमेचे विश्लेषण करा आणि तपशीलवार निदान द्या: "
        }
        
        # Get the appropriate prompt for the language
        prompt = prompts.get(language, prompts["English"])
        
        # Prepare the message with image
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and return the response
        diagnosis = response.choices[0].message.content.strip()
        return diagnosis
        
    except Exception as e:
        print(f"Error in analyze_image_with_query: {str(e)}")
        return f"Vision analysis failed: {str(e)}"

def analyze_image(image_path):
    """Analyze image using computer vision"""
    try:
        from image_analysis import analyze_image_colors
        analysis = analyze_image_colors(image_path)
        return f"Image analysis results: Dominant colors are {', '.join(analysis['dominant_colors'])}"
    except Exception as e:
        raise ValueError(f"Image analysis failed: {str(e)}")

@lru_cache(maxsize=100)
def analyze_text_query(query, language="English", model="llama3-8b-8192", max_retries=3):
    """Analyze text query using Groq"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Language-specific prompts
        prompts = {
            "English": "You are a medical expert. Analyze the following symptoms and provide a detailed diagnosis: ",
            "Hindi": "आप एक चिकित्सा विशेषज्ञ हैं। निम्नलिखित लक्षणों का विश्लेषण करें और विस्तृत निदान प्रदान करें: ",
            "Marathi": "तुम्ही एक वैद्यकीय तज्ज्ञ आहात. खालील लक्षणांचे विश्लेषण करा आणि तपशीलवार निदान द्या: "
        }
        
        # Get the appropriate prompt for the language
        prompt = prompts.get(language, prompts["English"])
        
        # Prepare the message
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and return the response
        diagnosis = response.choices[0].message.content.strip()
        return diagnosis
        
    except Exception as e:
        print(f"Error in analyze_text_query: {str(e)}")
        return f"Analysis failed: {str(e)}"

if __name__ == "__main__":
    os.system("python D:\\EDIT KAREGE\\ai-doctor-2.0-voice-and-vision\\ai-doctor-2.0-voice-and-vision\\ai_doctor_fully_fixed.py")
