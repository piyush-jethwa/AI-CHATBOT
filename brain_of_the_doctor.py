from dotenv import load_dotenv
load_dotenv()

import os
import sys
import base64
import time
import hashlib
import shutil
import tempfile
from functools import lru_cache
from groq import Groq, GroqError

# Try to get API key from environment variables first
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_api_key():
    """Get API key from Streamlit secrets or environment variables"""
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except (ImportError, KeyError, AttributeError):
        return os.environ.get("GROQ_API_KEY")

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
    """Generate a prescription based on diagnosis."""
    from datetime import datetime

    if not diagnosis or not isinstance(diagnosis, str):
        raise ValueError("Diagnosis must be a non-empty string")

    # Clean up the diagnosis string to get the primary condition
    # Example: "DIAGNOSIS:\n- Condition identified: Dandruff" -> "Dandruff"
    primary_diagnosis = diagnosis.splitlines()
    for line in primary_diagnosis:
        if "condition identified" in line.lower():
            primary_diagnosis = line.split(":")[-1].strip()
            break
    else:
        # If the specific line isn't found, use the first non-empty line
        primary_diagnosis = next((line for line in diagnosis.splitlines() if line.strip()), diagnosis)


    meds_map = {
        "Dandruff": {
            "English": {
                "medications": ["Ketoconazole 2% shampoo", "Selenium sulfide 2.5% shampoo"],
                "instructions": ["Use medicated shampoo twice weekly.", "Leave on scalp for 5-10 minutes."],
                "follow_up": "Follow up in 2 weeks if condition persists.",
            },
            "Hindi": {
                "medications": ["कीटोकोनाज़ोल 2% शैम्पू", "सेलेनियम सल्फाइड 2.5% शैम्पू"],
                "instructions": ["सप्ताह में दो बार मेडिकेटेड शैम्पू का उपयोग करें।", "5-10 मिनट तक स्कैल्प पर लगा रहने दें।"],
                "follow_up": "यदि स्थिति बनी रहती है तो 2 सप्ताह में फॉलो-अप करें।",
            },
            "Marathi": {
                "medications": ["कीटोकोनाझोल २% शाम्पू", "सेलेनियम सल्फाइड २.५% शाम्पू"],
                "instructions": ["आठवड्यातून दोनदा औषधी शाम्पू वापरावा.", "५-१० मिनिटे स्कॅल्पवर ठेवा."],
                "follow_up": "स्थिती कायम राहिल्यास २ आठवड्यांनी पुन्हा संपर्क साधा.",
            },
        }
    }
    
    # Multilingual fallback messages
    fallback_treatment = {
        "English": {
            "medications": ["No specific medication found for this diagnosis."],
            "instructions": ["Please consult a healthcare professional for a personalized treatment plan."],
            "follow_up": "Follow up with a doctor as soon as possible.",
        },
        "Hindi": {
            "medications": ["इस निदान के लिए कोई विशिष्ट दवा नहीं मिली।"],
            "instructions": ["व्यक्तिगत उपचार योजना के लिए कृपया किसी स्वास्थ्य देखभाल पेशेवर से परामर्श लें।"],
            "follow_up": "यथाशीघ्र डॉक्टर से संपर्क करें।",
        },
        "Marathi": {
            "medications": ["या निदानासाठी कोणतेही विशिष्ट औषध सापडले नाही."],
            "instructions": ["वैयक्तिकृत उपचार योजनेसाठी कृपया आरोग्यसेवा व्यावसायिकाचा सल्ला घ्या."],
            "follow_up": "शक्य तितक्या लवकर डॉक्टरांशी संपर्क साधा.",
        },
    }

    date = datetime.now().strftime("%d/%m/%Y")
    
    # Find treatment for the primary diagnosis, or use the multilingual fallback
    treatment_options = meds_map.get(primary_diagnosis, fallback_treatment)
    treatment = treatment_options.get(language, treatment_options.get("English"))


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

डॉक्टर: AI Doctor
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

डॉक्टर: AI Doctor
"""
    }

    template = templates.get(language, templates["English"])

    return template.format(
        date=date,
        diagnosis=diagnosis, # Keep the full original diagnosis for display
        medications="\n".join(f"- {med}" for med in treatment["medications"]),
        instructions="\n".join(f"- {inst}" for inst in treatment["instructions"]),
        follow_up=treatment["follow_up"],
    )

@lru_cache(maxsize=100)
def analyze_image_with_query(query, encoded_image, language="English", model="llama3-8b-8192"):
    """Analyze image with text query using GROQ's vision model with caching"""
    import logging
    if not query or not encoded_image:
        logging.error("Missing required parameters for analyze_image_with_query")
        return "Error: Missing required parameters for image analysis."
        
    client = Groq(api_key=get_api_key())
    
    # Since llama3-8b-8192 doesn't support vision, we'll analyze the text query
    # and provide guidance based on the image context
    logging.info("Vision model not available, falling back to text analysis with image context")
    
    # Language-specific prompts for image-based analysis
    language_prompts = {
        "English": """You are a dermatology specialist AI assistant. A patient has uploaded an image of their skin condition and provided the following description. 
        Please analyze their symptoms and provide a comprehensive diagnosis.
        
        For skin conditions like dandruff, look for these symptoms in their description:
        1. White or yellowish flakes on the scalp
        2. Itchy scalp
        3. Dry or oily scalp
        4. Redness or inflammation
        5. Any visible skin changes or rashes
        
        Provide your analysis in this format:
        
        DIAGNOSIS:
        - Condition identified (based on described symptoms)
        - Severity level (Mild/Moderate/Severe)
        - Key symptoms mentioned
        
        RECOMMENDATIONS:
        - Immediate care steps
        - Lifestyle changes
        - Products to use/avoid
        
        PRESCRIPTION:
        - Specific medications or treatments
        - Application instructions
        - Follow-up timeline
        
        Note: This analysis is based on the patient's description. For more accurate diagnosis, please consult a healthcare professional.""",
        
        "Hindi": """आप एक त्वचा विशेषज्ञ AI सहायक हैं। एक रोगी ने अपनी त्वचा की स्थिति की तस्वीर अपलोड की है और निम्नलिखित विवरण प्रदान किया है।
        कृपया उनके लक्षणों का विश्लेषण करें और एक व्यापक निदान प्रदान करें।
        
        रूसी जैसी त्वचा की स्थितियों के लिए, उनके विवरण में इन लक्षणों को देखें:
        1. स्कैल्प पर सफेद या पीले रंग के फ्लेक्स
        2. खुजली वाला स्कैल्प
        3. सूखा या तैलीय स्कैल्प
        4. लालिमा या सूजन
        5. कोई दृश्य त्वचा परिवर्तन या चकत्ते
        
        अपना विश्लेषण इस प्रारूप में प्रदान करें:
        
        निदान:
        - पहचानी गई स्थिति (वर्णित लक्षणों के आधार पर)
        - गंभीरता स्तर (हल्का/मध्यम/गंभीर)
        - मुख्य लक्षण
        
        सिफारिशें:
        - तत्काल देखभाल के कदम
        - जीवनशैली में परिवर्तन
        - उपयोग करने/बचने के उत्पाद
        
        नुस्खा:
        - विशिष्ट दवाएं या उपचार
        - अनुप्रयोग निर्देश
        - फॉलो-अप समयरेखा
        
        नोट: यह विश्लेषण रोगी के विवरण के आधार पर है। अधिक सटीक निदान के लिए, कृपया एक स्वास्थ्य देखभाल पेशेवर से परामर्श करें।""",
        
        "Marathi": """तुम्ही एक त्वचारोग तज्ज्ञ AI सहाय्यक आहात. एक रुग्णाने त्यांच्या त्वचेच्या स्थितीचे चित्र अपलोड केले आहे आणि खालील वर्णन प्रदान केले आहे.
        कृपया त्यांच्या लक्षणांचे विश्लेषण करा आणि एक व्यापक निदान द्या.
        
        कोंड्यासारख्या त्वचेच्या स्थितींसाठी, त्यांच्या वर्णनात या लक्षणे शोधा:
        1. डोक्यावर पांढरे किंवा पिवळे फ्लेक्स
        2. खाज सुटणारे डोके
        3. कोरडे किंवा तैलयुक्त डोके
        4. लालसरपणा किंवा सूज
        5. कोणतेही दृश्य त्वचा बदल किंवा पुरळ
        
        तुमचे विश्लेषण या स्वरूपात द्या:
        
        निदान:
        - ओळखलेली स्थिती (वर्णन केलेल्या लक्षणांच्या आधारे)
        - गंभीरता पातळी (हलकी/मध्यम/गंभीर)
        - मुख्य लक्षणे
        
        शिफारसी:
        - त्वरित काळजीचे पावले
        - जीवनशैली बदल
        - वापरण्यासाठी/टाळण्यासाठी उत्पादने
        
        औषधोपचार:
        - विशिष्ट औषधे किंवा उपचार
        - वापरण्याच्या सूचना
        - पुन्हा तपासणी वेळ
        
        टीप: हे विश्लेषण रुग्णाच्या वर्णनावर आधारित आहे. अधिक अचूक निदानासाठी, कृपया वैद्यकीय व्यावसायिकांशी सल्लामसलत करा."""
    }
    
    # Get the appropriate prompt for the selected language
    system_prompt = language_prompts.get(language, language_prompts["English"])
    
    # Create a comprehensive query that includes image context
    enhanced_query = f"""Patient has uploaded an image of their skin condition and reports: {query}
    
    Please provide a detailed medical analysis based on their description. Consider common skin conditions that match their symptoms.
    
    Focus on providing helpful medical guidance while noting that this is based on their description and not a direct visual analysis."""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": enhanced_query
        }
    ]
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=800
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            logging.error("Empty response content from analyze_image_with_query")
            return "Error: Empty response from image analysis."
        
        # Add a note about the analysis method
        note = {
            "English": "\n\nNote: This analysis is based on your description. For more accurate diagnosis, please consult a healthcare professional.",
            "Hindi": "\n\nनोट: यह विश्लेषण आपके विवरण के आधार पर है। अधिक सटीक निदान के लिए, कृपया एक स्वास्थ्य देखभाल पेशेवर से परामर्श करें।",
            "Marathi": "\n\nटीप: हे विश्लेषण तुमच्या वर्णनावर आधारित आहे. अधिक अचूक निदानासाठी, कृपया वैद्यकीय व्यावसायिकांशी सल्लामसलत करा."
        }
        
        return content + note.get(language, note["English"])
        
    except Exception as e:
        logging.error(f"Vision analysis failed: {str(e)}")
        if "model_not_found" in str(e):
            return analyze_text_query(query, language)
        return f"Vision analysis failed: {str(e)}"

# Validate GROQ API key
if not GROQ_API_KEY:
    error_msg = """
    ERROR: GROQ_API_KEY not found. Please make sure it's set in:
    1. Streamlit secrets (for deployment) - st.secrets["GROQ_API_KEY"]
    2. Environment variables (for local development) - GROQ_API_KEY
    3. .env file (for local development) - GROQ_API_KEY=your_key_here
    
    You can get an API key from: https://console.groq.com/
    """
    print(error_msg)
    # Don't exit in Streamlit environment, just show error
    if 'streamlit' not in sys.modules:
        sys.exit(1)
else:
    print(f"🔑 API Key Status: Found (Length: {len(GROQ_API_KEY)} characters)")
    if GROQ_API_KEY.startswith("gsk_"):
        print("✅ API Key format looks correct (starts with 'gsk_')")
        # Test the API key
        if test_api_key(GROQ_API_KEY):
            print("✅ API Key is valid and working!")
        else:
            print("❌ API Key test failed - key may be invalid or expired")
    else:
        print("⚠️ API Key format may be incorrect (should start with 'gsk_')")

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
    """Process text queries with GROQ API with caching"""
    import logging
    if not query or not isinstance(query, str):
        logging.error("Invalid query parameter for analyze_text_query")
        return "Error: Invalid query parameter."
        
    client = Groq(api_key=get_api_key())
    
    # Language-specific prompts
    language_prompts = {
        "English": "You are a medical specialist. Analyze the following symptoms and provide a diagnosis in English:",
        "Hindi": "आप एक चिकित्सा विशेषज्ञ हैं। कृपया उत्तर हिंदी में दें। निम्नलिखित लक्षणों का विश्लेषण करें और हिंदी में निदान प्रदान करें:",
        "Marathi": "तुम्ही एक वैद्यकीय तज्ज्ञ आहात. कृपया उत्तर मराठीत द्या. खालील लक्षणांचे विश्लेषण करा आणि मराठीमध्ये निदान द्या:"
    }
    
    # Get the appropriate prompt for the selected language
    system_prompt = language_prompts.get(language, language_prompts["English"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=800
            )
            
            if not response.choices:
                logging.error("Empty response from API in analyze_text_query")
                return "Error: Empty response from text analysis."
                
            content = response.choices[0].message.content
            print("MODEL RAW OUTPUT:", repr(content))
            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                logging.error("Empty content string from analyze_text_query")
                return "Error: Empty content from text analysis."
            return content
            
        except GroqError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            logging.error(f"API request failed after {max_retries} attempts: {str(e)}")
            return f"Text analysis failed: {str(e)}"
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return f"Text analysis failed: {str(e)}"

if __name__ == "__main__":
    os.system("python D:\\EDIT KAREGE\\ai-doctor-2.0-voice-and-vision\\ai-doctor-2.0-voice-and-vision\\ai_doctor_fully_fixed.py")
