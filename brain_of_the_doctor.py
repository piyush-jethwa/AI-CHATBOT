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
        print("Available models:", [model.id for model in models.data])
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

import random

def generate_prescription(diagnosis, language="English"):
    """Generate a prescription based on diagnosis using AI to suggest medications."""
    from datetime import datetime

    if not diagnosis or not isinstance(diagnosis, str):
        raise ValueError("Diagnosis must be a non-empty string")

    date = datetime.now().strftime("%d/%m/%Y")
    
    # Use AI to generate appropriate medications based on the diagnosis
    client = Groq(api_key=get_api_key())
    
    # Language-specific prompts for medication generation with detailed instructions
    medication_prompts = {
        "English": """Based on the following medical diagnosis, provide 2-3 appropriate medications or treatments with specific instructions for each.
        For each medication, include: medication name, dosage, frequency, duration, and any special instructions.
        Return in this format:
        - Medication Name: Dosage instructions (e.g., 500mg tablet), Frequency (e.g., twice daily), Duration (e.g., for 7 days), Special instructions
        
        Diagnosis: {diagnosis}
        
        Medications with Instructions:""",
        
        "Hindi": """निम्नलिखित चिकित्सा निदान के आधार पर, प्रत्येक के लिए विशिष्ट निर्देशों के साथ 2-3 उपयुक्त दवाएं या उपचार प्रदान करें।
        प्रत्येक दवा के लिए शामिल करें: दवा का नाम, खुराक, आवृत्ति, अवधि और कोई विशेष निर्देश।
        इस प्रारूप में लौटाएं:
        - दवा का नाम: खुराक निर्देश (उदा., 500mg गोली), आवृत्ति (उदा., दिन में दो बार), अवधि (उदा., 7 दिनों के लिए), विशेष निर्देश
        
        निदान: {diagnosis}
        
        निर्देशों के साथ दवाएं:""",
        
        "Marathi": """खालील वैद्यकीय निदानावर आधारित, प्रत्येकासाठी विशिष्ट सूचनांसह २-३ योग्य औषधे किंवा उपचार द्या.
        प्रत्येक औषधासाठी समाविष्ट करा: औषधाचे नाव, खुराक, वारंवारता, कालावधी आणि कोणतीही विशेष सूचना.
        या स्वरूपात परत करा:
        - औषधाचे नाव: खुराक सूचना (उदा., 500mg गोळी), वारंवारता (उदा., दिवसातून दोन वेळा), कालावधी (उदा., ७ दिवसांसाठी), विशेष सूचना
        
        निदान: {diagnosis}
        
        सूचनांसह औषधे:"""
    }
    
    prompt_template = medication_prompts.get(language, medication_prompts["English"])
    prompt = prompt_template.format(diagnosis=diagnosis[:500])  # Limit diagnosis length
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical professional providing medication recommendations."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.3
        )
        
        medications_text = response.choices[0].message.content.strip()
        
        # Parse the medications from the response
        medications = []
        lines = medications_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that are just headers
            if not line or any(phrase in line.lower() for phrase in ["here is", "medications:", "list of", "following"]):
                continue
            
            # Handle different list formats
            if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                # Clean up list items
                clean_med = line.replace('-', '').replace('•', '').replace('*', '').strip()
                # Remove numbering
                if clean_med and clean_med[0].isdigit() and '.' in clean_med:
                    clean_med = clean_med.split('.', 1)[1].strip()
                if clean_med and len(clean_med) > 3:
                    medications.append(clean_med)
            else:
                # Handle plain text medication names
                if len(line) > 3 and not any(word in line.lower() for word in ["medication", "treatment", "prescription"]):
                    medications.append(line)
        
        # If parsing failed, use fallback medications
        if not medications:
            fallback_meds = {
                "English": ["Consult healthcare professional for specific medication"],
                "Hindi": ["विशिष्ट दवा के लिए स्वास्थ्य देखभाल पेशेवर से परामर्श करें"],
                "Marathi": ["विशिष्ट औषधासाठी आरोग्यसेवा व्यावसायिकांचा सल्ला घ्या"]
            }
            medications = fallback_meds.get(language, fallback_meds["English"])
            
    except Exception as e:
        print(f"Medication generation failed: {str(e)}")
        fallback_meds = {
            "English": ["Consult healthcare professional for medication"],
            "Hindi": ["दवा के लिए स्वास्थ्य देखभाल पेशेवर से परामर्श करें"],
            "Marathi": ["औषधासाठी आरोग्यसेवा व्यावसायिकांचा सल्ला घ्या"]
        }
        medications = fallback_meds.get(language, fallback_meds["English"])

    templates = {
        "English": """
PRESCRIPTION
Date: {date}
Patient: [Patient Name]
Diagnosis: {diagnosis}

Medications:
{medications}

Doctor: AI Doctor
""",
        "Hindi": """
नुस्खा
दिनांक: {date}
रोगी: [रोगी का नाम]
निदान: {diagnosis}

दवाइयां:
{medications}

डॉक्टर: AI Doctor
""",
        "Marathi": """
औषधोपचार
दिनांक: {date}
रुग्ण: [रुग्णाचे नाव]
निदान: {diagnosis}

औषधे:
{medications}

डॉक्टर: AI Doctor
"""
    }

    template = templates.get(language, templates["English"])

    return template.format(
        date=date,
        diagnosis=diagnosis[:80] + "..." if len(diagnosis) > 80 else diagnosis,  # Show first 80 chars of diagnosis
        medications="\n".join(f"- {med}" for med in medications),
    )

@lru_cache(maxsize=100)
def analyze_image_with_query(query, encoded_image, language="English", model="llama3.1-8b-instant"):
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
        - फॉलो-अप समय
        
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
    
    # Add explicit language instruction to the system prompt
    language_instructions = {
        "English": "Respond in English only.",
        "Hindi": "केवल हिंदी में उत्तर दें।",
        "Marathi": "केवळ मराठीत उत्तर द्या।"
    }
    
    system_prompt = f"{system_prompt} {language_instructions.get(language, 'Respond in English only.')}"
    
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
def analyze_text_query(query, language="English", model="llama-3.1-8b-instant", max_retries=3):
    """Process text queries with GROQ API with caching and focused diagnosis"""
    import logging
    if not query or not isinstance(query, str):
        logging.error("Invalid query parameter for analyze_text_query")
        return "Error: Invalid query parameter."
        
    client = Groq(api_key=get_api_key())
    
    # Language-specific prompts with varied response patterns - focused on concise diagnosis only
    language_prompts = {
        "English": [
            "You are a medical specialist. Provide a concise diagnosis for these symptoms. Focus on the most likely condition and key symptoms. Keep it brief:",
            "As a healthcare professional, give a quick medical assessment of these symptoms. Be concise and focus on the primary diagnosis:",
            "Provide a brief medical diagnosis of these symptoms. Keep it short and focused on the main condition:"
        ],
        "Hindi": [
            "आप एक चिकित्सा विशेषज्ञ हैं। इन लक्षणों का संक्षिप्त निदान प्रदान करें। मुख्य स्थिति और प्रमुख लक्षणों पर ध्यान दें। संक्षिप्त रहें:",
            "एक स्वास्थ्य देखभाल पेशेवर के रूप में, इन लक्षणों का संक्षिप्त चिकित्सा आकलन दें। संक्षिप्त रहें और प्राथमिक निदान पर ध्यान दें:",
            "इन लक्षणों का संक्षिप्त चिकित्सा निदान प्रदान करें। संक्षिप्त रहें और मुख्य स्थिति पर ध्यान केंद्रित करें:"
        ],
        "Marathi": [
            "तुम्ही एक वैद्यकीय तज्ज्ञ आहात. या लक्षणांचे संक्षिप्त निदान द्या. मुख्य स्थिती आणि प्रमुख लक्षणांवर लक्ष केंद्रित करा. संक्षिप्त रहा:",
            "आरोग्यसेवा व्यावसायिक म्हणून, या लक्षणांचे संक्षिप्त वैद्यकीय मूल्यांकन द्या. संक्षिप्त रहा आणि प्राथमिक निदानावर लक्ष केंद्रित करा:",
            "या लक्षणांचे संक्षिप्त वैद्यकीय निदान द्या. संक्षिप्त रहा आणि मुख्य स्थितीवर लक्ष केंद्रित करा:"
        ]
    }
    
    # Get random prompt for the selected language to add variability
    prompts = language_prompts.get(language, language_prompts["English"])
    system_prompt = random.choice(prompts) if isinstance(prompts, list) else prompts
    
    # Add explicit language instruction to the system prompt
    language_instructions = {
        "English": "Respond in English only.",
        "Hindi": "केवल हिंदी में उत्तर दें।",
        "Marathi": "केवळ मराठीत उत्तर द्या।"
    }
    
    system_prompt_with_language = f"{system_prompt} {language_instructions.get(language, 'Respond in English only.')}"
    
    # Add some variability to the query to get different responses
    query_variations = [
        query,
        f"Patient reports: {query}. Please provide medical analysis.",
        f"Symptoms described: {query}. Need professional diagnosis.",
        f"Medical consultation request: {query}"
    ]
    
    user_query = random.choice(query_variations)
    
    messages = [
        {"role": "system", "content": system_prompt_with_language},
        {"role": "user", "content": user_query}
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=800,
                temperature=0.7  # Add some randomness to responses
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
            
            # Add some post-processing to ensure varied responses
            diagnosis_variations = [
                content,
                f"MEDICAL ANALYSIS:\n{content}",
                f"DIAGNOSTIC ASSESSMENT:\n{content}",
                f"CLINICAL EVALUATION:\n{content}"
            ]
            
            return random.choice(diagnosis_variations)
            
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
