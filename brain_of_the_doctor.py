import os
import base64
import tempfile
import shutil
from functools import lru_cache
from groq import Groq, GroqError

# --- Load GROQ_API_KEY ---
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in Streamlit secrets or .env")

# --- Utility: Test API key ---
def test_api_key(api_key):
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        return bool(models)
    except Exception as e:
        print(f"API key test failed: {e}")
        return False

# --- Utility: Image encoding ---
def handle_long_path(file_path):
    try:
        temp_dir = tempfile.mkdtemp()
        _, ext = os.path.splitext(file_path)
        new_path = os.path.join(temp_dir, f"temp{ext}")
        shutil.copy2(file_path, new_path)
        return new_path
    except Exception as e:
        print(f"Path error: {e}")
        return file_path

def encode_image(image_path, max_size=256):
    try:
        image_path = handle_long_path(image_path)
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image")
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

# --- Generate Prescription ---
def generate_prescription(diagnosis, language="English"):
    from datetime import datetime
    date = datetime.now().strftime("%d/%m/%Y")
    meds_map = {
        "Dandruff": {
            "English": {
                "medications": [
                    "Ketoconazole 2% shampoo",
                    "Selenium sulfide 2.5%",
                    "Zinc pyrithione 1%"
                ],
                "instructions": [
                    "Use shampoo twice weekly",
                    "Leave for 5-10 mins before rinsing",
                    "Avoid scratching",
                    "Use mild products"
                ],
                "follow_up": "Follow up in 2 weeks if needed"
            },
            "Hindi": {
                "medications": ["कीटोकोनाज़ोल 2%", "सेलेनियम सल्फाइड 2.5%", "जिंक पायरिथियोन 1%"],
                "instructions": [
                    "हफ्ते में दो बार उपयोग करें", "5-10 मिनट स्कैल्प पर छोड़ें",
                    "खरोंच से बचें", "सुगंध-मुक्त शैम्पू का उपयोग करें"
                ],
                "follow_up": "2 सप्ताह में पुनः जांच करें"
            },
            "Marathi": {
                "medications": ["कीटोकोनाज़ोल 2%", "सेलेनियम सल्फाइड 2.5%", "जिंक पायरिथियोन 1%"],
                "instructions": [
                    "आठवड्यातून दोनदा वापरा", "5-10 मिनिट ठेवून धुवा",
                    "खाज टाळा", "मृदू उत्पादन वापरा"
                ],
                "follow_up": "2 आठवड्यांनी पुनरावलोकन"
            }
        }
    }
    template = {
        "English": """
PRESCRIPTION
Date: {date}
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
निदान: {diagnosis}

औषधे:
{medications}

सूचना:
{instructions}

पुन्हा तपासणी: {follow_up}

डॉक्टर: AI डॉक्टर
"""
    }.get(language, "English")
    
    treatment = meds_map.get(diagnosis, {}).get(language, {
        "medications": ["Consult doctor"],
        "instructions": ["Follow doctor’s advice"],
        "follow_up": "As needed"
    })
    
    return template.format(
        date=date,
        diagnosis=diagnosis,
        medications="\n".join(f"- {m}" for m in treatment["medications"]),
        instructions="\n".join(f"- {i}" for i in treatment["instructions"]),
        follow_up=treatment["follow_up"]
    )

# --- Groq Text Analysis ---
@lru_cache(maxsize=100)
def analyze_text_query(query, language="English", model="llama3-8b-8192"):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompts = {
            "English": "You are a medical expert. Provide a full diagnosis.",
            "Hindi": "आप एक चिकित्सा विशेषज्ञ हैं। कृपया पूरा निदान प्रदान करें।",
            "Marathi": "तुम्ही वैद्यकीय तज्ज्ञ आहात. कृपया निदान द्या."
        }
        prompt = prompts.get(language, prompts["English"])
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Text analysis failed: {e}")
        return f"Analysis failed: {str(e)}"

# --- Groq Image + Text Analysis ---
@lru_cache(maxsize=100)
def analyze_image_with_query(query, encoded_image, language="English", model="llama3-8b-8192"):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompts = {
            "English": "You are a medical expert. Analyze image and query.",
            "Hindi": "आप एक चिकित्सा विशेषज्ञ हैं। छवि और विवरण का विश्लेषण करें।",
            "Marathi": "तुम्ही वैद्यकीय तज्ज्ञ आहात. कृपया प्रतिमा आणि वर्णनाचे विश्लेषण करा."
        }
        prompt = prompts.get(language, prompts["English"])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Image analysis failed: {e}")
        return f"Vision analysis failed: {str(e)}"
