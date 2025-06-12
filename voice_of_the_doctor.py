# Load environment variables
from dotenv import load_dotenv
load_dotenv()

#Step1a: Setup Text to Speech–TTS–model with gTTS
import os
from gtts import gTTS
from pydub import AudioSegment

def text_to_speech_with_gtts_old(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)


input_text="Hi this is Ai with Hassan!"
text_to_speech_with_gtts_old(input_text=input_text, output_filepath="gtts_testing.mp3")

#Step1b: Setup Text to Speech–TTS–model with ElevenLabs
import elevenlabs
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY=os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio=client.generate(
        text= input_text,
        voice= "Aria",
        output_format= "mp3_22050_32",
        model= "eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

#text_to_speech_with_elevenlabs_old(input_text, output_filepath="elevenlabs_testing.mp3") 

#Step2: Use Model for Text output to Voice

import subprocess
import platform

def play_human_voice(input_text, output_filepath="final.wav", recursion_depth=0):
    """Play pre-recorded human voice samples"""
    MAX_RECURSION = 3
    
    if recursion_depth >= MAX_RECURSION:
        raise ValueError("Maximum recursion depth reached in voice generation")
        
    # Create cache dir if needed
    os.makedirs("voice_cache", exist_ok=True)
    
    # Check if we have a cached version
    cache_key = f"human_male-{hash(input_text)}"
    cache_file = f"voice_cache/{cache_key}.wav"
    
    if os.path.exists(cache_file):
        return cache_file
    
    # Common phrases mapping to sample files
    common_phrases = {
        "hello": "human_voice_samples/greeting.wav",
        "hi": "human_voice_samples/greeting.wav",
        "how are you": "human_voice_samples/how_are_you.wav",
        "what's wrong": "human_voice_samples/whats_wrong.wav",
        "let me check": "human_voice_samples/let_me_check.wav"
    }
    
    # Check for exact phrase matches first
    lower_text = input_text.lower()
    for phrase, sample_file in common_phrases.items():
        if phrase in lower_text:
            try:
                if os.path.exists(sample_file):
                    # Copy the sample to output path
                    sound = AudioSegment.from_wav(sample_file)
                    sound.export(output_filepath, format="wav")
                    return output_filepath
            except Exception as e:
                print(f"Warning: Could not process voice sample {sample_file}: {str(e)}")
                break  # Skip trying other samples if one fails
    
    # If no matching sample or samples failed, use standard gTTS with warning
    print("Warning: Human voice samples not available - using standard gTTS")
    return text_to_speech_with_gtts(
        input_text,
        output_filepath,
        voice_pack="default"
    )

from functools import lru_cache
import os

@lru_cache(maxsize=100)
def text_to_speech_with_gtts(input_text, output_filepath="final.wav", language="English", voice_pack="default"):
    # First check if human voice requested
    if voice_pack == "human_male":
        return play_human_voice(input_text, output_filepath)
        
    # Check cache first
    cache_key = f"{hash(input_text)}-{language}-{voice_pack}"
    cache_file = f"voice_cache/{cache_key}.wav"
    if os.path.exists(cache_file):
        return cache_file
        
    supported_languages = {
        'English': {'code': 'en', 'voices': ['default', 'uk', 'us', 'au', 'human_male']},
        'Hindi': {'code': 'hi', 'voices': ['default', 'mumbai', 'delhi', 'human_male']},
        'Marathi': {'code': 'mr', 'voices': ['default', 'pune', 'nagpur', 'human_male']}
    }
    
    lang_config = supported_languages.get(language, {'code': 'en', 'voices': ['default']})
    lang_code = lang_config['code']
    
    # Apply voice pack specific adjustments
    if voice_pack == "uk":
        lang_code = "en-uk"
    elif voice_pack == "us":
        lang_code = "en-us"
    elif voice_pack == "au":
        lang_code = "en-au"
    
    # Create cache dir if needed
    os.makedirs("voice_cache", exist_ok=True)
    
    # First save as temporary MP3
    temp_file = "temp.mp3"
    audioobj = gTTS(
        text=input_text,
        lang=lang_code,
        slow=False
    )
    audioobj.save(temp_file)
    
    # Convert to WAV format
    sound = AudioSegment.from_mp3(temp_file)
    sound.export(cache_file, format="wav")
    os.remove(temp_file)
    
    # Copy to requested output path if different
    if cache_file != output_filepath:
        sound.export(output_filepath, format="wav")
    
    return cache_file  # Return cached file path


input_text="Hi this is Ai with Hassan, autoplay testing!"
#text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3")


def text_to_speech_with_elevenlabs(input_text, output_filepath, voice="Aria"):
    if voice == "human_male":
        return play_human_voice(input_text, output_filepath)
    if not ELEVENLABS_API_KEY:
        print("Warning: ElevenLabs API key not configured - falling back to gTTS")
        return text_to_speech_with_gtts(input_text, output_filepath)
        
    try:
        client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        voice_packs = {
            'professional': 'Aria',
            'friendly': 'Ethan',
            'serious': 'Dr. Watson',
            'compassionate': 'Charlotte'
        }
        
        selected_voice = voice_packs.get(voice, 'Aria')
        
        audio=client.generate(
            text=input_text,
            voice=selected_voice,
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        elevenlabs.save(audio, output_filepath)
        return output_filepath
    except Exception as e:
        raise ValueError(f"ElevenLabs error: {str(e)} - falling back to gTTS")

#text_to_speech_with_elevenlabs(input_text, output_filepath="elevenlabs_testing_autoplay.mp3")