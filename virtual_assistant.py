import os
from openai import OpenAI
from voice_of_the_doctor import text_to_speech_with_gtts

class VirtualAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.assistant_context = """You are a helpful medical virtual assistant. 
        Provide clear, concise answers to general health questions.
        For medical diagnoses, always recommend consulting the AI Doctor.
        Keep responses under 3 sentences."""
        
    def generate_response(self, query, language="English"):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.assistant_context},
                    {"role": "user", "content": query}
                ],
                max_tokens=150
            )
            
            text_response = response.choices[0].message.content
            audio_file = "assistant_response.mp3"
            text_to_speech_with_gtts(text_response, audio_file, language)
            
            return text_response, audio_file
            
        except Exception as e:
            return f"Assistant error: {str(e)}", None
