from voice_of_the_doctor import text_to_speech_with_elevenlabs

# Test input
input_text = "This is a test to check ElevenLabs fallback to gTTS."
output_filepath = "test_output.wav"

# Call the function
try:
    result = text_to_speech_with_elevenlabs(input_text, output_filepath)
    print(f"Output saved to: {result}")
except Exception as e:
    print(f"Error: {str(e)}")
