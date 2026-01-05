
import requests
import base64
import json
import soundfile as sf
import numpy as np
import io
import os

# Configuration
API_URL = "http://127.0.0.1:8000/generate"
OUTPUT_FILE = "output_generation.wav"

def create_dummy_audio():
    """Create a 1-second dummy audio file (sine wave) for testing."""
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    # 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    return buffer.getvalue()

def main():
    print("Preparing request...")
    
    # 1. Get Reference Audio
    # You can load from a file like this:
    # with open("path/to/ref.wav", "rb") as f:
    #     audio_bytes = f.read()
    
    # Here we create dummy audio for demonstration
    audio_bytes = create_dummy_audio()
    
    # 2. Encode to Base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # 3. Create Payload
    payload = {
        "text": "Hello, this is a test of the F5 TTS API.",
        "ref_audio": audio_b64,
        "ref_text": "Reference text for the dummy audio.",
        "speed": 1.0,
        "steps": 30
    }
    
    print(f"Sending request to {API_URL}...")
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # 4. Decode Response
        if "audio_base64" in data:
            generated_audio_b64 = data["audio_base64"]
            generated_audio_bytes = base64.b64decode(generated_audio_b64)
            
            with open(OUTPUT_FILE, "wb") as f:
                f.write(generated_audio_bytes)
            
            print(f"Success! Generated audio saved to {OUTPUT_FILE}")
            print(f"Sample rate: {data.get('sample_rate')}")
        else:
            print("Error: No audio in response")
            print(data)
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {API_URL}. Is the server running?")
        print("Run: uvicorn src.maxdiffusion.f5_api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'response' in locals():
            print("Response text:", response.text)

if __name__ == "__main__":
    main()
