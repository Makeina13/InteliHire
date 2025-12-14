import sys
import subprocess
import os
from dotenv import load_dotenv

print("------------------------------------------------------------")
print(f"🔍 DIAGNOSTIC MODE using Python: {sys.executable}")
print("------------------------------------------------------------")

# 1. CHECK & UPDATE LIBRARY VERSION
try:
    import google.generativeai as genai
    current_version = genai.__version__
    print(f"📉 Current Library Version: {current_version}")
    
    # We need at least version 0.7.0 for Gemini 1.5/2.0
    if current_version < "0.7.0":
        print("⚠️  Library is TOO OLD. Force updating now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'google-generativeai'])
        print("✅ Update Complete. Please restart this script to test again.")
        sys.exit() # Stop here so user restarts
    else:
        print("✅ Library Version is OK.")
        
except ImportError:
    print("❌ Library missing. Installing now...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'google-generativeai'])
    sys.exit()

# 2. TEST API KEY & MODELS
load_dotenv()
key = os.getenv('GEMINI_API_KEY')

if not key:
    print("❌ ERROR: No API Key found in .env file.")
else:
    print(f"🔑 API Key Found: {key[:5]}...")

genai.configure(api_key=key)

print("\n📡 Testing Connection to Google...")
try:
    # Try the most standard model first
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello")
    print(f"✅ SUCCESS! Gemini 1.5 Flash replied: {response.text}")
    print("------------------------------------------------------------")
    print("🚀 YOU ARE READY. The issue was the library version.")
    print("   Run 'python app.py' now.")
    print("------------------------------------------------------------")
    
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("\nAttempting to list available models for your key...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   - Available: {m.name}")
    except Exception as list_e:
        print(f"   Could not list models: {list_e}")