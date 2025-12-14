import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# 1. Force reload the .env file (override cache)
print("--------------------------------------------------")
print("🔍 DEBUGGING ENVIRONMENT...")
dotenv_path = find_dotenv()

if not dotenv_path:
    print("❌ FATAL ERROR: No .env file found!")
    print("   Make sure the file is named exactly '.env' (no .txt)")
    exit()

print(f"✅ Found .env file at: {dotenv_path}")
load_dotenv(dotenv_path, override=True)

# 2. Inspect the Key
key = os.getenv('GEMINI_API_KEY')

if not key:
    print("❌ ERROR: .env found, but GEMINI_API_KEY is empty inside it.")
    exit()

# Show the first 5 and last 5 chars to verify it's the NEW key
# (Do not share the full key)
masked_key = f"{key[:5]}...{key[-5:]}"
print(f"🔑 Loaded API Key: {masked_key}")

# 3. Test a Real Request (This SHOULD show up in Google Studio)
print("\n📡 Attempting to PING Google with this key...")
try:
    genai.configure(api_key=key)
    # We use the simplest model just to trigger a usage blip
    model = genai.GenerativeModel('gemini-1.5-flash') 
    response = model.generate_content("Ping")
    print(f"✅ SUCCESS! Google replied: {response.text}")
    print("👉 Check your Google AI Studio now. You should see 1 request.")
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")