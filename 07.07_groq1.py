import os
from dotenv import load_dotenv

# 🔧 Absolute path to .env file (edit if needed)
env_path = "C:\\Users\\johnn\\Desktop\\groq_compare\\.env"

print("📂 Attempting to load:", env_path)

# ✅ Step 1: Check if the .env file exists
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ .env file not found at: {env_path}")
else:
    print("✅ .env file found.")

# ✅ Step 2: Print raw file contents
print("\n📄 .env file contents:")
with open(env_path, "r", encoding="utf-8") as f:
    print(f.read())

# ✅ Step 3: Try loading .env and accessing the key
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")
print("\n🔐 GROQ_API_KEY:", api_key if api_key else "❌ NOT FOUND")

# ✅ Step 4: Extra safety check
if not api_key:
    raise Exception("🚫 The GROQ_API_KEY was not loaded. Check formatting or try recreating the file.")


