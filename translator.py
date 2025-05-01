# translator.py
import requests
from dotenv import load_dotenv
import os
import json
from groq import Groq

load_dotenv()

# --- Language Dictionary (Optional but good for reference) ---
# You might not need this for the current logic if detect_lang returns full names
# and translate_text uses names directly. Keep it if you need code mapping elsewhere.
# try:
#     with open('./static/supported_languages.json', 'r', encoding='utf-8') as file:
#         data = json.load(file)
#         language_dict = data[0]
# except FileNotFoundError:
#     print("Warning: supported_languages.json not found. Language code mapping unavailable.")
#     language_dict = {}
# except Exception as e:
#     print(f"Error loading supported_languages.json: {e}")
#     language_dict = {}

# --- Groq Client Initialization ---
# Consider initializing the client once if possible, but ensure API key is loaded correctly
try:
    groq_api_key = os.environ.get("GROQ_API_KEY2")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY2 environment variable not set.")
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"FATAL: Failed to initialize Groq client: {e}")
    groq_client = None # Indicate client failed to load

# --- Translation Function ---
def translate_text(text: str, target_language_name: str) -> str:
    """
    Translate text using the Groq API.

    Args:
        text: Text to translate.
        target_language_name: The full name of the target language (e.g., "Spanish", "French").
                                This name will be used directly in the prompt.
    Returns:
        Translated text string or an error message string.
    """
    if not groq_client:
        return "[Error: Groq client not initialized]"
    if not text:
        return "[Error: No text provided for translation]"
    if not target_language_name:
        return "[Error: No target language specified]"

    print(f"Attempting to translate to: {target_language_name}")
    try:
        # Directly use the target_language_name in the prompt
        prompt = f'{text}\n\nTranslate the above text into {target_language_name}. Only provide the translated text as the response, with no additional explanations or conversational text.'

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192", # Use a capable model like Llama 3 70b
            temperature=0.2,        # Lower temperature for more deterministic translation
            stream=False,
            # Consider adding response_format={"type": "text"} if available/needed
        )

        translated_content = chat_completion.choices[0].message.content
        print(f"Groq Translation successful to {target_language_name}.")
        # Optional: Add basic cleaning if Groq sometimes adds extra quotes or labels
        # translated_content = translated_content.strip().strip('"')
        return translated_content

    except Exception as e:
        print(f"ERROR in translate_text to {target_language_name}: {e}")
        # Log the specific error if possible
        # Consider logging traceback: import traceback; traceback.print_exc()
        return f"[Error translating text to {target_language_name}]"

# --- Language Detection Function ---
def detect_lang(text: str) -> str:
    """
    Detect the language of the input text using the Groq API.

    Args:
        text: Text whose language needs detection.

    Returns:
        The detected language name (e.g., "English", "Spanish") or an error message string.
    """
    if not groq_client:
        return "[Error: Groq client not initialized]"
    if not text:
        return "[Error: No text provided for language detection]"

    print("Attempting language detection...")
    try:
        # Updated prompt for clarity and robustness
        prompt = f'"{text}"\n\nWhat language is the above text written in? Respond with only the name of the language (e.g., "English", "Spanish", "French"). Do not include any other words or punctuation.'

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # A smaller model is often sufficient for detection
            temperature=0.0,       # Zero temperature for highest confidence prediction
            stream=False,
        )

        detected_language = chat_completion.choices[0].message.content.strip().capitalize()
        # Basic cleaning: remove potential trailing periods or quotes if the model adds them
        if detected_language.endswith('.'):
             detected_language = detected_language[:-1]
        detected_language = detected_language.strip('"')

        print(f"Groq Detection successful: {detected_language}")
        return detected_language

    except Exception as e:
        print(f"ERROR in detect_lang: {e}")
        return "[Error detecting language]"

