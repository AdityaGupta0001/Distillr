import requests
from dotenv import load_dotenv
import os
load_dotenv()

TRANSLATOR_API_URL = os.getenv('TRANSLATOR_API_URL')
TRANSLATOR_API_KEY = os.getenv('TRANSLATOR_API_KEY')

def translate_text(text, target_lang="hi"):
    """
    Translate text using MyMemory API
    Args:
        text: Text to translate
        target_lang: ISO 639-1 language code (e.g. "hi" for Hindi)
    """
    try:
        url = TRANSLATOR_API_URL
        params = {
            "q": text,
            "langpair": f"en|{target_lang}",
            "key": TRANSLATOR_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()["responseData"]["translatedText"]
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# dat = translate_text("Hello", target_lang="fr")
# print(dat)