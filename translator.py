import requests
from dotenv import load_dotenv
import os
import json
from groq import Groq
load_dotenv()

with open('./static/supported_languages.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    language_dict = data[0]

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
        lang = "English"
        for language, code in language_dict.items():
            if code==target_lang:
                lang = language
                break
        print(lang)
        client = Groq(
           api_key=str(os.environ.get("GROQ_API_KEY2")),
        )
        chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text+f"\n\nTranslate this text into {lang}. Only provide the translated text as the response and nothing else.",
            }
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return e
    # try:
    #     url = TRANSLATOR_API_URL
    #     params = {
    #         "q": text,
    #         "langpair": f"en|{target_lang}",
    #         "key": TRANSLATOR_API_KEY
    #     }
    #     response = requests.get(url, params=params)
    #     response.raise_for_status()
    #     translated_text = response.json()["responseData"]["translatedText"]
    #     print(translated_text)
    #     if "PLEASE SELECT TWO DISTINCT LANGUAGES" in translated_text or "QUERY LENGTH LIMIT EXCEEDED" in translated_text:
    #         raise Exception("Translator API Error")
    #     else:
    #         return translated_text
    # except:
        


# dat = translate_text("Hello", target_lang="fr")
# print(dat)