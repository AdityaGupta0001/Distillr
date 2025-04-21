import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import math
from tqdm import tqdm
from langdetect import detect
from threading import Lock
from dotenv import load_dotenv
import os
load_dotenv()

tokenizer_lock = Lock()

from langdetect import detect

def get_lang_code(text):
    try:
        detected_lang = detect(text)
        lang_map = {
            "af": "af_ZA", "ar": "ar_AR", "bg": "bg_BG", "bn": "bn_BD", "ca": "ca_ES",
            "cs": "cs_CZ", "cy": "cy_GB", "da": "da_DK", "de": "de_DE", "el": "el_GR",
            "en": "en_XX", "es": "es_XX", "et": "et_EE", "fa": "fa_IR", "fi": "fi_FI",
            "fr": "fr_XX", "gu": "gu_IN", "he": "he_IL", "hi": "hi_IN", "hr": "hr_HR",
            "hu": "hu_HU", "id": "id_ID", "it": "it_IT", "ja": "ja_XX", "kn": "kn_IN",
            "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "mk": "mk_MK", "ml": "ml_IN",
            "mr": "mr_IN", "ne": "ne_NP", "nl": "nl_XX", "no": "no_NO", "pa": "pa_IN",
            "pl": "pl_PL", "pt": "pt_XX", "ro": "ro_RO", "ru": "ru_RU", "sk": "sk_SK",
            "sl": "sl_SI", "so": "so_SO", "sq": "sq_AL", "sv": "sv_SE", "sw": "sw_KE",
            "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tl": "tl_PH", "tr": "tr_TR",
            "uk": "uk_UA", "ur": "ur_PK", "vi": "vi_VN", "zh-cn": "zh_CN", "zh-tw": "zh_TW",
            "zh": "zh_CN"
        }
        return lang_map.get(detected_lang, "en_XX")
    except Exception:
        return "en_XX"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model_name = "facebook/mbart-large-50"

adapter_path = os.getenv("MODEL_PATH")

try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    print(f"Tokenizer loaded successfully from {adapter_path}")
except Exception as e:
    print(f"Error loading tokenizer from {adapter_path}: {e}")
    print(f"Attempting to load tokenizer from base model {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print(f"Tokenizer loaded successfully from {base_model_name}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

print(f"Loading base model: {base_model_name}...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
print("Base model loaded.")

print(f"Loading PEFT adapter from: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("PEFT adapter loaded.")

model = model.to(device)
print(f"Combined PEFT model moved to {device}")

model.eval()
print("Model set to evaluation mode.")

def summarize_text_chunked(text, chunk_size=1000, chunk_overlap=50, summary_ratio=0.3):
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return ""

    step_size = max(1, chunk_size - chunk_overlap)
    estimated_chunks = math.ceil(max(1, total_tokens - chunk_overlap) / step_size) if total_tokens > chunk_size else 1

    summaries = []
    start = 0

    with tqdm(total=estimated_chunks, desc="Summarizing Chunks", unit="chunk") as pbar:
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]

            if not chunk_tokens:
                if start >= total_tokens: break
                else:
                    start = end - chunk_overlap if end > chunk_overlap else 0
                    continue

            chunk_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk_tokens), skip_special_tokens=True)

            if not chunk_text.strip():
                if end == total_tokens: break
                start = end - chunk_overlap if end > chunk_overlap else 0
                continue

            lang_code = get_lang_code(chunk_text)

            with tokenizer_lock:
                tokenizer.src_lang = lang_code
                inputs = tokenizer(chunk_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

            input_token_len = inputs.input_ids.shape[1]
            target_max_len = max(20, math.ceil(input_token_len * summary_ratio))
            target_min_len = max(10, math.ceil(target_max_len * 0.5))

            with torch.no_grad():
                summary_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=target_max_len,
                    min_length=target_min_len,
                    length_penalty=1.5,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    decoder_start_token_id=tokenizer.lang_code_to_id[lang_code]
                )

            decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(decoded_summary)

            if end == total_tokens:
                break

            next_start_ideal = start + chunk_size - chunk_overlap
            next_start_safe = end - chunk_overlap
            start = max(next_start_ideal, next_start_safe)
            if start <= pbar.n * step_size:
                start = end - chunk_overlap if end > chunk_overlap else end

    final_summary = " ".join(summaries)
    print(final_summary)
    return final_summary



# # --- Test it ---
# user_input = """Weather is the day-to-day condition of the Earth's atmosphere in a particular place and time. It includes various atmospheric phenomena such as temperature, humidity, precipitation (rain, snow, sleet), wind, visibility, and atmospheric pressure. Unlike climate, which refers to long-term patterns over decades or centuries, weather is temporary and can change rapidly—sometimes within minutes or hours.

# The main driver of weather is the sun. Solar energy heats the Earth’s surface unevenly due to factors like latitude, surface materials, and cloud cover. This uneven heating causes differences in air pressure and temperature, setting the atmosphere in motion. Warm air rises, cool air sinks, and these movements generate wind and storms. Water vapor in the atmosphere also plays a crucial role—when it condenses into clouds and falls as precipitation, it redistributes heat and moisture around the globe.

# Weather can vary significantly from place to place. For example, tropical regions often experience warm temperatures and high humidity year-round, while polar regions remain cold and dry. Mountainous areas might see rapid weather shifts due to elevation, and coastal cities usually have milder, more stable weather due to the moderating influence of the ocean.

# Meteorologists study weather to predict upcoming conditions using data from satellites, weather stations, balloons, and radar systems. Modern forecasting relies heavily on computer models that simulate atmospheric processes based on current data. While short-term forecasts (up to five days) are generally reliable, long-range forecasts are less accurate due to the complexity and chaotic nature of the atmosphere.

# Weather affects almost every aspect of human life. Agriculture, transportation, construction, and even mood and health are influenced by weather conditions. For instance, prolonged droughts can lead to crop failure, while heavy snowstorms can disrupt travel and power supplies. Severe weather events like hurricanes, tornadoes, and floods can cause widespread damage and loss of life.

# Climate change is also influencing weather patterns. Rising global temperatures are leading to more extreme and unpredictable weather events. For example, we are seeing more intense heatwaves, stronger hurricanes, and shifting rainfall patterns. Understanding and adapting to these changes is one of the major challenges of our time.

# Despite technological advances, weather remains a complex and fascinating natural phenomenon. From the calm of a sunny afternoon to the fury of a thunderstorm, weather reflects the dynamic balance of natural forces at work in our atmosphere. It connects us to our environment and reminds us of nature’s power and unpredictability.

# In conclusion, weather is more than just a daily forecast—it is a vital part of the Earth's system that influences life on a global scale. By studying weather, we gain insight into the interactions between air, water, and land, and we learn how to prepare for the conditions that shape our daily experiences."""

# summary = summarize_text_chunked(user_input, chunk_size=900, chunk_overlap=100, summary_ratio=0.3)
# print("\n--- Final Generated Summary ---\n", summary)