import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import math
from tqdm import tqdm
from langdetect import detect
from threading import Lock
from dotenv import load_dotenv
import os

# --- Assume previous setup code is here (imports, load_dotenv, get_lang_code, model/tokenizer loading) ---
load_dotenv()
tokenizer_lock = Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"

# (Include your get_lang_code function here)
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
        return "en_XX" # Default to English on error

# --- Load Model and Tokenizer ---
# (Your model/tokenizer loading code - ensure model and tokenizer are loaded globally or passed)
# Example placeholder - replace with your actual loading logic
print(f"Using device: {device}")
base_model_name = "facebook/mbart-large-50"
adapter_path = os.getenv("MODEL_PATH") # Make sure MODEL_PATH is set in your .env
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
except Exception:
    print(f"Loading base tokenizer {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
print(f"Loading base model {base_model_name}")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print(f"Loading PEFT adapter {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()
print("Model loaded and ready.")
# --- End Model Loading ---


def summarize_text_chunked(
    text: str,
    chunk_size: int = 1000,        # Max tokens per chunk input
    chunk_overlap: int = 150,       # Overlap between chunks (more overlap can improve coherence)
    base_summary_ratio: float = 0.4,# Default desired ratio (can be tuned)
    min_summary_tokens: int = 40,   # Absolute minimum summary tokens (tune this!)
    max_summary_tokens: int = 300   # Absolute maximum summary tokens per chunk (tune this!)
) -> str:
    """
    Summarizes text using a chunking approach with adaptive length calculation.

    Args:
        text: The input text to summarize.
        chunk_size: Maximum number of tokens processed by the model in one go.
        chunk_overlap: Number of tokens to overlap between chunks.
        base_summary_ratio: The desired summary length as a fraction of input length.
        min_summary_tokens: The hard minimum number of tokens for any chunk's summary.
        max_summary_tokens: The hard maximum number of tokens for any chunk's summary.

    Returns:
        The generated summary string.
    """
    if not text or not text.strip():
        print("Warning: Input text is empty.")
        return ""

    # Use encode for potentially more accurate token count respecting special tokens
    # Use add_special_tokens=False if you only want content tokens for length calculation
    all_input_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(all_input_ids)

    if total_tokens == 0:
         print("Warning: Input text resulted in zero tokens.")
         return ""

    # Ensure overlap is less than chunk_size
    chunk_overlap = min(chunk_overlap, chunk_size - 50) # Ensure substantial non-overlap
    if chunk_overlap < 0: chunk_overlap = 0

    step_size = max(1, chunk_size - chunk_overlap)
    # Estimate chunks more accurately based on actual token IDs
    estimated_chunks = math.ceil(max(1, total_tokens - chunk_overlap) / step_size) if total_tokens > chunk_size else 1

    summaries = []
    start_idx = 0

    print(f"Total tokens: {total_tokens}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}, Step size: {step_size}")

    with tqdm(total=estimated_chunks, desc="Summarizing Chunks", unit="chunk") as pbar:
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_input_ids = all_input_ids[start_idx:end_idx]

            if not chunk_input_ids:
                 if start_idx >= total_tokens: break # End of text
                 else: # Should not happen with correct logic, but safe-guard
                      print(f"Warning: Empty chunk slice at index {start_idx}. Advancing.")
                      start_idx += step_size # Advance to avoid infinite loop
                      continue

            # Decode only the current chunk's IDs for language detection and model input
            # skip_special_tokens=True is important here for clean text
            chunk_text = tokenizer.decode(chunk_input_ids, skip_special_tokens=True)

            if not chunk_text or not chunk_text.strip():
                print(f"Warning: Decoded chunk is empty at index {start_idx}. Skipping.")
                # Advance pointer: Move to the beginning of the next logical chunk
                start_idx += step_size
                pbar.update(1)
                continue

            lang_code = get_lang_code(chunk_text)

            # Prepare model inputs - Use the raw IDs for the model tensor
            # Add special tokens (like <s>, </s>) expected by the model here
            # We use encode_plus to get attention mask easily
            with tokenizer_lock:
                 tokenizer.src_lang = lang_code # Set source language for mBART
                 # Let the tokenizer handle adding special tokens and creating attention mask
                 inputs = tokenizer.encode_plus(
                      chunk_text,
                      return_tensors="pt",
                      max_length=tokenizer.model_max_length, # Use tokenizer's max length (e.g., 1024 for mBART)
                      truncation=True,
                      padding=False # No padding needed for single sequence generation
                 )


            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            input_token_len = input_ids.shape[1] # Actual length fed to model

            # --- Optimized Length Calculation ---
            # 1. Calculate desired length based on ratio of *actual* model input length
            desired_max_len = math.ceil(input_token_len * base_summary_ratio)

            # 2. Clamp the desired length between absolute min and max
            target_max_len = max(min_summary_tokens, desired_max_len)
            target_max_len = min(target_max_len, max_summary_tokens)

            # 3. Calculate min_length based on the final clamped target_max_len
            #    Using 0.4 or 0.5 * max_len as min often gives a good range.
            target_min_len = max(10, math.ceil(target_max_len * 0.4))
            # Ensure min_length is strictly less than max_length
            target_min_len = min(target_min_len, target_max_len - 5) # Leave a gap
            target_min_len = max(10, target_min_len) # Ensure min_len doesn't drop too low

            # Final defensive check: Ensure min < max
            if target_min_len >= target_max_len:
                print(f"Warning: Min length ({target_min_len}) >= Max length ({target_max_len}). Adjusting.")
                if target_max_len > min_summary_tokens + 5: # Can we lower max?
                     target_max_len = target_min_len + 5
                else: # Must lower min
                     target_min_len = max(10, target_max_len - 5)
                # Last resort if still bad
                if target_min_len >= target_max_len:
                     target_min_len = 10
                     target_max_len = max(min_summary_tokens, 20)
                print(f"--> Adjusted lengths: Min={target_min_len}, Max={target_max_len}")
            # --- End Length Calculation ---

            try:
                with torch.no_grad():
                    summary_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=target_max_len,
                        min_length=target_min_len,
                        length_penalty=1.5,  # Encourages model not to stop too early (tune this)
                        num_beams=4,         # Beam search width (tune this)
                        early_stopping=True, # Stop when beams converge
                        no_repeat_ngram_size=2, # Avoid repeating phrases
                        # Set target language for mBART using detected source lang code
                        decoder_start_token_id=tokenizer.lang_code_to_id[lang_code]
                    )

                # Decode the generated summary
                # skip_special_tokens=True removes language codes and <s>/</s>
                decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(decoded_summary.strip())

            except Exception as e:
                 print(f"\nError during generation for chunk starting at index {start_idx}: {e}")
                 # Decide how to handle: skip chunk, add placeholder, etc.
                 summaries.append("[GENERATION ERROR]") # Add placeholder

            # Advance pointer for the next chunk
            start_idx += step_size
            pbar.update(1)


    # Combine chunk summaries
    final_summary = " ".join(s for s in summaries if s and s != "[GENERATION ERROR]") # Join non-empty, non-error summaries

    # --- Optional Final Pass (Advanced) ---
    # If the combined summary is very long (e.g., > 1024 tokens) and originated
    # from many chunks, you might consider summarizing the combined summary itself
    # for better coherence. This adds complexity and compute time.
    # combined_tokens = tokenizer.encode(final_summary, add_special_tokens=False)
    # if len(combined_tokens) > tokenizer.model_max_length and estimated_chunks > 3: # Example condition
    #     print("\nRunning final consolidation pass...")
    #     final_summary = summarize_text_chunked(
    #         final_summary,
    #         # Use different settings for the final pass if needed
    #         chunk_size=tokenizer.model_max_length,
    #         chunk_overlap=200,
    #         base_summary_ratio=0.5, # Maybe allow longer final summary
    #         min_summary_tokens=100,
    #         max_summary_tokens=500
    #     )
    # --- End Optional Final Pass ---


    print("\n--- Final Generated Summary ---")
    print(final_summary)
    return final_summary

# --- Test it ---
# # user_input_short = "Weather is the day-to-day condition of the Earth's atmosphere. It includes temperature, humidity, and wind."
# user_input_long = """Sustainable development is crucial for ensuring a balanced relationship between economic growth, environmental preservation, and social well-being. It aims to meet the needs of the present without compromising the ability of future generations to meet their own needs. As global populations rise and natural resources become increasingly strained, sustainable development offers a framework for using resources efficiently while reducing environmental impact.

# This approach promotes renewable energy, responsible consumption, and conservation of biodiversity, helping combat climate change and environmental degradation. Socially, it focuses on reducing poverty, improving education, and promoting equality, thereby fostering inclusive and resilient communities. Economically, it encourages innovation and long-term planning, making businesses more adaptive and environmentally conscious."""

# # print("\nTesting with SHORT input:")
# # summary_short = summarize_text_chunked(user_input_short)

# print("\nTesting with LONG input:")
# # Ensure you have a long text example here if you want to test chunking
# summary_long = summarize_text_chunked(user_input_long)