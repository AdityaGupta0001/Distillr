import torch
import math
from transformers import pipeline, AutoTokenizer

def summarize_text(
    text: str,
    summary_ratio: float
) -> str:
    """
    Summarizes the input text to a specified ratio of its original length
    using a Hugging Face summarization pipeline.

    Args:
        text: The input string to be summarized.
        summary_ratio: The desired summary length as a fraction of the input text length
                       (e.g., 0.2 for 20%). Should be between 0.0 and 1.0.

    Returns:
        The generated summary string, or an error message if summarization fails.
    """
    # --- Input Validation ---
    if not text or not text.strip():
        print("Warning: Input text is empty.")
        return "[Error: Input text is empty]"

    # Validate and clamp summary_ratio
    if not (0.0 < summary_ratio <= 1.0):
        original_ratio = summary_ratio
        summary_ratio = max(0.05, min(0.8, summary_ratio)) # Clamp to a reasonable range
        print(f"Warning: summary_ratio ({original_ratio}) is outside the valid range (0.0, 1.0]. Clamped to {summary_ratio}.")
    
    # --- Setup Model and Tokenizer ---
    model_name = "sshleifer/distilbart-cnn-12-6" # A common pre-trained summarization model
    min_absolute_length = 15  # Hard minimum tokens for any summary
    max_absolute_length = 512 # Hard maximum tokens (consider model limits)

    try:
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return "[Error: Tokenizer could not be loaded]"

    # --- Calculate Lengths ---
    # Encode text to get token count for ratio calculation
    # Use add_special_tokens=False for content token count
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_token_count = len(input_ids)

    if input_token_count == 0:
        print("Warning: Input text resulted in zero tokens after tokenization.")
        return "[Error: Input text has zero tokens]"

    # Calculate target max length based on ratio
    target_max_len = math.ceil(input_token_count * summary_ratio)

    # Clamp target max length within absolute boundaries
    target_max_len = min(target_max_len, max_absolute_length)
    target_max_len = max(target_max_len, min_absolute_length + 5) # Ensure max is reasonably larger than min absolute

    # Calculate target min length based on the *clamped* target max length
    # Use a fraction (e.g., 0.4) of max_len, but respect the absolute minimum
    target_min_len = math.ceil(target_max_len * 0.4)
    target_min_len = max(min_absolute_length, target_min_len)

    # Final check: Ensure min_length < max_length
    if target_min_len >= target_max_len:
        print(f"Warning: Calculated min_length ({target_min_len}) >= max_length ({target_max_len}). Adjusting.")
        # Prioritize absolute minimum, then try to create a small gap
        target_min_len = max(min_absolute_length, target_max_len - 10) # Try to enforce absolute min with gap
        target_min_len = max(5, target_min_len) # Ensure it's at least 5

        # If still bad, set a small fixed range respecting absolute min
        if target_min_len >= target_max_len:
             target_min_len = min_absolute_length
             target_max_len = min_absolute_length + 10 # Smallest possible range

    print(f"Input tokens: {input_token_count}, Target Min Tokens: {target_min_len}, Target Max Tokens: {target_max_len}")

    # --- Setup Pipeline ---
    try:
        print(f"Loading summarization pipeline for model: {model_name}")
        # Determine device: Use GPU if available, otherwise CPU
        device_id = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if device_id == 0 else "cpu"
        print(f"Using device: {device_name}")

        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=tokenizer, # Pass the loaded tokenizer
            device=device_id
        )
    except Exception as e:
        print(f"Error loading pipeline for {model_name}: {e}")
        return "[Error: Pipeline could not be loaded]"

    # --- Generate Summary ---
    print("Generating summary...")
    try:
        # Pipeline handles tokenization, running the model, and decoding
        # Pass generation parameters directly
        summary_output = summarizer(
            text,
            max_length=target_max_len,
            min_length=target_min_len,
            num_beams=4,             # Common setting for beam search
            length_penalty=1.5,      # Adjusts preference for length (values > 1 prefer longer)
            repetition_penalty=2.0,  # Penalizes repeating words/phrases
            early_stopping=True      # Stop generation when beams finish
        )
        
        # The pipeline returns a list of dictionaries
        summary = summary_output[0]['summary_text']

    except Exception as e:
        print(f"\nError during pipeline summarization: {e}")
        # Check for common errors like input length exceeding model max length
        # Note: DistilBART's default max input is often 1024 tokens
        if "maximum sequence length" in str(e):
             print(f"Hint: The input text ({input_token_count} tokens) might be too long for this model's maximum input size.")
        elif "out of memory" in str(e).lower():
             print("Hint: GPU out of memory. Try reducing batch size (if applicable) or using a smaller model/CPU.")
        summary = "[Error: Summary generation failed]"

    # --- Return Result ---
    print("\n--- Generated Summary ---")
    print(summary)
    print("------------------------")
    return summary
