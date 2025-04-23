
#TODO Implement the following features

#TODO Abstractive vs. Extractive Summaries: Offering the option to generate either abstractive (paraphrased) or extractive (verbatim) summaries.
#TODO Keyword Highlighting: Automatically highlight the most relevant terms.
#TODO Document and PDF Summaries: Extract summaries from PDF, DOCX, or TXT files.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig   
from peft import PeftModel
from pprint import pprint
import tempfile
from werkzeug.utils import secure_filename

from summarizer import summarize_text_chunked
from translator import translate_text
from file_processor import read_file

app = Flask(__name__)

MAX_FILE_SIZE = 3 * 1024 * 1024  # 3 MB limit
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# device = "cuda" if torch.cuda.is_available() else "cpu"

# base_model_name = "facebook/mbart-large-50"
# adapter_path = r"C:/Users/Dell/Desktop/trained_summarization_model_v4"

# if not os.path.exists(adapter_path):
#     raise FileNotFoundError(f"Adapter path not found: {adapter_path}.")

# try:
#     tokenizer = AutoTokenizer.from_pretrained(adapter_path)
#     print(f"Tokenizer loaded successfully from {adapter_path}")
# except Exception as e:
#     print(f"Error loading tokenizer from {adapter_path}: {e}")
#     print(f"Attempting to load tokenizer from base model {base_model_name}...")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     print(f"Tokenizer loaded successfully from {base_model_name}")

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

# print(f"Loading base model: {base_model_name}...")
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True
# )
# print("Base model loaded.")

# print(f"Loading PEFT adapter from: {adapter_path}...")
# model = PeftModel.from_pretrained(base_model, adapter_path)
# print("PEFT adapter loaded.")

# model = model.to(device)
# print(f"Combined PEFT model moved to {device}")

# model.eval()
# print("Model set to evaluation mode.")

# def summarize_text_chunked(text, chunk_size=1000, chunk_overlap=50, summary_ratio=0.3):
#     tokens = tokenizer.tokenize(text)
#     total_tokens = len(tokens)

#     if total_tokens == 0:
#         return ""

#     summaries = []
#     start = 0
#     chunk_count = 0

#     while start < total_tokens:
#         chunk_count += 1
#         end = min(start + chunk_size, total_tokens)
#         chunk_tokens = tokens[start:end]

#         if not chunk_tokens:
#             if start >= total_tokens: break
#             else:
#                 start = end - chunk_overlap if end > chunk_overlap else 0
#                 continue

#         chunk_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk_tokens), skip_special_tokens=True)

#         if not chunk_text.strip():
#             if end == total_tokens: break
#             start = end - chunk_overlap if end > chunk_overlap else 0
#             continue

#         inputs = tokenizer(chunk_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
#         input_token_len = inputs.input_ids.shape[1]
#         target_max_len = max(20, math.ceil(input_token_len * summary_ratio))
#         target_min_len = max(10, math.ceil(target_max_len * 0.5))

#         with torch.no_grad():
#             summary_ids = model.generate(
#                 input_ids=inputs.input_ids,
#                 attention_mask=inputs.attention_mask, # Pass attention mask
#                 max_length=target_max_len,
#                 min_length=target_min_len,
#                 length_penalty=1.5,
#                 num_beams=4,
#                 early_stopping=True,
#                 no_repeat_ngram_size=2,
#                 decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"]
#             )

#         decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(decoded_summary)

#         if end == total_tokens:
#             break

#         next_start_ideal = start + chunk_size - chunk_overlap
#         next_start_safe = end - chunk_overlap
#         start = max(next_start_ideal, next_start_safe)
#         if start <= (end - chunk_overlap if end > chunk_overlap else end):
#              start = end - chunk_overlap if end > chunk_overlap else end

#     final_summary = " ".join(summaries)
#     return final_summary

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summarizer')
def summarizer():
    return render_template('summarizer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get("text", "")
    length_ratio = float(data.get("length_ratio", 0.3))
    if not text:
        return jsonify({"summary": "Please enter some text to summarize."})
    summary = summarize_text_chunked(text, chunk_size=900, chunk_overlap=100, base_summary_ratio=length_ratio)
    return jsonify({"summary": summary})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "")

    if not text:
        return jsonify({"error": "Please enter text to translate."}), 400
    if not target_lang:
        return jsonify({"error": "Please select a language."}), 400

    translated_text = translate_text(text, target_lang)
    print(translated_text)
    return jsonify({"translated_text": translated_text})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        temp_dir = None
        temp_filepath = None

        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_filepath = os.path.join(temp_dir, filename)

            # Save the file temporarily
            file.save(temp_filepath)

            # Check file size after saving (more reliable)
            if os.path.getsize(temp_filepath) > MAX_FILE_SIZE:
                 return jsonify({"error": f"File exceeds the size limit of {MAX_FILE_SIZE / (1024*1024)} MB."}), 400

            # Process the file using your file_processor
            extracted_text = read_file(temp_filepath)

            if "Unsupported file format" in extracted_text:
                 return jsonify({"error": "Unsupported file format."}), 400

            # Return the extracted text
            return jsonify({"text": extracted_text})

        except Exception as e:
            print(f"Error processing file {filename}: {e}") # Log the error
            # Provide a more generic error to the user
            return jsonify({"error": "An error occurred while processing the file."}), 500
        finally:
            # Clean up the temporary file and directory
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    else:
        return jsonify({"error": "Invalid file type. Only .txt, .pdf, and .docx are allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True)
