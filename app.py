from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "Aditya1000101/trained_summarization_model"  # Replace with your Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

def summarize_text(text):
    summary_len = len(text.split(" "))
    """Generate an abstractive summary for the input text with a longer output."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=int(summary_len/2),  # Increased max_length for a longer summary
        min_length=int(summary_len/4),    # Increased min_length to ensure more content is retained
        length_penalty=1.0,
        num_beams=3,       # Increased num_beams for better quality summaries
        early_stopping=True,
        no_repeat_ngram_size=2,
        forced_bos_token_id=tokenizer.bos_token_id,
    )
    decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return decoded_summary

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
    if not text:
        return jsonify({"summary": "Please enter some text to summarize."})
    summary = summarize_text(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
