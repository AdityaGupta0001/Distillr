import PyPDF2
from docx import Document

def read_pdf(file_path):
    """Reads text from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return ' '.join(text.split())

def read_txt(file_path):
    """Reads text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_docx(file_path):
    """Reads text from a DOCX file."""
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def read_file(file_path):
    """Determines the file type and reads its content."""
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.txt'):
        return read_txt(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    else:
        return "Unsupported file format."
