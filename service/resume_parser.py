from PyPDF2 import PdfReader
import re

try:
    from docx import Document
except ImportError:
    Document = None

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    if not Document:
        print("python-docx is not installed. Cannot parse DOCX files.")
        return ""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return clean_text(text)
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean the extracted text by removing excessive whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(file_path: str, filename: str) -> str:
    """Route file to appropriate parser based on extension."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        # Fallback to reading as text (for .txt files)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read())
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""