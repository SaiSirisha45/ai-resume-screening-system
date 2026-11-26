"""Extract plain text from PDF, DOCX and TXT resumes."""
from io import BytesIO
from pathlib import Path


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")
