import os

AUTH_KEY='<Ключ>'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

JSON_PATH = os.path.join(BASE_DIR, "data", "questions.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
DOCS_PATH = os.path.join(BASE_DIR, "docs.pkl")
