import os

AUTH_KEY='MDE5YmE3ZmUtNTk1MS03OWI3LTg0MTgtZjdmODc0NmJjNjAzOmIyNjQ4OGRhLTk2MGItNDVmZS1iNGJlLTA2ODRhZWEwMzVhYg=='

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

JSON_PATH = os.path.join(BASE_DIR, "data", "questions.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
DOCS_PATH = os.path.join(BASE_DIR, "docs.pkl")
