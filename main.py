from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import faiss
import requests
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import uuid

# # ------------------------ Config ------------------------

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------------ Config ------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Poppler & Tesseract
POPPLER_PATH = os.getenv("POPPLER_PATH")
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

# ------------------------ FastAPI App ------------------------
app = FastAPI()

# Allow CORS (frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ in production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ Session Store ------------------------
sessions = {}  # {session_id: {"index": faiss.Index, "chunks": [str]}}

# ------------------------ Helper Functions ------------------------
def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text += page_text.strip() + "\n"

        # OCR fallback
        images = convert_from_path(file_path, first_page=page_num, last_page=page_num, poppler_path=POPPLER_PATH)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                text += ocr_text.strip() + "\n"
    return text

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image).strip()
    return text if text else "Nothing extracted from the image."

def extract_text(file_content, filename):
    ext = os.path.splitext(filename)[1].lower()
    
    # Create temporary file with the content
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(tmp_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return extract_text_from_image(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    return EMBED_MODEL.encode(chunks)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def query_faiss_index(index, query_embedding, chunks, k=5):
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def generate_response_with_groq(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    json_data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=json_data)
    return response.json()["choices"][0]["message"]["content"]

# ------------------------ API Models ------------------------
class AskRequest(BaseModel):
    session_id: str
    question: str

# ------------------------ Endpoints ------------------------
@app.post("/process-file/")
async def process_file(file: UploadFile):
    text = extract_text(await file.read(), file.filename)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"index": index, "chunks": chunks}

    return {"session_id": session_id, "message": "File processed. Ready for Q/A."}

# @app.post("/process-file/")
# async def process_file(file: UploadFile):
#     try:
#         print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
        
#         # Read file content
#         file_content = await file.read()
#         print(f"File size: {len(file_content)} bytes")
        
#         # Extract text based on file type
#         text = extract_text(file_content, file.filename)
#         print(f"Extracted text length: {len(text)}")
        
#         if not text or len(text.strip()) == 0:
#             return {"error": "No text could be extracted from the file"}
        
#         # Process the text
#         chunks = chunk_text(text)
#         print(f"Created {len(chunks)} chunks")
        
#         embeddings = embed_chunks(chunks)
#         index = build_faiss_index(embeddings)
        
#         session_id = str(uuid.uuid4())
#         sessions[session_id] = {"index": index, "chunks": chunks}
        
#         print(f"Session created: {session_id}")
#         return {"session_id": session_id, "message": "File processed successfully. Ready for Q/A."}
        
#     except Exception as e:
#         print(f"Error processing file: {str(e)}")
#         return {"error": f"Failed to process file: {str(e)}"}

@app.post("/ask/")
async def ask_question(req: AskRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "Invalid session_id"}

    query_embedding = EMBED_MODEL.encode([req.question])
    top_chunks = query_faiss_index(session["index"], query_embedding, session["chunks"])
    context = "\n\n".join(top_chunks)

    prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {req.question}"
    answer = generate_response_with_groq(prompt)

    return {"answer": answer}
