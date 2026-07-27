import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import unquote

try:
    from indexer import GalleryIndexer
    from ai_engine import FaceEngine
except ImportError:
    from backend.indexer import GalleryIndexer
    from backend.ai_engine import FaceEngine

app = FastAPI(
    title="The Instant Gallery - AI Photo Retrieval API",
    description="Facial Recognition & Vector Search API with Auth, Folder Management & DBSCAN Person Clustering",
    version="3.0.0"
)

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY_DIR = os.path.join(BASE_DIR, "data", "event_gallery")
SELFIES_DIR = os.path.join(BASE_DIR, "data", "test_selfies")
THUMBS_DIR = os.path.join(BASE_DIR, "data", "person_thumbnails")
ASSETS_DIR = os.path.join(BASE_DIR, "frontend", "assets")
DB_FILE = os.path.join(BASE_DIR, "data", "indexed_db", "embeddings.json")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(SELFIES_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# Mount static folders to serve gallery images, thumbnails & assets directly to web UI
app.mount("/gallery-photos", StaticFiles(directory=GALLERY_DIR), name="gallery-photos")
app.mount("/person-thumbnails", StaticFiles(directory=THUMBS_DIR), name="person-thumbnails")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Initialize indexer
indexer = GalleryIndexer(gallery_dir=GALLERY_DIR, db_file=DB_FILE, thumbs_dir=THUMBS_DIR)

class CreateFolderRequest(BaseModel):
    folder_name: str

class DeletePhotoRequest(BaseModel):
    rel_path: str

class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    name: str
    email: str
    studio_name: str
    password: str

# Photographers memory database for demo authentication
PHOTOGRAPHERS_DB = {
    "photographer@event.com": {
        "name": "Alex Vance",
        "email": "photographer@event.com",
        "studio_name": "Vance Event Photography",
        "password": "password123"
    }
}

@app.get("/")
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "The Instant Gallery AI Engine v3.0",
        "facenet_enabled": indexer.engine.use_facenet,
        "clustering_algorithm": "DBSCAN (Cosine Distance)"
    }

@app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "service": "The Instant Gallery AI Engine v3.0",
        "facenet_enabled": indexer.engine.use_facenet,
        "clustering_algorithm": "DBSCAN (Cosine Distance)"
    }

@app.post("/api/auth/login")
def photographer_login(req: LoginRequest):
    """Photographer / Admin Login endpoint."""
    email = req.email.lower().strip()
    user = PHOTOGRAPHERS_DB.get(email)
    if not user or user["password"] != req.password:
        if req.email and req.password:
            user = {
                "name": email.split("@")[0].title(),
                "email": email,
                "studio_name": "Studio Gallery Pro",
                "password": req.password
            }
            PHOTOGRAPHERS_DB[email] = user
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "status": "success",
        "message": "Login successful",
        "token": "demo_photographer_token_98765",
        "user": {
            "name": user["name"],
            "email": user["email"],
            "studio_name": user["studio_name"]
        }
    }

@app.post("/api/auth/signup")
def photographer_signup(req: SignupRequest):
    """Photographer Registration / Signup endpoint."""
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    email = req.email.lower().strip()
    user = {
        "name": req.name or "Photographer",
        "email": email,
        "studio_name": req.studio_name or "Event Studio Pro",
        "password": req.password
    }
    PHOTOGRAPHERS_DB[email] = user
    return {
        "status": "success",
        "message": "Account created successfully",
        "token": "demo_photographer_token_98765",
        "user": {
            "name": user["name"],
            "email": user["email"],
            "studio_name": user["studio_name"]
        }
    }

@app.get("/api/download-photo")
def download_photo(rel_path: str):
    """Allows attendees to download high-resolution event photos directly."""
    decoded_rel = unquote(rel_path)
    clean_rel = decoded_rel.replace('\\', '/').lstrip('/')
    full_path = os.path.join(GALLERY_DIR, clean_rel)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Photo file not found: {clean_rel}")
    filename = os.path.basename(full_path)
    return FileResponse(
        path=full_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/api/gallery")
def get_gallery():
    """Returns all indexed event photos, face metadata, folders, and unique person cluster profiles."""
    if not os.path.exists(DB_FILE):
        return {
            "total_photos": 0, 
            "total_faces": 0, 
            "total_unique_persons": 0, 
            "folders": indexer.get_folders(),
            "persons": [], 
            "records": []
        }
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "folders" not in data:
        data["folders"] = indexer.get_folders()
    return data

@app.get("/api/folders")
def get_folders():
    """Returns all gallery folders."""
    return indexer.get_folders()

@app.post("/api/folders")
def create_folder(req: CreateFolderRequest):
    """Creates a new folder inside event gallery."""
    if not req.folder_name or not req.folder_name.strip():
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")
    safe_name = indexer.create_folder(req.folder_name)
    summary = indexer.index_gallery()
    return {
        "message": f"Folder '{safe_name}' created successfully.",
        "folder_name": safe_name,
        "summary": summary
    }

@app.delete("/api/folders/{folder_name}")
def delete_folder(folder_name: str):
    """Deletes a folder and all contained photos, then re-indexes."""
    summary = indexer.delete_folder(folder_name)
    return {
        "message": f"Folder '{folder_name}' deleted successfully.",
        "summary": summary
    }

@app.post("/api/upload-gallery")
async def upload_gallery(
    files: List[UploadFile] = File(...),
    folder_name: Optional[str] = Form(default=None)
):
    """Bulk upload photos by photographer into optional folder & auto-index embeddings + person clusters."""
    target_dir = GALLERY_DIR
    if folder_name and folder_name.strip() and folder_name.strip().lower() not in ["default", "default gallery"]:
        safe_folder = indexer.create_folder(folder_name)
        target_dir = os.path.join(GALLERY_DIR, safe_folder)
        os.makedirs(target_dir, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    # Re-index full gallery and calculate unique person clusters
    summary = indexer.index_gallery()
    return {
        "message": f"Successfully uploaded {len(saved_files)} images.",
        "indexed_summary": summary
    }

@app.post("/api/delete-photo")
def delete_photo(req: DeletePhotoRequest):
    """Deletes an individual photo and re-indexes."""
    if not req.rel_path:
        raise HTTPException(status_code=400, detail="Photo path required")
    summary = indexer.delete_photo(req.rel_path)
    return {
        "message": f"Photo '{req.rel_path}' deleted successfully.",
        "summary": summary
    }

@app.delete("/api/photos")
def delete_photo_delete_method(rel_path: str):
    """Deletes an individual photo (DELETE method) and re-indexes."""
    if not rel_path:
        raise HTTPException(status_code=400, detail="Photo path required")
    summary = indexer.delete_photo(rel_path)
    return {
        "message": f"Photo '{rel_path}' deleted successfully.",
        "summary": summary
    }

@app.post("/api/clear-all")
def clear_all():
    """Deletes all photos and folders."""
    summary = indexer.clear_all()
    return {
        "message": "All photos and folders cleared successfully.",
        "summary": summary
    }

@app.post("/api/search-selfie")
async def search_selfie(file: UploadFile = File(...), threshold: float = 0.55):
    """
    High-precision search endpoint: Receives attendee selfie, extracts aligned 512-d FaceNet vector,
    and returns matching event photos with identity recognition & confidence scores.
    """
    selfie_path = os.path.join(SELFIES_DIR, file.filename)
    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    matches = indexer.search_by_selfie(selfie_path, similarity_threshold=threshold)

    # Format paths to public static URLs
    for match in matches:
        match['url'] = f"/gallery-photos/{match['rel_path']}"

    return {
        "query_image": file.filename,
        "match_count": len(matches),
        "threshold_used": threshold,
        "matches": matches
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

