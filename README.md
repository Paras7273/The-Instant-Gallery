# 📸 The Instant Gallery — AI Commercial Event Photo Retrieval

**The Instant Gallery** is an AI-powered facial recognition and photo indexing platform for commercial event photographers and event attendees. Guests can snap a live selfie or upload a photo to instantly find all professional photos of themselves across large event galleries.

---

## ✨ Key Features

- 👤 **PyTorch MTCNN + FaceNet (512-d) Embeddings**: High-precision deep learning facial detection and vector representation.
- 🧩 **Automatic Person Clustering (DBSCAN)**: Group photos by guest identity automatically without manual tagging.
- ⚡ **FastAPI High-Performance Backend**: RESTful endpoints for real-time selfie matching, photo uploading, folder management, and authentication.
- 🎨 **Modern Responsive UI**: Custom CSS design supporting both Light and Dark mode, live webcam capture, smooth micro-animations, and interactive event dashboards.
- 🌐 **Live Public Access (ngrok integration)**: Expose a live HTTPS URL with venue QR codes for event attendees.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyTorch, `facenet-pytorch`, Scikit-Learn (DBSCAN), OpenCV, Pillow.
- **Frontend**: Vanilla HTML5, CSS3 with Design Tokens (Light/Dark themes), JavaScript (ES6+), Lucide Icons.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd PhotoEvent
pip install -r requirements.txt
```

### 2. Run the Application

Start the local server and web interface:

```bash
python run_app.py
```

The app will initialize the AI engine and automatically open your default browser to **`http://127.0.0.1:8000`**.

### 3. Share Live Event Link (Optional)

To enable live venue access over the internet for guests:

```bash
python start_live_ngrok.py
```

---

## 📁 Project Structure

```
PhotoEvent/
├── backend/
│   ├── ai_engine.py      # MTCNN & FaceNet neural network integration
│   ├── app.py            # FastAPI endpoints & static file serving
│   └── indexer.py        # DBSCAN clustering & vector database indexing
├── frontend/
│   ├── assets/           # UI media & static assets
│   └── index.html        # Single-page web application UI
├── data/                 # Event photos, selfies & database storage
├── create_sample_data.py # Sample data generator script
├── run_app.py            # Main server & browser launcher script
├── start_live_ngrok.py   # Public ngrok tunnel launcher
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
