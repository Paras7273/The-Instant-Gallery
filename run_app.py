import os
import sys
import time
import threading
import urllib.request
import webbrowser

def open_browser_when_ready(url, health_url, timeout=30):
    """
    Waits for the FastAPI server to start and respond on health_url,
    then automatically opens the web browser to url.
    """
    start_time = time.time()
    print(f"Waiting for backend AI engine to initialize at {url}...")
    
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print(f"\n✅ FastAPI server is online! Opening web interface at {url}\n")
                    time.sleep(0.5)
                    webbrowser.open(url)
                    return
        except Exception:
            pass
        time.sleep(0.5)
        
    print("\n⚠️ Server took too long to respond, opening browser anyway...")
    webbrowser.open(url)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("  Starting The Instant Gallery AI Server & Web UI")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")

    # Add backend directory to sys.path
    sys.path.append(backend_dir)

    server_url = "http://127.0.0.1:8000"
    health_url = f"{server_url}/api/health"

    # Launch background thread to wait for server readiness before opening browser
    threading.Thread(
        target=open_browser_when_ready, 
        args=(server_url, health_url), 
        daemon=True
    ).start()

    print(f"\n1. Initializing PyTorch/FaceNet AI models & FastAPI server at {server_url}...")
    print("   Press Ctrl+C to stop the server.\n")

    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=backend_dir)

if __name__ == "__main__":
    main()


