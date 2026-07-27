import os
import sys
import time
import webbrowser
from pyngrok import ngrok

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 65)
    print("Starting Live Public Server for The Instant Gallery (ngrok)")
    print("=" * 65)

    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    saved_token_path = os.path.join(os.path.dirname(__file__), ".ngrok_token")

    if not authtoken and os.path.exists(saved_token_path):
        with open(saved_token_path, "r", encoding="utf-8") as f:
            authtoken = f.read().strip()

    if not authtoken:
        print("\n📌 TIP: ngrok requires a free auth token to start public HTTPS tunnels.")
        print("   1. Sign up free at: https://dashboard.ngrok.com/signup")
        print("   2. Copy your Authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken\n")
        try:
            user_input = input("Enter your free ngrok Authtoken (or press Enter to attempt default connection): ").strip()
            if user_input:
                authtoken = user_input
                with open(saved_token_path, "w", encoding="utf-8") as f:
                    f.write(authtoken)
        except Exception:
            pass

    try:
        if authtoken:
            ngrok.set_auth_token(authtoken)

        print("\n1. Launching ngrok tunnel on http://127.0.0.1:8000 ...")
        tunnel = ngrok.connect(8000, "http")
        public_url = tunnel.public_url
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://")

        print("\n" + "=" * 65)
        print("🎉 YOUR WEBSITE IS NOW LIVE OVER THE INTERNET!")
        print("=" * 65)
        print(f"\n🌐 Live Public HTTPS URL:  {public_url}")
        print(f"📱 Venue QR Code Link:    {public_url}")
        print(f"🤳 Guest Mobile Search:   {public_url}")
        print("\n" + "=" * 65)
        print("Share this link with event guests or display the venue QR code!")
        print("Press Ctrl+C to stop the live public server.\n")

        webbrowser.open(public_url)

        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()

    except OSError as e:
        if "225" in str(e) or "virus" in str(e).lower() or "potentially unwanted" in str(e).lower():
            print("\n❌ WINDOWS DEFENDER BLOCKED NGROK (WinError 225)")
            print("=" * 65)
            print("Windows Security flagged the 'ngrok.exe' binary as a false-positive threat.")
            print("\nHow to solve this:")
            print("1. Open Windows Security (Start Menu -> type 'Windows Security').")
            print("2. Go to 'Virus & threat protection' -> 'Protection history'.")
            print("3. Find the recent blocked threat for 'ngrok.exe' and click 'Actions' -> 'Allow on device'.")
            print("4. Re-run: python start_live_ngrok.py")
            print("=" * 65)
        else:
            print(f"\n❌ System error: {e}")
    except Exception as e:
        print(f"\n❌ Tunnel error: {e}")
        print("\nHow to fix:")
        print("1. Get your free authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
        print("2. Run command: python -c \"from pyngrok import ngrok; ngrok.set_auth_token('YOUR_AUTH_TOKEN')\"")

if __name__ == "__main__":
    main()
