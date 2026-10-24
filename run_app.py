import sys
import subprocess
import webbrowser
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
VENV_PYTHON = REPO_DIR / ".venv312" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = REPO_DIR.parent / ".venv" / "Scripts" / "python.exe"
    if not VENV_PYTHON.exists():
        VENV_PYTHON = Path(sys.executable)

import socket

def find_available_port(preferred_port=8000, fallback_ports=(8080, 8001, 8002, 8888)):
    candidates = [preferred_port] + list(fallback_ports)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred_port

def main():
    port = find_available_port()

    print("=" * 65)
    print("  MinuteAI — AI Meeting Minutes Generator Launcher")
    print("=" * 65)
    print(f"Working Directory : {REPO_DIR}")
    print(f"Python Executable : {VENV_PYTHON}")
    print(f"Starting Server   : http://127.0.0.1:{port}")
    print("Press CTRL+C in this terminal to stop the server.")
    print("=" * 65)

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{port}")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        subprocess.run(
            [
                str(VENV_PYTHON),
                "-m",
                "uvicorn",
                "src.backend.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--reload",
            ],
            cwd=str(REPO_DIR),
            check=True
        )
    except KeyboardInterrupt:
        print("\n[Server Stopped] Thank you for using MinuteAI!")

if __name__ == "__main__":
    main()
