"""Starts Streamlit with a local browser and keeps the server tied to this console."""
import subprocess
import sys
import time
import urllib.request
import webbrowser


def main():
    address = "http://localhost:8501"
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", "8501", "--browser.gatherUsageStats", "false"]
    print("Launching MachineGuard AI at", address)
    print("Leave this window open; press Ctrl+C to stop the app.")
    proc = subprocess.Popen(cmd)
    try:
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError("Streamlit exited unexpectedly; see error above.")
            try:
                with urllib.request.urlopen(address + "/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(address)
                        break
            except (OSError, ValueError):
                time.sleep(.5)
        else:
            print("Browser did not open automatically. Visit", address)
        return proc.wait()
    except KeyboardInterrupt:
        print("\nStopping MachineGuard AI...")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
