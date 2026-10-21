"""Start a real local Uvicorn process; verify HTTP responses; stop only that process."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    command = [sys.executable, "-m", "uvicorn", "fedmaint.api.main:app", "--host", "127.0.0.1", "--port", str(port)]
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        health = json.load(response)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError(log.read().decode(errors="replace"))
                    time.sleep(0.2)
            assert health == {"status": "ok", "metadata_ready": True}, health
            for path, count in (("/api/datasets", 2), ("/api/factories", 8)):
                with urlopen(f"http://127.0.0.1:{port}{path}") as response:
                    assert len(json.load(response)) == count
            with urlopen(f"http://127.0.0.1:{port}/api/factories/azure-model4/summary") as response:
                assert json.load(response)["role"] == "held_out"
            print("PASS: live Uvicorn HTTP health, datasets, factories, held-out summary")
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
