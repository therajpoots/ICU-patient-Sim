"""
ICU Biosignal Simulator — Main Runner
Generates TLS certs if needed, then launches the FastAPI server.

Usage:
    python run_simulator.py           # HTTPS on port 8443 (default)
    python run_simulator.py --http    # HTTP on port 8080 (dev/testing only)
    python run_simulator.py --port 9000
    python run_simulator.py --reload  # hot-reload (dev)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CERT_FILE = Path("certs/server.crt")
KEY_FILE  = Path("certs/server.key")
HOST      = os.getenv("HOST", "0.0.0.0")
PORT      = int(os.getenv("PORT", "8443"))
HTTP_PORT = 8080    # plain-HTTP fallback port


def ensure_env():
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("📄 Created .env from .env.example")
            print("   → Edit .env to set AES_SHARED_SECRET and API_KEY before going to production!\n")


def ensure_certs():
    if not CERT_FILE.exists() or not KEY_FILE.exists():
        print("🔐 TLS certificates not found — generating self-signed cert...")
        import setup_tls
        setup_tls.generate_self_signed_cert()
    else:
        print(f"✅ TLS certificates: {CERT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="ICU Biosignal Simulator Server")
    parser.add_argument("--http",   action="store_true",
                        help="Run over plain HTTP (no TLS) — for local testing only")
    parser.add_argument("--port",   type=int, default=None,
                        help="Override port (default: 8443 TLS, 8080 HTTP)")
    parser.add_argument("--host",   type=str, default=HOST)
    parser.add_argument("--reload", action="store_true",
                        help="Enable hot-reload (development only)")
    args = parser.parse_args()

    ensure_env()

    use_tls = not args.http
    port    = args.port or (PORT if use_tls else HTTP_PORT)
    scheme  = "https" if use_tls else "http"

    if use_tls:
        ensure_certs()
    else:
        print("⚠️  TLS disabled (--http flag). Use HTTPS in production.")

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          ICU Biosignal Simulator                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  URL:        {scheme}://{args.host}:{port}")
    print(f"║  Swagger UI: {scheme}://localhost:{port}/docs")
    print(f"║  Health:     {scheme}://localhost:{port}/api/v1/health")
    print(f"║  Patient:    {scheme}://localhost:{port}/api/v1/patient")
    print(f"║  Vitals SSE: {scheme}://localhost:{port}/api/v1/vitals/stream")
    print(f"║  Encryption: {'TLS + AES-256-GCM' if use_tls else 'AES-256-GCM only (no TLS)'}")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Press CTRL+C to stop the simulator.")
    print()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", args.host,
        "--port", str(port),
        "--log-level", "info",
    ]
    if use_tls:
        cmd += ["--ssl-keyfile", str(KEY_FILE), "--ssl-certfile", str(CERT_FILE)]
    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Simulator stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
