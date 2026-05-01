#!/usr/bin/env python3
"""
Live Translate — Startup Script
================================
Usage:
  python run.py                   # default host 0.0.0.0, port 8765
  python run.py --port 9000
  python run.py --reload          # hot-reload for development
  python run.py --debug           # show raw whisper-stream output for diagnosis
"""
import argparse
import logging
import os
import sys

# Ensure project root is on sys.path so "backend.*" imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description="Live Translate Server")
    parser.add_argument("--host",   default=settings.host,  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   type=int, default=settings.port, help="Port (default: 8765)")
    parser.add_argument("--reload", action="store_true",    help="Hot-reload (dev only)")
    parser.add_argument("--debug",  action="store_true",    help="Enable DEBUG logging (shows raw whisper output)")
    args = parser.parse_args()

    # Configure logging level BEFORE uvicorn touches it
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )

    settings.host = args.host
    settings.port = args.port

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          LIVE TRANSLATE — Real-time AST                  ║
╠══════════════════════════════════════════════════════════╣
║  Operator panel :  http://{args.host}:{args.port}/operator
║  Client portal  :  http://{args.host}:{args.port}/client
║  API status     :  http://{args.host}:{args.port}/api/status
╚══════════════════════════════════════════════════════════╝
{'  DEBUG MODE — raw whisper output will be shown' if args.debug else ''}
""")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        log_level="debug" if args.debug else "info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )


if __name__ == "__main__":
    main()
