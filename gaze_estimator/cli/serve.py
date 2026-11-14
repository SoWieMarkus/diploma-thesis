"""Start the API (and the built web app) for viewing and labeling processed videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from ..api.app import create_app
from ..core.paths import DEFAULT_DATABASE
from .common import setup_logging


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE, help="SQLite database file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    setup_logging()

    app = create_app(args.db)
    print(f"Open http://{args.host}:{args.port} (API docs: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
