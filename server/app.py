"""
FastAPI server exposing HRHiringEnv over HTTP (OpenEnv).

Run locally:
  uv run server
  python -m server.app --port 8000

The ``main()`` function is the console entry point (see ``if __name__ == "__main__"``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path (src layout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openenv.core.env_server.http_server import create_app  # noqa: E402

from src.env import HRHiringEnv  # noqa: E402
from src.models import HRAction, HRObservation  # noqa: E402

app = create_app(
    HRHiringEnv,
    HRAction,
    HRObservation,
    env_name="hr-hiring-agent",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
    # openenv validate looks for a literal main() reference:
    # main()
