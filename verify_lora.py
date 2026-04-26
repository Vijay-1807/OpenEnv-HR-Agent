"""
Verify LoRA adapter files before running Streamlit / inference.

Usage:
  python verify_lora.py
  python verify_lora.py path/to/qwen-hr-agent-trained

Exit 0 if adapter_model.safetensors parses; exit 1 otherwise.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    path = (
        os.path.abspath(sys.argv[1])
        if len(sys.argv) > 1
        else os.path.abspath(
            os.environ.get("SENTINEL_ADAPTER_PATH", "qwen-hr-agent-trained")
        )
    )
    st_path = os.path.join(path, "adapter_model.safetensors")
    cfg_path = os.path.join(path, "adapter_config.json")

    print(f"Checking: {path}")

    if not os.path.isdir(path):
        print("FAIL: not a directory")
        return 1
    if not os.path.isfile(st_path):
        print("FAIL: adapter_model.safetensors missing")
        return 1
    if not os.path.isfile(cfg_path):
        print("WARN: adapter_config.json missing (base model unknown)")

    try:
        from safetensors import safe_open

        with safe_open(st_path, framework="pt") as f:
            keys = list(f.keys())
        print(f"OK: safetensors readable, {len(keys)} tensors")
        print("Next: streamlit run app.py  -> Agent backend should show 'llm' if GPU/CPU load succeeds.")
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        print(
            "Your file is likely truncated (OneDrive sync) or not a valid adapter export.\n"
            "Fix: re-run training save, or copy the folder from a machine with a good file,\n"
            "or upload to Hugging Face and set SENTINEL_ADAPTER_REPO=<user>/<repo>."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
