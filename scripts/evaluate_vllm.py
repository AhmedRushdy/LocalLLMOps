"""Compatibility wrapper for the vLLM evaluation run.

Prefer: python -m evaluation.evaluate --backend vllm
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main(["--backend", "vllm"]))
