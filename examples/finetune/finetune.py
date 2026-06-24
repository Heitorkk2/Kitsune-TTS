#!/usr/bin/env python3
"""Legacy embedding-only CLI. For full fine-tuning use root finetune.py --config."""

from pathlib import Path
import sys

# Keep existing commands/imports working; implementation lives in the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kitsune.finetune.embedding_only import main


if __name__ == "__main__":
    main()
