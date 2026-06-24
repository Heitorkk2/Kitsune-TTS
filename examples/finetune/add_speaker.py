#!/usr/bin/env python3
"""Compatibility entry point for appending a speaker embedding."""

from pathlib import Path
import sys

# Keep existing commands/imports working; implementation lives in the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kitsune.finetune.speakers import _generator_state, expand_speaker, main


if __name__ == "__main__":
    main()
