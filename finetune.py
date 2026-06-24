#!/usr/bin/env python3
"""Fine-tune Kitsune speakers from one JSON configuration (local or Colab)."""

import argparse
import json

from kitsune.finetune.config import FineTuneConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Run config JSON (not model_config.json)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate settings and print resolved paths; no downloads or training")
    args = parser.parse_args(argv)
    try:
        config = FineTuneConfig.load(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.validate_only:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    from kitsune.finetune.runner import run_finetune
    run_finetune(config)


if __name__ == "__main__":
    main()
