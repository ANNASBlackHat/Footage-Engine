"""Run the async footage-search job worker (thin wrapper).

Usage:
    uv run python scripts/run_worker.py --backend qwen --idle-exit 300
    uv run python scripts/run_worker.py --mock --dry-run
    uv run python scripts/run_worker.py --backend qwen --tasks search_footage,fine_localize_clip

See `python scripts/run_worker.py --help` for every option.
"""

import sys

from footage_engine.worker.runner import main

if __name__ == "__main__":
    sys.exit(main())
