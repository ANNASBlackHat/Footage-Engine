"""Submit a footage-search job to the queue (thin wrapper).

Usage:
    python scripts/submit_job.py --query "container ship aerial" --wait
    python scripts/submit_job.py --task search_script_beat --beat "..." --rerank --wait
    python scripts/submit_job.py --task fine_localize_clip --chunk-id <id> --query "..." --wait
    python scripts/submit_job.py --stats

See `python scripts/submit_job.py --help` for every option.
"""

import sys

from footage_engine.worker.client import main

if __name__ == "__main__":
    sys.exit(main())
