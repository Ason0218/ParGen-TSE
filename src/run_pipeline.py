"""Convenience wrapper: `python src/run_pipeline.py ...` forwards to `pargen.run_pipeline`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pargen.run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
