"""Launcher that sets env vars before any import, then runs the experiment."""
import os
os.environ["USE_TF"]  = "0"
os.environ["USE_JAX"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.retrieval_experiment import main
main()
