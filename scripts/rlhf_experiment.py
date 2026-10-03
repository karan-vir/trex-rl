"""Run one experiment card end-to-end.   python -m scripts.rlhf_experiment how_many"""

import sys

from trex.rlhf.experiments import run_card

if __name__ == "__main__":
    run_card(sys.argv[1])
