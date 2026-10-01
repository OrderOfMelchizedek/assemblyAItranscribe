#!/usr/bin/env python3
"""Compatibility entry point using the current streaming implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live import main as live_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # The original command selected BlackHole; explicit device options win.
    if not any(arg == "--device" or arg.startswith("--device=") for arg in args):
        args = ["--device", "BlackHole", *args]
    return live_main(args)


if __name__ == "__main__":
    main()
