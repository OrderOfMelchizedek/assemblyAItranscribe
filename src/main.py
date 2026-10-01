#!/usr/bin/env python3
"""Compatibility entry point for the original file/URL/folder command."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcribe import main as transcribe_main


def main(argv=None):
    # Retain the original output directory, filename suffix, and serial default.
    return transcribe_main(
        argv,
        default_output_dir=os.path.join(os.getcwd(), "transcripts"),
        output_suffix="_transcription",
        default_jobs=1,
    )


if __name__ == "__main__":
    main()
