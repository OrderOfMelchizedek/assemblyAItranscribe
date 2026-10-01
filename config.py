"""Load configuration from the environment or the project-local .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit environment configuration wins and lets offline tests avoid .env reads.
if not (os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("API_KEY")):
    load_dotenv(Path(__file__).with_name(".env"))

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("API_KEY")
