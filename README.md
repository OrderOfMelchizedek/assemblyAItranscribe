# AssemblyAI Transcription Tools

Python scripts for transcribing audio using AssemblyAI.

## Setup

Create and activate a virtual environment (or activate your existing one):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Scripts

### transcribe.py

Transcribe audio files, HTTP(S) audio URLs, or folders using AssemblyAI.

The async script uses `universal-3-5-pro` first, then falls back to `universal-3-pro` and `universal-2`.

**Basic usage:**
```bash
python transcribe.py recording.wav
```

Output is saved to `transcriptions/<filename>.txt`.

**Multiple files at once:**
```bash
python transcribe.py recording1.wav recording2.m4a recording3.mp3
python transcribe.py /path/to/recordings/*.m4a --jobs 2 --output-dir transcripts
python transcribe.py /path/to/recordings https://example.com/audio.mp3
```

Folders are scanned without recursion for supported audio extensions. Individual files and HTTP(S) URLs can be mixed with folders. URL query parameters are excluded from output filenames. Unreadable or empty folders and missing inputs are reported as failures, while other inputs continue.

Up to three files are transcribed simultaneously by default. Use `--jobs` (or `-j`) to change the limit; `--jobs 1` processes files sequentially. Each file gets its own `.txt` output, and speaker/language options apply to every input. If a file fails, the remaining files continue, and the command exits with a nonzero status after finishing. Inputs with the same output filename are rejected before transcription starts to prevent collisions.

**Custom output path:**
```bash
python transcribe.py recording.wav -o custom_output.txt
```

`-o` is available for a single input only. Use `--output-dir` for multiple inputs. Existing output files are replaced, as with single-file transcription.

**Useful options:**
| Option | Description |
|--------|-------------|
| `--disfluencies` | Preserve filler words and false starts for verbatim transcription |
| `--speaker-labels`, `--diarize`, `-d` | Enable speaker diarization and save speaker-labeled utterances |
| `--speakers`, `-s` | Expected speaker count; automatically enables diarization |
| `--speaker-names` | Identify diarized speakers by comma-separated names, such as `Sarah,Marcus` |
| `--speaker-roles` | Identify diarized speakers by comma-separated roles, such as `advisor,student` |
| `--language-detection` | Detect the dominant spoken language automatically |

**Speaker examples:**
```bash
# Generic diarization labels
python transcribe.py recording.wav --speaker-labels

# Speaker Identification by name
python transcribe.py recording.wav --speaker-names "Sarah,Marcus"

# Speaker Identification by role
python transcribe.py recording.wav --speaker-roles "advisor,student"
```

`--speaker-names` and `--speaker-roles` automatically enable speaker diarization. Without either option, transcription behavior is unchanged.

**Supported formats:** mp3, wav, m4a, flac, ogg, mp4, webm, aac, and more.

### live.py

Live transcription using AssemblyAI's real-time WebSocket API. Streams audio directly and displays transcription as you speak.

The realtime script uses `universal-3-6-pro`. Agent-context carryover is aimed at voice-agent apps that know what the agent just asked; this microphone transcription script does not need that extra plumbing.

With `--speaker-labels`, the script requests speaker-label corrections approximately every five minutes of audio and applies final corrections before saving on Ctrl+C. The final transcript uses the latest labels, with one turn per line; previously printed live lines retain their original labels. If the connection drops before final corrections arrive, the latest received labels are retained. AssemblyAI SDK 1.6.1 is the tested version for this workflow.

**Basic usage:**
```bash
python live.py
```

Press Ctrl+C to stop. Transcription is automatically saved to `transcriptions/live_YYYY-MM-DD_HH-MM-SS.txt`.

**Options:**
| Option | Description |
|--------|-------------|
| `--device` | Device name or index (default: Aggregate Device) |
| `-o, --output` | Custom output filename |
| `--no-save` | Don't save transcription to file |
| `-l, --list-devices` | List available input devices |
| `--speaker-labels` | Enable real-time speaker diarization and save speaker labels |
| `--max-speakers` | Optional expected maximum speaker count for diarization |

**Examples:**
```bash
# Use BlackHole to capture system audio
python live.py --device "BlackHole 2ch"

# Custom output filename
python live.py -o meeting_notes.txt

# Don't save to file
python live.py --no-save

# Add real-time speaker labels
python live.py --speaker-labels --max-speakers 2
```

## Configuration

Create a `.env` file with your API key:
```bash
ASSEMBLYAI_API_KEY=your_api_key_here
```

The configuration also accepts the legacy `API_KEY` environment variable. `ASSEMBLYAI_API_KEY` takes precedence. The project-local `.env` is loaded only when neither variable is already set.

## Compatibility with the original GitHub commands

Both original entry points use the current implementations:

```bash
python src/main.py recording.wav https://example.com/audio.mp3 /path/to/recordings -d -s 2
python src/transcript_stream.py
```

`src/main.py` retains the original `transcripts/<filename>_transcription.txt` output relative to the current working directory and processes one input at a time by default. It also accepts the new options such as `--jobs`, `--output-dir`, and speaker identification.

`src/transcript_stream.py` keeps BlackHole as its default device and accepts the `live.py` options. It now saves through the current implementation in `transcriptions/live_YYYY-MM-DD_HH-MM-SS.txt`; use `-o` for an explicit output path. PyAudio is no longer required.

## Offline tests

```bash
ASSEMBLYAI_API_KEY=offline-test-key python -m unittest discover -s tests -v
```

Tests mock transcription services and microphone access. They make no paid API calls and use no personal recordings or transcripts.

## Local data

Keep credentials in `.env` and outputs in `transcriptions/` or `transcripts/`. These directories, recordings, email PDFs, and local history files are ignored by Git. Output saved elsewhere may need an additional ignore rule. Commit only reviewed source, documentation, and tests.
