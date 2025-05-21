# AssemblyAI Transcription Tools

A collection of Python scripts for audio transcription using the AssemblyAI API.

## Scripts

### Main Transcription Script (`src/main.py`)

A command-line tool for transcribing audio files or URLs with optional speaker diarization.

#### Features:
- Transcribe audio files from local path or URL
- Optional speaker diarization
- Specify expected number of speakers
- Automatic saving of transcription results

#### Usage:
```bash
# Basic transcription
python src/main.py /path/to/audio.mp3

# Transcription with speaker diarization
python src/main.py /path/to/audio.mp3 --diarize

# Specify number of speakers
python src/main.py /path/to/audio.mp3 --diarize --speakers 3

# Transcribe from URL
python src/main.py https://example.com/audio.mp3 --diarize
```

### Real-time Transcription Script (`src/transcript_stream.py`)

A script for real-time transcription of audio from a BlackHole virtual audio device input.

#### Features:
- Real-time audio transcription
- Automatic saving of transcriptions with session ID as filename
- Uses BlackHole virtual audio device to capture system audio

#### Usage:
```bash
# Run the real-time transcription script
python src/transcript_stream.py
```

## Requirements
- Python 3.6+
- AssemblyAI API key
- PyAudio (for real-time transcription)
- BlackHole virtual audio device (for system audio capture)

## Setup
1. Install required packages:
   ```bash
   pip install assemblyai pyaudio
   ```

2. Replace the API key in the scripts with your own AssemblyAI API key.

3. For real-time transcription, ensure BlackHole virtual audio device is installed and configured.