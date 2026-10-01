#!/usr/bin/env python3
"""Live transcription using AssemblyAI's v3 streaming WebSocket API."""

import argparse
import sys
import os
from datetime import datetime
import numpy as np
import sounddevice as sd
from assemblyai.streaming.v3.client import StreamingClient
from assemblyai.streaming.v3.models import (
    BeginEvent,
    SpeakerRevisionEvent,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from config import ASSEMBLYAI_API_KEY

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTIONS_DIR = os.path.join(SCRIPT_DIR, "transcriptions")

# Sample rate for real-time transcription (16kHz recommended for speech)
SAMPLE_RATE = 16000
REALTIME_SPEECH_MODEL = "universal-3-6-pro"
SPEAKER_REVISION_INTERVAL_MS = 300000


class RevisionCompatibleClient(StreamingClient):
    """Accept revision words without metadata required by SDK 1.6.1."""

    @classmethod
    def _parse_message(cls, data):
        if data.get("type") == "SpeakerRevision":
            # Revisions only carry text, timestamps, and speaker assignments.
            # The SDK reuses its Turn word schema, which also requires these
            # two fields. These defaults are only for parsing; our revision
            # handler copies speaker labels alone, never confidence values.
            data = dict(data, revisions=[
                dict(revision, words=[
                    dict({"confidence": 0.0, "word_is_final": True}, **word)
                    for word in revision.get("words", [])
                ])
                for revision in data.get("revisions", [])
            ])
        return super()._parse_message(data)


def list_devices():
    """List all available audio input devices."""
    print("Available audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")
            print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
    print("-" * 60)


def find_device(device_name):
    """Find a device by name (partial match)."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0 and device_name.lower() in device['name'].lower():
            return i, device
    return None, None


class LiveTranscriber:
    """Handle live transcription with AssemblyAI's v3 streaming API."""

    def __init__(self, speaker_labels=False, max_speakers=None):
        self.client = None
        self.turns_by_order = {}
        self.current_partial = ""
        self.speaker_labels = speaker_labels
        self.max_speakers = max_speakers

    def format_turn(self, event: TurnEvent):
        """Return turn text with a speaker prefix when diarization is enabled."""
        speaker = getattr(event, "speaker_label", None)
        if self.speaker_labels and speaker:
            return f"Speaker {speaker}: {event.transcript}"
        return event.transcript

    def on_begin(self, client, event: BeginEvent):
        print(f"Connected to AssemblyAI (session: {event.id})")
        print("Listening... Press Ctrl+C to stop.\n")

    def on_turn(self, client, event: TurnEvent):
        if not event.transcript:
            return

        if event.end_of_turn:
            # Final transcript - print on new line and save
            turn_text = self.format_turn(event)
            print(f"\r{' ' * (len(self.current_partial) + 10)}\r", end="")
            print(f"{turn_text}")
            self.turns_by_order[event.turn_order] = event.model_copy(deep=True)
            self.current_partial = ""
        else:
            # Partial transcript - update same line
            self.current_partial = self.format_turn(event)
            print(f"\r[...] {self.current_partial}", end="", flush=True)

    def on_speaker_revision(self, client, event: SpeakerRevisionEvent):
        """Apply incremental corrections without changing transcript text."""
        for revision in event.revisions:
            turn = self.turns_by_order.get(revision.turn_order)
            if turn is None:
                continue
            turn.speaker_label = revision.speaker_label
            for word, revised_word in zip(turn.words, revision.words):
                word.speaker = revised_word.speaker

    def on_error(self, client, error: StreamingError):
        print(f"\nError: {error}")

    def on_termination(self, client, event: TerminationEvent):
        print("\nSession ended.")

    def start(self, device_id):
        """Start live transcription."""
        options = StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
        )

        self.client = RevisionCompatibleClient(options)

        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.SpeakerRevision, self.on_speaker_revision)
        self.client.on(StreamingEvents.Error, self.on_error)
        self.client.on(StreamingEvents.Termination, self.on_termination)

        streaming_kwargs = {
            "sample_rate": SAMPLE_RATE,
            "speech_model": REALTIME_SPEECH_MODEL,
            "speaker_labels": self.speaker_labels,
        }
        if self.max_speakers is not None:
            streaming_kwargs["max_speakers"] = self.max_speakers
        if self.speaker_labels:
            streaming_kwargs["speaker_labels_revision_interval_ms"] = SPEAKER_REVISION_INTERVAL_MS
        params = StreamingParameters(**streaming_kwargs)
        self.client.connect(params)

        # Create audio input stream
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"\nAudio status: {status}", file=sys.stderr)
            # Convert float32 to int16 PCM
            audio_data = (indata * 32767).astype(np.int16)
            self.client.stream(audio_data.tobytes())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            device=device_id,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
        ):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                pass

        # Wait for final turns and speaker revisions before building the transcript.
        self.client.disconnect(terminate=True)

    def get_full_transcript(self):
        """Render the latest labels in chronological turn order."""
        separator = "\n" if self.speaker_labels else " "
        return separator.join(
            self.format_turn(self.turns_by_order[order])
            for order in sorted(self.turns_by_order)
        )


def save_transcript(text, output_path):
    """Save transcript to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Transcription saved to: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Live transcription using AssemblyAI's streaming API."
    )
    parser.add_argument(
        "--device",
        default="Aggregate Device",
        help="Audio device name or index (default: Aggregate Device). Use --list-devices to see options."
    )
    parser.add_argument(
        "--output", "-o",
        help="Output filename (default: auto-generated with timestamp in transcriptions/)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save transcription to file"
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--speaker-labels",
        action="store_true",
        help="Enable real-time speaker diarization and save speaker labels."
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Optional expected maximum speaker count for diarization."
    )

    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if args.max_speakers is not None:
        if not args.speaker_labels:
            parser.error("--max-speakers requires --speaker-labels")
        if not 1 <= args.max_speakers <= 10:
            parser.error("--max-speakers must be between 1 and 10")

    # Find the audio device
    device = args.device
    device_id = None
    device_info = None

    if device is not None:
        if device.isdigit():
            device_id = int(device)
            device_info = sd.query_devices(device_id)
        else:
            device_id, device_info = find_device(device)
            if device_id is None:
                print(f"Error: Device '{device}' not found.")
                print("Use --list-devices to see available devices.")
                sys.exit(1)

    if device_info:
        print(f"Audio device: {device_info['name']}")
    else:
        print("Using default input device")

    print(f"Sample rate: {SAMPLE_RATE} Hz")
    if args.speaker_labels:
        speaker_text = "enabled"
        if args.max_speakers is not None:
            speaker_text += f" (max speakers: {args.max_speakers})"
        print(f"Speaker diarization: {speaker_text}")
    print()

    # Start live transcription
    live = LiveTranscriber(
        speaker_labels=args.speaker_labels,
        max_speakers=args.max_speakers,
    )
    live.start(device_id)

    # Get full transcript
    full_text = live.get_full_transcript()

    if not full_text:
        print("\nNo transcription captured.")
        sys.exit(0)

    print(f"\n{'=' * 40}")
    print("Full transcript:")
    print(f"{'=' * 40}")
    print(full_text)

    # Save transcript
    if not args.no_save:
        if args.output:
            output_path = args.output
            if not os.path.isabs(output_path):
                output_path = os.path.join(TRANSCRIPTIONS_DIR, output_path)
        else:
            os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(TRANSCRIPTIONS_DIR, f"live_{timestamp}.txt")

        save_transcript(full_text, output_path)


if __name__ == "__main__":
    main()
