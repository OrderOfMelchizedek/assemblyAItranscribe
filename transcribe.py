#!/usr/bin/env python3
"""Transcribe audio files using AssemblyAI and save separate .txt files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from urllib.parse import unquote, urlsplit
import assemblyai as aai
from config import ASSEMBLYAI_API_KEY

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTIONS_DIR = os.path.join(SCRIPT_DIR, "transcriptions")
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".mp4", ".webm",
}
MAX_KNOWN_VALUE_LENGTH = 35
ASYNC_SPEECH_MODELS = ["universal-3-5-pro", "universal-3-pro", "universal-2"]


def is_url(value):
    """Recognize HTTP(S) audio URLs without making a network request."""
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def expand_inputs(inputs):
    """Expand folders nonrecursively, retaining valid inputs after an error."""
    audio_paths = []
    failures = 0
    for value in inputs:
        if is_url(value) or os.path.isfile(value):
            audio_paths.append(value)
        elif os.path.isdir(value):
            try:
                files = [
                    os.path.join(value, name)
                    for name in sorted(os.listdir(value))
                    if os.path.splitext(name)[1].lower() in SUPPORTED_AUDIO_EXTENSIONS
                    and os.path.isfile(os.path.join(value, name))
                ]
            except OSError:
                print(f"Error: Cannot read directory: {value}", file=sys.stderr)
                failures += 1
                continue
            if files:
                audio_paths.extend(files)
            else:
                print(f"Error: No supported audio files in directory: {value}", file=sys.stderr)
                failures += 1
        else:
            print(f"Error: Input not found or unsupported URL: {value}", file=sys.stderr)
            failures += 1
    return audio_paths, failures


def output_stem(audio_path):
    """Use a URL's filename without query parameters or encoded path segments."""
    path = unquote(urlsplit(audio_path).path) if is_url(audio_path) else audio_path
    return os.path.splitext(os.path.basename(path))[0] or "audio"


def parse_known_values(value):
    """Return a clean list of comma-separated speaker names or roles."""
    if not value:
        return None

    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def validate_known_values(parser, option_name, raw_value):
    """Validate comma-separated Speaker Identification values."""
    if raw_value is None:
        return None

    values = parse_known_values(raw_value)
    if not values:
        parser.error(f"{option_name} requires at least one speaker value")

    too_long = [value for value in values if len(value) > MAX_KNOWN_VALUE_LENGTH]
    if too_long:
        parser.error(
            f"{option_name} values must be {MAX_KNOWN_VALUE_LENGTH} characters or fewer: "
            f"{', '.join(too_long)}"
        )

    return values


def build_speaker_identification(speaker_type, known_values):
    """Build an AssemblyAI Speaker Identification request."""
    if not known_values:
        return None

    return aai.SpeechUnderstandingRequest(
        request=aai.SpeechUnderstandingFeatureRequests(
            speaker_identification=aai.SpeakerIdentificationRequest(
                speaker_type=speaker_type,
                known_values=known_values,
            )
        )
    )


def get_speaker_mapping(transcript):
    """Return Speaker Identification's label-to-name mapping when available."""
    speech_understanding = getattr(transcript, "speech_understanding", None)
    if not speech_understanding:
        return {}

    response = getattr(speech_understanding, "response", None)
    if not response:
        return {}

    speaker_identification = getattr(response, "speaker_identification", None)
    if not speaker_identification:
        return {}

    return getattr(speaker_identification, "mapping", None) or {}


def format_speaker(speaker, speaker_mapping, speaker_identification_enabled):
    """Return a speaker prefix for diarized or identified speaker output."""
    if not speaker:
        return None

    identified_speaker = speaker_mapping.get(speaker, speaker)
    if speaker_identification_enabled:
        return identified_speaker
    return f"Speaker {identified_speaker}"


def format_transcript(transcript, use_speaker_labels, speaker_identification_enabled=False):
    """Return transcript text, optionally split by speaker utterance."""
    if not use_speaker_labels:
        return transcript.text or ""

    utterances = getattr(transcript, "utterances", None)
    if not utterances:
        return transcript.text or ""

    speaker_mapping = get_speaker_mapping(transcript)
    lines = []
    for utterance in utterances:
        speaker = getattr(utterance, "speaker", None)
        text = getattr(utterance, "text", "")
        speaker_text = format_speaker(
            speaker,
            speaker_mapping,
            speaker_identification_enabled,
        )
        if speaker_text:
            lines.append(f"{speaker_text}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def transcribe_audio(
    audio_path,
    include_disfluencies=False,
    speaker_labels=False,
    speaker_names=None,
    speaker_roles=None,
    language_detection=False,
    speakers_expected=None,
):
    """Transcribe an audio file and return the transcription text."""
    if not is_url(audio_path) and not os.path.isfile(audio_path):
        print(f"Error: File not found: {audio_path}")
        raise FileNotFoundError(audio_path)

    aai.settings.api_key = ASSEMBLYAI_API_KEY

    print(f"Transcribing: {audio_path}")
    print("This may take a moment...")

    speech_understanding = None
    if speaker_names:
        speech_understanding = build_speaker_identification(
            aai.SpeakerType.name,
            speaker_names,
        )
    elif speaker_roles:
        speech_understanding = build_speaker_identification(
            aai.SpeakerType.role,
            speaker_roles,
        )

    speaker_identification_enabled = speech_understanding is not None
    speaker_labels = speaker_labels or speaker_identification_enabled or speakers_expected is not None

    config = aai.TranscriptionConfig(
        speech_models=ASYNC_SPEECH_MODELS,
        disfluencies=include_disfluencies,
        speaker_labels=speaker_labels,
        speakers_expected=speakers_expected,
        language_detection=language_detection,
        speech_understanding=speech_understanding,
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription error: {transcript.error}")

    return format_transcript(
        transcript,
        speaker_labels,
        speaker_identification_enabled=speaker_identification_enabled,
    )


def main(argv=None, *, default_output_dir=TRANSCRIPTIONS_DIR, output_suffix="", default_jobs=3):
    parser = argparse.ArgumentParser(
        description="Transcribe audio files, HTTP(S) URLs, or folders using AssemblyAI."
    )
    parser.add_argument(
        "audio_file",
        nargs="+",
        help="Audio file paths, HTTP(S) URLs, or folders (folders are scanned without recursion)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output .txt file path (single input only)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {default_output_dir})"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=default_jobs,
        help="Maximum simultaneous transcriptions (default: %(default)s)"
    )
    parser.add_argument(
        "--disfluencies",
        action="store_true",
        help="Preserve filler words and false starts for verbatim transcription."
    )
    parser.add_argument(
        "--speaker-labels", "--diarize", "-d",
        action="store_true",
        help="Enable speaker diarization and save speaker-labeled utterances."
    )
    parser.add_argument(
        "--speakers", "-s", type=int,
        help="Expected speaker count; also enables speaker diarization."
    )
    speaker_id_group = parser.add_mutually_exclusive_group()
    speaker_id_group.add_argument(
        "--speaker-names",
        help="Comma-separated speaker names to identify, such as 'Sarah,Marcus'. Enables speaker diarization."
    )
    speaker_id_group.add_argument(
        "--speaker-roles",
        help="Comma-separated speaker roles to identify, such as 'advisor,student'. Enables speaker diarization."
    )
    parser.add_argument(
        "--language-detection",
        action="store_true",
        help="Detect the dominant spoken language automatically."
    )

    args = parser.parse_args(argv)
    speaker_names = validate_known_values(parser, "--speaker-names", args.speaker_names)
    speaker_roles = validate_known_values(parser, "--speaker-roles", args.speaker_roles)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.speakers is not None and args.speakers < 1:
        parser.error("--speakers must be at least 1")
    if args.output and args.output_dir:
        parser.error("--output and --output-dir cannot be combined")

    audio_paths, failures = expand_inputs(args.audio_file)
    if not audio_paths:
        raise SystemExit(1)
    if args.output and len(audio_paths) != 1:
        parser.error("--output requires a single input; use --output-dir for multiple files")

    output_dir = args.output_dir or default_output_dir
    tasks = []
    destinations = set()
    for audio_path in audio_paths:
        base_name = output_stem(audio_path)
        output_path = args.output or os.path.join(output_dir, f"{base_name}{output_suffix}.txt")
        destination = os.path.realpath(output_path).casefold()
        if destination in destinations:
            parser.error(f"Multiple inputs would overwrite {output_path}; use distinct filenames")
        destinations.add(destination)
        tasks.append((audio_path, output_path))

    local_inputs = {os.path.realpath(path).casefold() for path in audio_paths if not is_url(path)}
    if destinations & local_inputs:
        parser.error("An output path would overwrite an input file; choose another output path")

    if not args.output:
        os.makedirs(output_dir, exist_ok=True)

    options = dict(
        include_disfluencies=args.disfluencies,
        speaker_labels=args.speaker_labels,
        speaker_names=speaker_names,
        speaker_roles=speaker_roles,
        language_detection=args.language_detection,
        speakers_expected=args.speakers,
    )

    input_failures = failures
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
        pending = {
            executor.submit(transcribe_audio, audio_path, **options): (audio_path, output_path)
            for audio_path, output_path in tasks
        }
        for future in as_completed(pending):
            audio_path, output_path = pending[future]
            try:
                text = future.result()
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as exc:
                failures += 1
                print(f"Error processing {audio_path}: {exc}", file=sys.stderr)
                continue

            print(f"\nTranscription saved to: {output_path}")
            print(f"\nPreview ({min(500, len(text))} chars):")
            print("-" * 40)
            print(text[:500])
            if len(text) > 500:
                print("...")

    if len(tasks) > 1:
        print(f"\nFinished: {len(tasks) - (failures - input_failures)} succeeded, {failures} failed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
