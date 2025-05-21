import sys
import os
import assemblyai as aai
from urllib.parse import urlparse
import argparse

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def save_transcription(text, input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = f"{base_name}_transcription.txt"
    with open(output_file, 'w') as f:
        f.write(text)
    print(f"Transcription saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio with optional speaker diarization')
    parser.add_argument('input', help='Input file path or URL')
    parser.add_argument('--diarize', '-d', action='store_true', 
                      help='Enable speaker diarization')
    parser.add_argument('--speakers', '-s', type=int,
                      help='Expected number of speakers (optional)')
    
    args = parser.parse_args()

    # Replace with your API key
    aai.settings.api_key = "ec3c0a51354d439997beb46021f0e138"

    if not is_url(args.input) and not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        sys.exit(1)

    # Configure transcription settings
    config = aai.TranscriptionConfig(
        speaker_labels=args.diarize
    )

    # Add number of speakers if specified
    if args.speakers:
        config.speakers_expected = args.speakers

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(args.input, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Error: {transcript.error}")
    else:
        # If diarization is enabled, format output with speaker labels
        if args.diarize:
            formatted_text = ""
            for utterance in transcript.utterances:
                formatted_text += f"Speaker {utterance.speaker}: {utterance.text}\n"
            print(formatted_text)
            save_transcription(formatted_text, args.input)
        else:
            print(transcript.text)
            save_transcription(transcript.text, args.input)

if __name__ == "__main__":
    main()
