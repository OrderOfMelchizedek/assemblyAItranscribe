import sys
import os
import assemblyai as aai
from urllib.parse import urlparse

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
    if len(sys.argv) != 2:
        print("Usage: python main.py <file_path_or_url>")
        sys.exit(1)

    # Replace with your API key
    aai.settings.api_key = "ec3c0a51354d439997beb46021f0e138"

    file_path_or_url = sys.argv[1]

    if not is_url(file_path_or_url) and not os.path.exists(file_path_or_url):
        print(f"Error: The file '{file_path_or_url}' does not exist.")
        sys.exit(1)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path_or_url)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Error: {transcript.error}")
    else:
        print(transcript.text)
        save_transcription(transcript.text, file_path_or_url)

if __name__ == "__main__":
    main()