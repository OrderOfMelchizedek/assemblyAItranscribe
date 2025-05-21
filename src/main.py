import sys
import os
from dotenv import load_dotenv
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

def process_file(file_path_or_url, args):
    # Ensure aai.settings.api_key is set (it's global in the current script)
    
    if not is_url(file_path_or_url) and not os.path.exists(file_path_or_url):
        print(f"Error: The file '{file_path_or_url}' does not exist.")
        return # Use return instead of sys.exit(1) to continue with other files

    config = aai.TranscriptionConfig(speaker_labels=args.diarize)
    if args.speakers:
        config.speakers_expected = args.speakers

    transcriber = aai.Transcriber()
    print(f"Transcribing {file_path_or_url}...") # Add a print statement
    transcript = transcriber.transcribe(file_path_or_url, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Error transcribing {file_path_or_url}: {transcript.error}")
    else:
        if args.diarize:
            formatted_text = ""
            for utterance in transcript.utterances:
                formatted_text += f"Speaker {utterance.speaker}: {utterance.text}\n"
            save_transcription(formatted_text, file_path_or_url)
        else:
            save_transcription(transcript.text, file_path_or_url)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio with optional speaker diarization')
    parser.add_argument('input', nargs='+', help='Input file path(s) or URL(s) or folder(s)')
    parser.add_argument('--diarize', '-d', action='store_true', 
                      help='Enable speaker diarization')
    parser.add_argument('--speakers', '-s', type=int,
                      help='Expected number of speakers (optional)')
    
    args = parser.parse_args()

    # Get API key from environment
    load_dotenv()
    aai.settings.api_key = os.getenv("API_KEY")

    if not is_url(args.input) and not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        sys.exit(1)


    supported_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.opus'] # Define supported extensions

    for input_path in args.input:
        if is_url(input_path):
            process_file(input_path, args)
        elif os.path.isfile(input_path):
            # Optional: Check extension if you want to be strict for local files too
            _, ext = os.path.splitext(input_path)
            if ext.lower() in supported_extensions:
                process_file(input_path, args)
            else:
                print(f"Skipping non-audio file: {input_path}")
        elif os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            processed_audio_files_in_dir = 0 # Counter
            try:
                for item in os.listdir(input_path):
                    item_path = os.path.join(input_path, item)
                    if os.path.isfile(item_path):
                        _, ext = os.path.splitext(item_path)
                        if ext.lower() in supported_extensions:
                            process_file(item_path, args)
                            processed_audio_files_in_dir += 1
                        # else:
                        #     print(f"Skipping non-audio file in directory: {item_path}") # Optional: kept for now
                
                if processed_audio_files_in_dir == 0:
                    # This message will print if the directory was empty or contained no supported audio files
                    print(f"No supported audio files found in directory '{input_path}'.")

            except PermissionError:
                print(f"Error: Permission denied when trying to read directory '{input_path}'. Skipping.")
            except OSError as e:
                print(f"Error: Could not read directory '{input_path}' ({e}). Skipping.")
        else:
            print(f"Error: Input '{input_path}' is not a valid file, URL, or directory. Skipping.")

if __name__ == "__main__":
    main()

#
# Detailed Testing Guide
#
# This script processes audio files from various sources (local files, URLs, directories)
# for transcription using the AssemblyAI API. The following outlines manual testing steps
# and a suggested structure for automated tests.
#
# Manual Testing Guide:
# ---------------------
# **Prerequisites:**
#   - Ensure `assemblyai` Python package is installed.
#   - Set your AssemblyAI API key in `aai.settings.api_key` within the script,
#     or configure it via environment variables if the script is modified to support that.
#   - Have some sample audio files (e.g., .mp3, .wav) and non-audio files (.txt, .jpg).
#
# **Test Cases:**
#
# 1. Single File Processing:
#    - Test with a single local audio file path:
#      `python src/main.py path/to/your/audio.mp3`
#    - Test with a single publicly accessible URL to an audio file:
#      `python src/main.py https://example.com/audio.wav`
#    - Expected: Transcription text printed to console and saved to `audio_transcription.txt`.
#
# 2. Multiple Files Processing:
#    - Test with multiple local audio file paths:
#      `python src/main.py audio1.mp3 path/to/audio2.wav`
#    - Test with a mix of local files and URLs:
#      `python src/main.py audio1.mp3 https://example.com/audio2.wav`
#    - Expected: Each file/URL transcribed sequentially. Output files `audio1_transcription.txt`,
#      `audio2_transcription.txt` created.
#
# 3. Directory Input Processing:
#    - Setup:
#      - Create a directory, e.g., `test_audio/`.
#      - Inside `test_audio/`, place:
#        - Supported audio files (e.g., `sample1.mp3`, `sample2.wav`).
#        - Non-audio files (e.g., `notes.txt`, `image.jpg`).
#        - Optionally, create a subdirectory with audio files (current script does not recurse).
#    - Test with the directory path:
#      `python src/main.py test_audio/`
#    - Expected: Only `sample1.mp3` and `sample2.wav` are processed. Non-audio files and
#      subdirectories (if any) are skipped with messages. Transcription files created for audio files.
#    - Test with a directory containing only non-audio files:
#      `python src/main.py path/to/non_audio_dir/`
#    - Expected: Message "No supported audio files found in directory 'path/to/non_audio_dir/'."
#    - Test with an empty directory:
#      `python src/main.py path/to/empty_dir/`
#    - Expected: Message "No supported audio files found in directory 'path/to/empty_dir/'."
#
# 4. Multiple Directories and Files (Mixed Input):
#    - Test with a combination of individual files, URLs, and directories:
#      `python src/main.py audio1.mp3 test_audio/ https://example.com/audio3.wav audio4.flac`
#    - Expected: All valid inputs processed correctly.
#
# 5. Error Conditions:
#    - Invalid file path:
#      `python src/main.py non_existent_file.mp3 audio1.mp3`
#    - Expected: Error message for `non_existent_file.mp3`, script continues to process `audio1.mp3`.
#    - Invalid URL (e.g., malformed, or points to a non-audio resource or 404):
#      `python src/main.py https://example.com/not_an_audio_file.txt`
#    - Expected: Error message from AssemblyAI or URL validation, script continues if other inputs.
#    - Invalid directory path:
#      `python src/main.py non_existent_directory/`
#    - Expected: Error message "Input 'non_existent_directory/' is not a valid file, URL, or directory. Skipping."
#    - Directory with restricted read permissions (if possible to simulate, e.g., using `chmod`):
#      `chmod 000 restricted_dir/`
#      `python src/main.py restricted_dir/`
#    - Expected: Error message "Permission denied when trying to read directory 'restricted_dir/'. Skipping."
#      (Then restore permissions: `chmod 755 restricted_dir/`)
#
# Suggested Automated Test Structure (e.g., using `unittest` or `pytest`):
# --------------------------------------------------------------------
#
# 1. Setup (`setUpClass` or `session` scope fixtures):
#    - Create a temporary root directory for test files (e.g., `temp_test_workspace/`).
#    - Inside, create subdirectories like `test_audio_valid/`, `test_audio_empty/`,
#      `test_audio_mixed/`, `test_audio_non_audio_only/`.
#    - Populate these directories with dummy audio files (e.g., small, valid WAV or MP3 files
#      created programmatically or copied from a test assets folder). Use unique names.
#    - Create some dummy non-audio files (e.g., `dummy.txt`, `fake.jpg`).
#    - Mock `aai.Transcriber` and its `transcribe` method. The mock should:
#      - Allow inspection of calls (which file/URL was passed).
#      - Return a mock `Transcript` object. This object should have a `status`
#        (e.g., `aai.TranscriptStatus.completed` or `aai.TranscriptStatus.error`)
#        and `text` or `error` attributes.
#        Example mock transcript:
#        ```python
#        class MockTranscript:
#            def __init__(self, status, text=None, error=None, speaker=None, utterances=None):
#                self.status = status
#                self.text = text
#                self.error = error
#                # For diarization
#                self.utterances = utterances if utterances else []
#
#        # Successful transcription
#        mock_success_transcript = MockTranscript(status=aai.TranscriptStatus.completed, text="Test transcription.")
#        # Error transcription
#        mock_error_transcript = MockTranscript(status=aai.TranscriptStatus.error, error="API error")
#        ```
#    - Mock `save_transcription` if direct file I/O is not desired in unit tests, or to check
#      it's called with correct parameters.
#
# 2. Teardown (`tearDownClass` or `session` scope fixtures):
#    - Remove the temporary root directory and all its contents.
#
# 3. Test Case Examples (functions named `test_*`):
#
#    - `test_single_valid_local_file()`:
#      - Call `main()` or `process_file()` with a path to a dummy audio file.
#      - Assert `aai.Transcriber().transcribe` was called once with the correct file path and config.
#      - Assert `save_transcription` was called with expected text and output file name.
#
#    - `test_single_valid_url()`:
#      - Similar to above, but with a dummy URL.
#      - Assert `aai.Transcriber().transcribe` was called with the URL.
#
#    - `test_multiple_valid_local_files()`:
#      - Provide a list of paths to dummy audio files.
#      - Assert `aai.Transcriber().transcribe` was called for each file.
#
#    - `test_directory_with_audio_files()`:
#      - Provide path to `test_audio_valid/`.
#      - Assert `aai.Transcriber().transcribe` was called for each audio file in it,
#        and NOT for non-audio files.
#
#    - `test_directory_with_only_non_audio_files(capsys)`:
#      - Provide path to `test_audio_non_audio_only/`.
#      - Assert `aai.Transcriber().transcribe` was NOT called.
#      - Capture stdout/stderr (using `capsys` in pytest) and assert the
#        "No supported audio files found..." message is present.
#
#    - `test_empty_directory(capsys)`:
#      - Provide path to `test_audio_empty/`.
#      - Assert `aai.Transcriber().transcribe` was NOT called.
#      - Assert the "No supported audio files found..." message.
#
#    - `test_non_existent_file_input(capsys)`:
#      - Provide a non-existent file path.
#      - Assert `aai.Transcriber().transcribe` was NOT called for this input.
#      - Assert "Error: The file '...' does not exist." is printed.
#
#    - `test_diarization_option()`:
#      - Call with `--diarize` and a dummy audio file.
#      - Mock `transcribe` to return a `MockTranscript` with `utterances`.
#      - Assert `aai.TranscriptionConfig` was created with `speaker_labels=True`.
#      - Assert `save_transcription` receives formatted text with speaker labels.
#
#    - `test_speakers_option()`:
#      - Call with `--speakers 2` and a dummy audio file.
#      - Assert `aai.TranscriptionConfig` was created with `speakers_expected=2`.
#
# 4. Assertions & Mocking Details:
#    - Use `unittest.mock.patch` or `pytest-mock`'s `mocker` fixture.
#    - `mock_transcriber_instance.transcribe.assert_called_with(expected_path, config=expected_config)`
#    - `mock_transcriber_instance.transcribe.call_count`
#    - `mock_save_transcription.assert_called_with(expected_text, expected_output_filename)`
#
# This structure helps isolate the script's logic from actual API calls and file system
# dependencies during unit testing, making tests faster and more reliable.
#
