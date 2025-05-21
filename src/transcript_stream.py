import assemblyai as aai
import pyaudio
import asyncio
import sys
import os

# Set your AssemblyAI API key
aai.settings.api_key = "ec3c0a51354d439997beb46021f0e138"

# Global variables to store the session ID and the transcript text.
session_id = None
transcript_lines = []

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    # Only act on non-empty transcript texts.
    if not transcript.text:
        return

    # For final transcripts, save the text to our transcript_lines list.
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        transcript_lines.append(transcript.text)
        print(transcript.text, end="\r\n")
    else:
        # For interim transcripts, you might choose to display but not store.
        print(transcript.text, end="\r")

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    print("Closing Session")
    # When the session is closing, attempt to save the transcript.
    if session_id is not None:
        # Build the transcripts folder path (same directory as the script)
        transcripts_dir = os.path.join(os.getcwd(), "transcripts")
        # Create the folder if it doesn't exist.
        if not os.path.exists(transcripts_dir):
            os.makedirs(transcripts_dir)
        # Create the file path using session_id as the filename.
        file_path = os.path.join(transcripts_dir, f"{session_id}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Write each line on its own line in the file.
                f.write("\n".join(transcript_lines))
            print(f"Transcript saved to: {file_path}")
        except Exception as e:
            print("Failed to write transcript:", e)
    else:
        print("Session ID not set. Transcript not saved.")

# Function to get the device index for your BlackHole virtual input device.
def get_blackhole_device_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['name'].startswith('BlackHole'):
            return i
    return None

blackhole_index = get_blackhole_device_index()

# Create the AssemblyAI transcriber and set the event handlers.
transcriber = aai.RealtimeTranscriber(
    sample_rate=44100,
    on_data=on_data,
    on_error=on_error,
    on_open=on_open,
    on_close=on_close,
    end_utterance_silence_threshold=500
)

transcriber.connect()

# Set up the microphone stream using the BlackHole virtual device.
microphone_stream = aai.extras.MicrophoneStream(sample_rate=44100)
microphone_stream.device_index = blackhole_index
transcriber.stream(microphone_stream)

# Typically you would have an event loop or a mechanism to let the transcription run.
# For this example, you might be waiting for manual termination (e.g., via KeyboardInterrupt)
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Interrupt received. Closing transcriber.")
    transcriber.close()
