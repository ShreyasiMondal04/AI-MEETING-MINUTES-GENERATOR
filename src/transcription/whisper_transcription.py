import whisper
from pathlib import Path

audio_file = "data/raw_audio/meeting_01.mpeg"
output_file = "data/transcripts/meeting_01.txt"

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Transcribing audio...")
result = model.transcribe(audio_file)

Path("data/transcripts").mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as file:
    file.write(result["text"])

print("\nTranscription completed successfully!")
print(f"Transcript saved to: {output_file}")