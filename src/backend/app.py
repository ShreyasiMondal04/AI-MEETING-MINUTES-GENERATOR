from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

from src.transcription.whisper_transcription import (
    transcribe_audio,
    save_transcript
)

app = FastAPI(
    title="AI Meeting Minutes Generator",
    description="Backend API for the AI Meeting Minutes Generator",
    version="1.0.0"
)

UPLOAD_FOLDER = Path("data/uploads")
TRANSCRIPT_FOLDER = Path("data/transcripts")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "AI Meeting Minutes Generator Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/transcript")
def get_transcript():

    transcript_path = TRANSCRIPT_FOLDER / "meeting_01.txt"

    if not transcript_path.exists():
        return {
            "status": "error",
            "message": "Transcript file not found"
        }

    transcript = transcript_path.read_text(encoding="utf-8")

    return {
        "status": "success",
        "transcript": transcript
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Save uploaded audio
    audio_path = UPLOAD_FOLDER / file.filename

    with audio_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run Member 1's Whisper function
    result = transcribe_audio(str(audio_path))

    # Save transcript
    transcript_path = TRANSCRIPT_FOLDER / "meeting_01.txt"

    save_transcript(
        result,
        str(transcript_path)
    )

    return {
        "status": "success",
        "message": "Audio transcribed successfully",
        "filename": file.filename,
        "transcript_file": str(transcript_path)
    }