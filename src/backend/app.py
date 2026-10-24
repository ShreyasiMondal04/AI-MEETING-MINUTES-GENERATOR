import os
import sys
import json
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transcription.audio_preprocessing import (
    preprocess_audio,
    get_audio_metadata,
    ensure_ffmpeg_in_path,
)
from src.transcription.whisper_transcription import (
    transcribe_audio,
    save_transcript,
)
from src.diarization.speaker_diarization import diarize_audio
from src.diarization.integrate_transcription import (
    integrate_whisper_and_diarization,
    integrate_diarization_and_transcription,
)
from src.summarization.meeting_summarizer import (
    load_transcript,
    generate_minutes,
    save_minutes,
)
from src.evaluation.evaluate_all import (
    evaluate_meeting_full,
    generate_benchmark_summary,
)


# --------------------------------------------------
# FastAPI Application Configuration
# --------------------------------------------------

app = FastAPI(
    title="AI Meeting Minutes Generator API",
    description="Full-stack automated meeting intelligence pipeline: Audio Preprocessing -> Whisper STT -> Speaker Diarization -> NLP Summarization -> Multi-Format Export",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder Setup
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
PREPROCESSED_FOLDER = DATA_DIR / "preprocessed_audio"
RAW_AUDIO_FOLDER = DATA_DIR / "raw_audio"
TRANSCRIPT_FOLDER = DATA_DIR / "transcripts"
DIARIZATION_FOLDER = DATA_DIR / "diarization"
SPEAKER_TRANSCRIPT_FOLDER = DATA_DIR / "speaker_diarized_transcripts"
MINUTES_FOLDER = DATA_DIR / "meeting_minutes"
EVALUATION_FOLDER = DATA_DIR / "evaluation"

for folder in [
    UPLOAD_FOLDER,
    PREPROCESSED_FOLDER,
    RAW_AUDIO_FOLDER,
    TRANSCRIPT_FOLDER,
    DIARIZATION_FOLDER,
    SPEAKER_TRANSCRIPT_FOLDER,
    MINUTES_FOLDER,
    EVALUATION_FOLDER,
]:
    folder.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --------------------------------------------------
# Models
# --------------------------------------------------

class RenameSpeakersRequest(BaseModel):
    mapping: Dict[str, str]


# --------------------------------------------------
# Web Application Root
# --------------------------------------------------

@app.get("/")
def home():
    """Serve the AI Meeting Minutes Generator frontend application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)


@app.get("/health")
def health_check():
    """System health & environment diagnostic check."""
    import torch
    ffmpeg_path = ensure_ffmpeg_in_path()
    return {
        "status": "healthy",
        "service": "AI Meeting Minutes Generator",
        "version": "2.0.0",
        "device": "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU",
        "ffmpeg_available": ffmpeg_path is not None,
        "ffmpeg_path": ffmpeg_path,
        "hf_token_set": bool(os.getenv("HF_TOKEN")),
    }


# --------------------------------------------------
# Audio Stream / Serve
# --------------------------------------------------

@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    """Serve audio files for playback in the web UI."""
    candidates = [
        PREPROCESSED_FOLDER / filename,
        RAW_AUDIO_FOLDER / filename,
        UPLOAD_FOLDER / filename,
    ]
    for c in candidates:
        if c.exists():
            media_type = "audio/wav" if c.suffix.lower() == ".wav" else "audio/mpeg"
            return FileResponse(c, media_type=media_type)

    raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")


# --------------------------------------------------
# Sample Meetings List & Processing
# --------------------------------------------------

@app.get("/api/samples")
def list_sample_meetings():
    """List preloaded meeting recordings available for instant demonstration."""
    samples = [
        {
            "id": "meeting_01",
            "title": "AI Meeting Minutes Generator Sprint",
            "description": "Discussion on Whisper setup, pyannote diarization, WER evaluation, and UI integration.",
            "audio_filename": "meeting_01.wav",
            "duration": "01:43",
            "speakers": ["Speaker 1", "Speaker 2"],
            "has_reference": True,
        },
        {
            "id": "meeting_02",
            "title": "University Website Redesign & Delivery Sync",
            "description": "Discussion on homepage design, admission pages, responsive testing, and deadline commitments.",
            "audio_filename": "meeting_02.wav",
            "duration": "01:16",
            "speakers": ["Speaker 1", "Speaker 2"],
            "has_reference": True,
        },
    ]
    return {"status": "success", "samples": samples}


@app.post("/api/process-sample/{sample_id}")
def process_sample(sample_id: str):
    """
    Execute or load full pipeline results for preloaded sample recordings.
    """
    minutes_file = MINUTES_FOLDER / f"{sample_id}_minutes.json"
    speaker_transcript_file = SPEAKER_TRANSCRIPT_FOLDER / f"{sample_id}_speaker_transcript.json"

    # If already generated, return instantly
    if minutes_file.exists() and speaker_transcript_file.exists():
        with minutes_file.open("r", encoding="utf-8") as f:
            minutes = json.load(f)
        with speaker_transcript_file.open("r", encoding="utf-8") as f:
            transcript = json.load(f)

        return {
            "status": "success",
            "meeting_id": sample_id,
            "message": f"Sample meeting '{sample_id}' loaded successfully",
            "audio_url": f"/api/audio/{sample_id}.wav",
            "minutes": minutes,
            "speaker_transcript": transcript,
            "export_urls": {
                "pdf": f"/api/export/{sample_id}/pdf",
                "docx": f"/api/export/{sample_id}/docx",
                "txt": f"/api/export/{sample_id}/txt",
                "json": f"/api/export/{sample_id}/json",
                "md": f"/api/export/{sample_id}/md",
            },
        }

    # Otherwise process from preprocessed audio or raw audio
    audio_wav = PREPROCESSED_FOLDER / f"{sample_id}.wav"
    if not audio_wav.exists():
        raw_candidates = [
            RAW_AUDIO_FOLDER / f"{sample_id}.mpeg",
            RAW_AUDIO_FOLDER / f"{sample_id}.mp3",
            RAW_AUDIO_FOLDER / f"{sample_id}.wav",
        ]
        for rc in raw_candidates:
            if rc.exists():
                audio_wav = preprocess_audio(rc, output_file=audio_wav)
                break
        else:
            raise HTTPException(status_code=404, detail=f"Audio for sample '{sample_id}' not found.")

    # Run Whisper
    whisper_res = transcribe_audio(audio_wav)
    save_transcript(
        whisper_res,
        output_txt_file=TRANSCRIPT_FOLDER / f"{sample_id}.txt",
        output_json_file=TRANSCRIPT_FOLDER / f"{sample_id}.json",
    )

    # Run Diarization
    diar_segs = diarize_audio(audio_wav)

    # Integrate
    integration = integrate_whisper_and_diarization(
        audio_file=audio_wav,
        output_stem=sample_id,
        whisper_result=whisper_res,
        diarization_segments=diar_segs,
    )
    aligned_segs = integration["aligned_segments"]

    # Generate Minutes & Exports
    minutes = generate_minutes(aligned_segs)
    save_minutes(minutes, output_dir=MINUTES_FOLDER, meeting_stem=sample_id)

    return {
        "status": "success",
        "meeting_id": sample_id,
        "message": f"Sample meeting '{sample_id}' processed successfully",
        "audio_url": f"/api/audio/{sample_id}.wav",
        "minutes": minutes,
        "speaker_transcript": aligned_segs,
        "export_urls": {
            "pdf": f"/api/export/{sample_id}/pdf",
            "docx": f"/api/export/{sample_id}/docx",
            "txt": f"/api/export/{sample_id}/txt",
            "json": f"/api/export/{sample_id}/json",
            "md": f"/api/export/{sample_id}/md",
        },
    }


# --------------------------------------------------
# Main Audio Upload & Complete Pipeline Processing
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".mpeg", ".m4a", ".flac", ".ogg", ".aac"}


@app.post("/process-meeting")
@app.post("/api/process-meeting")
async def process_meeting_upload(
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    num_speakers: int = Form(2),
    hf_token: Optional[str] = Form(None)
):
    """
    End-to-End Meeting Intelligence Pipeline:
    1. Save uploaded audio (WAV, MP3, MPEG, M4A, FLAC, OGG)
    2. Preprocess to 16kHz mono WAV (volume normalized)
    3. Transcribe with OpenAI Whisper
    4. Diarize speakers (pyannote with local acoustic fallback)
    5. Align timestamps and integrate speaker turns
    6. Extract topics, key points, decisions, and action items
    7. Generate and export multi-format minutes (PDF, DOCX, TXT, JSON, MD)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    meeting_id = f"meeting_{uuid4().hex[:10]}"
    uploaded_raw_path = UPLOAD_FOLDER / f"{meeting_id}{ext}"
    preprocessed_wav_path = PREPROCESSED_FOLDER / f"{meeting_id}.wav"

    try:
        # Step 1: Save upload
        with uploaded_raw_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if uploaded_raw_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Step 2: Preprocess Audio
        preprocess_audio(
            input_file=uploaded_raw_path,
            output_file=preprocessed_wav_path,
            normalize=True
        )

        metadata = get_audio_metadata(preprocessed_wav_path)

        # Step 3: Whisper Speech-to-Text
        whisper_result = transcribe_audio(
            audio_file=preprocessed_wav_path,
            model_size=whisper_model
        )

        transcript_txt = TRANSCRIPT_FOLDER / f"{meeting_id}.txt"
        transcript_json = TRANSCRIPT_FOLDER / f"{meeting_id}.json"
        save_transcript(
            whisper_result,
            output_txt_file=transcript_txt,
            output_json_file=transcript_json
        )

        # Step 4: Speaker Diarization
        diarization_segments = diarize_audio(
            audio_path=preprocessed_wav_path,
            num_speakers=num_speakers,
            hf_token=hf_token
        )

        # Step 5: Integration
        integration = integrate_whisper_and_diarization(
            audio_file=preprocessed_wav_path,
            output_stem=meeting_id,
            whisper_result=whisper_result,
            diarization_segments=diarization_segments
        )
        aligned_segments = integration["aligned_segments"]

        # Step 6: Meeting Minutes Generation
        minutes = generate_minutes(aligned_segments)

        # Step 7: Export to all formats (PDF, DOCX, MD, TXT, JSON)
        save_minutes(minutes, output_dir=MINUTES_FOLDER, meeting_stem=meeting_id)

        return {
            "status": "success",
            "meeting_id": meeting_id,
            "filename": file.filename,
            "metadata": metadata,
            "audio_url": f"/api/audio/{meeting_id}.wav",
            "minutes": minutes,
            "speaker_transcript": aligned_segments,
            "export_urls": {
                "pdf": f"/api/export/{meeting_id}/pdf",
                "docx": f"/api/export/{meeting_id}/docx",
                "txt": f"/api/export/{meeting_id}/txt",
                "json": f"/api/export/{meeting_id}/json",
                "md": f"/api/export/{meeting_id}/md",
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    finally:
        await file.close()


# --------------------------------------------------
# Get Meeting Minutes Details
# --------------------------------------------------

@app.get("/minutes")
@app.get("/api/minutes")
def get_default_meeting_minutes(meeting_id: str = "meeting_01"):
    """Retrieve meeting minutes JSON for a given meeting ID."""
    minutes_file = MINUTES_FOLDER / f"{meeting_id}_minutes.json"
    if not minutes_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Meeting minutes for '{meeting_id}' not found. Please process audio first."
        )

    with minutes_file.open("r", encoding="utf-8") as f:
        minutes = json.load(f)

    return {"status": "success", "meeting_id": meeting_id, "minutes": minutes}


@app.get("/api/meetings/{meeting_id}")
def get_meeting_full_data(meeting_id: str):
    """Retrieve full data including minutes, transcript, and audio URL."""
    minutes_file = MINUTES_FOLDER / f"{meeting_id}_minutes.json"
    transcript_file = SPEAKER_TRANSCRIPT_FOLDER / f"{meeting_id}_speaker_transcript.json"

    if not minutes_file.exists():
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    with minutes_file.open("r", encoding="utf-8") as f:
        minutes = json.load(f)

    transcript = []
    if transcript_file.exists():
        with transcript_file.open("r", encoding="utf-8") as f:
            transcript = json.load(f)

    return {
        "status": "success",
        "meeting_id": meeting_id,
        "audio_url": f"/api/audio/{meeting_id}.wav",
        "minutes": minutes,
        "speaker_transcript": transcript,
        "export_urls": {
            "pdf": f"/api/export/{meeting_id}/pdf",
            "docx": f"/api/export/{meeting_id}/docx",
            "txt": f"/api/export/{meeting_id}/txt",
            "json": f"/api/export/{meeting_id}/json",
            "md": f"/api/export/{meeting_id}/md",
        }
    }


# --------------------------------------------------
# Speaker Renaming
# --------------------------------------------------

@app.post("/api/rename-speakers/{meeting_id}")
def rename_speakers(meeting_id: str, request: RenameSpeakersRequest):
    """
    Dynamically rename speakers (e.g., 'Speaker 1' -> 'Annyasha')
    and re-export updated minutes in all formats.
    """
    transcript_file = SPEAKER_TRANSCRIPT_FOLDER / f"{meeting_id}_speaker_transcript.json"
    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail=f"Transcript for '{meeting_id}' not found.")

    with transcript_file.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    # Apply mapping
    mapping = request.mapping
    for seg in segments:
        current = seg.get("speaker")
        if current in mapping:
            seg["speaker"] = mapping[current]

    # Save updated transcript
    with transcript_file.open("w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # Regenerate minutes with new speaker names
    minutes = generate_minutes(segments, speaker_mapping=mapping)
    save_minutes(minutes, output_dir=MINUTES_FOLDER, meeting_stem=meeting_id)

    return {
        "status": "success",
        "message": "Speakers renamed and minutes re-exported successfully",
        "minutes": minutes,
        "speaker_transcript": segments
    }


# --------------------------------------------------
# Multi-Format Document Export Download
# --------------------------------------------------

@app.get("/api/export/{meeting_id}/{format}")
def download_export(meeting_id: str, format: str):
    """
    Download generated minutes document in PDF, DOCX, TXT, JSON, or MD format.
    """
    format = format.lower().strip()
    allowed = {"pdf", "docx", "txt", "json", "md"}
    if format not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid format '{format}'. Allowed: {', '.join(allowed)}")

    target_file = MINUTES_FOLDER / f"{meeting_id}_minutes.{format}"

    # If file doesn't exist, try generating it from JSON
    if not target_file.exists():
        json_file = MINUTES_FOLDER / f"{meeting_id}_minutes.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail=f"Meeting minutes for '{meeting_id}' not found.")
        with json_file.open("r", encoding="utf-8") as f:
            minutes = json.load(f)
        save_minutes(minutes, output_dir=MINUTES_FOLDER, meeting_stem=meeting_id)

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Failed to generate {format.upper()} export.")

    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "json": "application/json",
        "md": "text/markdown",
    }

    return FileResponse(
        target_file,
        media_type=media_types.get(format, "application/octet-stream"),
        filename=target_file.name
    )


# --------------------------------------------------
# Evaluation API (WER, DER, ROUGE, BLEU)
# --------------------------------------------------

@app.get("/api/evaluate/{meeting_id}")
def get_evaluation_metrics(meeting_id: str):
    """
    Calculate and retrieve accuracy metrics (WER, DER, ROUGE, BLEU)
    comparing pipeline outputs against ground truth reference text.
    """
    ref_path = DATA_DIR / "reference_minutes" / f"{meeting_id}.txt"
    if not ref_path.exists():
        return {
            "status": "notice",
            "message": f"No ground-truth reference file available for '{meeting_id}' in data/reference_minutes.",
            "has_reference": False
        }

    try:
        results = evaluate_meeting_full(meeting_stem=meeting_id)
        return {
            "status": "success",
            "has_reference": True,
            "metrics": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/benchmark")
def get_benchmark_report():
    """Retrieve master benchmark report for all reference meetings."""
    master_file = EVALUATION_FOLDER / "master_benchmark_report.json"
    if not master_file.exists():
        try:
            generate_benchmark_summary()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate benchmark: {e}")

    with master_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {"status": "success", "benchmark": data}
