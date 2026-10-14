"""
Master End-to-End Runner for Member 2: Speaker Diarization Module.
Pipeline:
  1. Audio Verification & Preprocessing check
  2. Speaker Diarization (pyannote.audio)
  3. Whisper Transcript Integration (Interval Overlap Alignment)
  4. Speaker-Labelled Transcript Generation (TXT, JSON, CSV)
  5. Diarization Evaluation (DER / Turn Accuracy vs Reference)
  6. Deliverable Validation & Handoff Summary
"""

import sys
import json
from pathlib import Path

# Ensure module imports work regardless of execution directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

from src.diarization.integrate_transcription import (
    find_project_root,
    integrate_diarization_and_transcription
)
from src.diarization.evaluate_der import run_evaluation


def run_full_pipeline(meeting_stem: str = "meeting_01", skip_raw_diarization_if_exists: bool = True):
    root = find_project_root()
    print("=" * 70)
    print("AI-MEETING-MINUTES GENERATOR — MEMBER 2 DIARIZATION PIPELINE")
    print(f"Target Recording: {meeting_stem}")
    print(f"Project Root    : {root}")
    print("=" * 70)

    # 1. Locate Audio
    audio_candidates = [
        root / "data" / "preprocessed_audio" / f"{meeting_stem}.wav",
        root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "preprocessed_audio" / f"{meeting_stem}.wav",
        root / "data" / "raw_audio" / f"{meeting_stem}.wav",
    ]
    audio_file = None
    for c in audio_candidates:
        if c.exists():
            audio_file = c
            break

    if audio_file:
        print(f"[OK] Audio file verified: {audio_file.name} ({audio_file.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[!] Warning: Audio file for {meeting_stem} not found in preprocessed_audio. Using existing diarization output.")

    # 2. Check Diarization Outputs
    diar_json = root / "data" / "diarization" / f"{meeting_stem}_diarization.json"
    diar_csv = root / "data" / "diarization" / f"{meeting_stem}_diarization.csv"

    if (not diar_json.exists() or not diar_csv.exists()) or not skip_raw_diarization_if_exists:
        print("\n--- STEP 1: Running pyannote Speaker Diarization ---")
        from src.diarization.speaker_diarization import diarize_audio
        if not audio_file:
            raise FileNotFoundError(f"Cannot run diarization: Audio file for {meeting_stem} not found.")
        diarize_audio(audio_file)
    else:
        print(f"\n[OK] Diarization output found: {diar_json.name}")
        with open(diar_json, "r", encoding="utf-8") as f:
            diar_data = json.load(f)
            speakers = set(s["speaker"] for s in diar_data)
            print(f"    - Segments : {len(diar_data)}")
            print(f"    - Speakers : {len(speakers)} ({', '.join(sorted(speakers))})")

    # 3. Integrate with Whisper Transcript
    print("\n--- STEP 2: Integrating with Whisper Transcript (Interval Overlap) ---")
    integration_result = integrate_diarization_and_transcription(meeting_stem=meeting_stem)
    aligned_segments = integration_result["aligned_segments"]
    outputs = integration_result["output_files"]

    # 4. Evaluate against Ground Truth Reference
    print("\n--- STEP 3: Evaluating Diarization Accuracy / DER ---")
    run_evaluation(meeting_stem=meeting_stem)

    # 5. Final Validation Checks
    print("\n--- STEP 4: Deliverable Validation Checks ---")
    txt_path = outputs["txt"]
    json_path = outputs["json"]
    csv_path = outputs["csv"]

    assert txt_path.exists() and txt_path.stat().st_size > 0, "TXT transcript missing or empty!"
    assert json_path.exists() and json_path.stat().st_size > 0, "JSON transcript missing or empty!"
    assert csv_path.exists() and csv_path.stat().st_size > 0, "CSV transcript missing or empty!"

    # Verify timestamps
    for seg in aligned_segments:
        assert seg["start"] <= seg["end"], f"Invalid timestamp order in segment {seg['segment_id']}"
        assert seg["speaker"].startswith("Speaker "), f"Invalid speaker label in segment {seg['segment_id']}"

    print("[OK] All 3 deliverable files exist, are valid, and passed data assertions!")
    print("\n" + "=" * 70)
    print("MEMBER 2 DELIVERABLES READY FOR HANDOFF:")
    print(f"1. TXT Format : {txt_path}")
    print(f"2. JSON Format: {json_path} (Ready for Member 3 & Member 4)")
    print(f"3. CSV Format : {csv_path}")
    print("=" * 70)

    # Show first 3 sample lines
    print("\nSample Preview of Output:")
    with open(txt_path, "r", encoding="utf-8") as f:
        for _ in range(3):
            print("  " + f.readline().strip())
    print("=" * 70)


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "meeting_01"
    run_full_pipeline(meeting_stem=stem)
