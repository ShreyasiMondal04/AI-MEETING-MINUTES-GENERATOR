import os
import re
import csv
import json
from pathlib import Path
from typing import Union, List, Dict, Any


def format_timestamp(seconds: float) -> str:
    """Convert seconds to [HH:MM:SS] format."""
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def load_diarization_segments(diarization_input: Union[Path, str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Load diarization segments from an in-memory list, or from JSON or CSV file.
    Returns list of dicts: [{'start': float, 'end': float, 'speaker': str}]
    """
    if isinstance(diarization_input, list):
        segments = []
        for seg in diarization_input:
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": str(seg.get("speaker", "SPEAKER_00")).strip()
            })
        segments.sort(key=lambda s: s["start"])
        return segments

    diarization_path = Path(diarization_input)
    if not diarization_path.exists():
        raise FileNotFoundError(f"Diarization file not found: {diarization_path}")

    segments = []
    if diarization_path.suffix.lower() == ".json":
        with open(diarization_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for seg in data:
                segments.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": str(seg["speaker"]).strip()
                })
    elif diarization_path.suffix.lower() == ".csv":
        with open(diarization_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            delimiter = "\t" if "\t" in sample else ","
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                segments.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": str(row["speaker"]).strip()
                })
    else:
        raise ValueError(f"Unsupported diarization format: {diarization_path.suffix}")

    segments.sort(key=lambda s: s["start"])
    return segments


def load_whisper_transcript(transcript_input: Union[Path, str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Load Whisper transcript segments from in-memory dict (Whisper transcribe result),
    in-memory list of segments, or from TXT/JSON file.
    """
    if isinstance(transcript_input, dict):
        raw_segs = transcript_input.get("segments", [])
        segments = []
        for item in raw_segs:
            segments.append({
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": str(item.get("text", "")).strip()
            })
        segments.sort(key=lambda s: s["start"])
        return segments

    if isinstance(transcript_input, list):
        segments = []
        for item in transcript_input:
            segments.append({
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": str(item.get("text", "")).strip()
            })
        segments.sort(key=lambda s: s["start"])
        return segments

    transcript_path = Path(transcript_input)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    segments = []
    if transcript_path.suffix.lower() == ".json":
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_segs = data.get("segments", data) if isinstance(data, dict) else data
            for item in raw_segs:
                segments.append({
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "text": str(item.get("text", "")).strip()
                })
    else:
        pattern = re.compile(r"\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)")
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    start_sec = float(match.group(1))
                    end_sec = float(match.group(2))
                    text = match.group(3).strip()
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text
                    })

    segments.sort(key=lambda s: s["start"])
    return segments


def align_transcript_with_diarization(
    transcript_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Align Whisper transcript segments with speaker diarization segments using
    an interval overlap-based algorithm.

    Handles:
      - Overlapping speech (identifies secondary speaker)
      - Missing speaker assignments (large temporal gaps tagged as 'Unknown')
      - Empty text segments (preserved cleanly)
      - Zero-duration and point timestamps
      - Mismatched time ranges

    Returns:
        aligned_segments: list of dicts with integrated timestamps, speakers, and text
        speaker_mapping: dict mapping raw speaker IDs to clean labels ('Speaker 1', 'Speaker 2')
    """
    # 1. Build chronological speaker mapping (Speaker 1, Speaker 2, ...)
    speaker_order = []
    for d in diarization_segments:
        spk = d.get("speaker", "SPEAKER_00")
        if spk not in speaker_order and spk != "UNKNOWN":
            speaker_order.append(spk)

    speaker_mapping = {
        spk: f"Speaker {idx + 1}"
        for idx, spk in enumerate(speaker_order)
    }

    aligned_segments = []

    for idx, t_seg in enumerate(transcript_segments, start=1):
        t_start = float(t_seg["start"])
        t_end = float(t_seg["end"])
        t_text = str(t_seg.get("text", ""))

        if t_start < 0 or t_end < 0 or t_start > t_end:
            raise ValueError(
                f"Invalid transcript segment boundaries at segment {idx}: start={t_start}, end={t_end}"
            )

        total_duration = max(0.001, t_end - t_start)

        # Calculate time overlap with each diarization segment
        overlap_per_speaker = {}
        for d_seg in diarization_segments:
            d_start = float(d_seg["start"])
            d_end = float(d_seg["end"])
            d_spk = str(d_seg.get("speaker", "SPEAKER_00")).strip()

            overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
            if overlap > 0.0:
                overlap_per_speaker[d_spk] = overlap_per_speaker.get(d_spk, 0.0) + overlap

        secondary_spk = None
        if overlap_per_speaker:
            best_raw_speaker = max(overlap_per_speaker, key=overlap_per_speaker.get)
            overlap_duration = overlap_per_speaker[best_raw_speaker]
            confidence = min(1.0, round(overlap_duration / total_duration, 2))

            # Detect secondary overlapping speaker if >= 15% of segment duration
            for spk, dur in sorted(overlap_per_speaker.items(), key=lambda x: x[1], reverse=True):
                if spk != best_raw_speaker and (dur / total_duration) >= 0.15:
                    secondary_spk = speaker_mapping.get(spk, spk)
                    break
        else:
            # Fallback when no direct overlap exists:
            # Check temporal gap to closest diarization segment
            if diarization_segments:
                closest_seg = min(
                    diarization_segments,
                    key=lambda d: min(abs(d["start"] - t_start), abs(d["end"] - t_end))
                )
                gap = min(abs(closest_seg["start"] - t_end), abs(t_start - closest_seg["end"]))
                if gap <= 5.0:
                    best_raw_speaker = closest_seg["speaker"]
                    confidence = 0.50
                else:
                    best_raw_speaker = "UNKNOWN"
                    confidence = 0.0
            else:
                best_raw_speaker = "UNKNOWN"
                confidence = 0.0

        if best_raw_speaker == "UNKNOWN":
            clean_speaker = "Unknown"
        else:
            clean_speaker = speaker_mapping.get(best_raw_speaker, best_raw_speaker)

        timestamp_formatted = f"[{format_timestamp(t_start)} - {format_timestamp(t_end)}]"

        aligned_segments.append({
            "segment_id": idx,
            "start": round(t_start, 2),
            "end": round(t_end, 2),
            "duration": round(total_duration, 2),
            "timestamp": timestamp_formatted,
            "speaker": clean_speaker,
            "raw_speaker": best_raw_speaker,
            "secondary_speaker": secondary_spk,
            "confidence": confidence,
            "text": t_text
        })

    return aligned_segments, speaker_mapping


def save_speaker_transcript(
    aligned_segments: List[Dict[str, Any]],
    output_stem: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save the aligned speaker transcript in 3 formats: TXT, JSON, CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_file = output_dir / f"{output_stem}_speaker_transcript.txt"
    json_file = output_dir / f"{output_stem}_speaker_transcript.json"
    csv_file = output_dir / f"{output_stem}_speaker_transcript.csv"

    # 1. TXT Format
    with open(txt_file, "w", encoding="utf-8") as f:
        for seg in aligned_segments:
            f.write(f"{seg['timestamp']} {seg['speaker']}: {seg['text']}\n")

    # 2. JSON Format
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(aligned_segments, f, indent=2, ensure_ascii=False)

    # 3. CSV Format
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "start", "end", "duration", "timestamp",
            "speaker", "raw_speaker", "secondary_speaker", "confidence", "text"
        ])
        for seg in aligned_segments:
            writer.writerow([
                seg["segment_id"],
                seg["start"],
                seg["end"],
                seg["duration"],
                seg["timestamp"],
                seg["speaker"],
                seg["raw_speaker"],
                seg["secondary_speaker"],
                seg["confidence"],
                seg["text"]
            ])

    return {
        "txt": txt_file,
        "json": json_file,
        "csv": csv_file
    }


def integrate_diarization_and_transcription(
    meeting_stem: str = "meeting_01",
    diarization_file: Union[Path, str, List[Dict[str, Any]], None] = None,
    transcript_file: Union[Path, str, Dict[str, Any], List[Dict[str, Any]], None] = None,
    output_dir: Union[Path, None] = None
) -> dict:
    """
    High-level integration function that connects Whisper and pyannote diarization,
    either directly in-memory or from output files on disk.
    """
    root = find_project_root()

    # Locate Diarization File if not provided in-memory
    if diarization_file is None:
        candidates = [
            root / "data" / "diarization" / f"{meeting_stem}_diarization.json",
            root / "data" / "diarization" / f"{meeting_stem}_diarization.csv",
            root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "diarization" / f"{meeting_stem}_diarization.json",
            root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "diarization" / f"{meeting_stem}_diarization.csv",
        ]
        for c in candidates:
            if c.exists():
                diarization_file = c
                break
        else:
            raise FileNotFoundError(f"Diarization output not found for {meeting_stem}")

    # Locate Whisper Transcript File if not provided in-memory
    if transcript_file is None:
        candidates = [
            root / "data" / "transcripts" / f"{meeting_stem}.txt",
            root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "transcripts" / f"{meeting_stem}.txt",
            root / "data" / "transcripts" / f"{meeting_stem}.json",
            root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "transcripts" / f"{meeting_stem}.json",
        ]
        for c in candidates:
            if c.exists():
                transcript_file = c
                break
        else:
            raise FileNotFoundError(f"Whisper transcript not found for {meeting_stem}")

    # Define Output Directory
    if output_dir is None:
        output_dir = root / "data" / "speaker_diarized_transcripts"

    diar_segs = load_diarization_segments(diarization_file)
    trans_segs = load_whisper_transcript(transcript_file)

    aligned_segs, mapping = align_transcript_with_diarization(trans_segs, diar_segs)
    saved_paths = save_speaker_transcript(aligned_segs, meeting_stem, output_dir)

    # Sync copy to inner repository folder if it exists
    inner_output_dir = root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "speaker_diarized_transcripts"
    if (root / "AI-MEETING-MINUTES-GENERATOR").exists():
        save_speaker_transcript(aligned_segs, meeting_stem, inner_output_dir)

    return {
        "aligned_segments": aligned_segs,
        "speaker_mapping": mapping,
        "output_files": saved_paths
    }


def integrate_whisper_and_diarization(
    audio_file: Union[Path, str],
    output_stem: Union[str, None] = None,
    whisper_result: Union[Dict[str, Any], None] = None,
    diarization_segments: Union[List[Dict[str, Any]], None] = None,
    output_dir: Union[Path, None] = None
) -> dict:
    """
    End-to-end integration entry point for downstream modules (Member 3 & Member 4).
    Accepts an audio file and either uses pre-computed results or automatically
    runs/loads transcription and diarization.
    """
    audio_path = Path(audio_file)
    stem = output_stem or audio_path.stem

    # If in-memory results provided, align directly
    if whisper_result is not None and diarization_segments is not None:
        trans_segs = load_whisper_transcript(whisper_result)
        diar_segs = load_diarization_segments(diarization_segments)
        aligned_segs, mapping = align_transcript_with_diarization(trans_segs, diar_segs)
        root = find_project_root()
        target_dir = output_dir or (root / "data" / "speaker_diarized_transcripts")
        saved_paths = save_speaker_transcript(aligned_segs, stem, target_dir)
        return {
            "aligned_segments": aligned_segs,
            "speaker_mapping": mapping,
            "output_files": saved_paths
        }

    # Otherwise fall back to file-based or pipeline execution
    return integrate_diarization_and_transcription(
        meeting_stem=stem,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else "meeting_01"
    res = integrate_diarization_and_transcription(meeting_stem=stem)
    print(f"Integrated {len(res['aligned_segments'])} segments successfully for '{stem}'.")
