import os
import json
from pathlib import Path

import torch
import soundfile as sf
from pyannote.audio import Pipeline


# Hugging Face model
MODEL_NAME = "pyannote/speaker-diarization-community-1"


def find_project_root():
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def load_diarization_pipeline():
    """Load the pyannote speaker diarization pipeline."""

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set your Hugging Face token first."
        )

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=hf_token
    )

    if pipeline is None:
        raise RuntimeError("Failed to load the diarization pipeline.")

    return pipeline


def diarize_audio(audio_path, pipeline=None, output_dir=None, num_speakers=2):
    """Perform speaker diarization on an audio file."""

    audio_path = Path(audio_path)

    # Resolve audio file path if not directly found
    if not audio_path.exists():
        root = find_project_root()
        candidate = root / audio_path
        if candidate.exists():
            audio_path = candidate
        else:
            alt_candidates = [
                root / "AI-MEETING-MINUTES-GENERATOR" / audio_path,
                root / "data" / "preprocessed_audio" / audio_path.name,
                root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "preprocessed_audio" / audio_path.name,
            ]
            for alt in alt_candidates:
                if alt.exists():
                    audio_path = alt
                    break
            else:
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if pipeline is None:
        pipeline = load_diarization_pipeline()

    waveform, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    if len(waveform) == 0:
        raise ValueError(f"Audio file contains no readable frames: {audio_path}")

    duration_sec = len(waveform) / sample_rate
    if duration_sec < 0.5:
        raise ValueError(f"Audio duration too short for diarization ({duration_sec:.2f}s): {audio_path}")

    # pyannote expects (channels, time) tensor
    waveform = torch.from_numpy(waveform.T)
    if waveform.shape[0] > 1:
        # Convert multi-channel to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    output = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate
        },
        num_speakers=num_speakers
    )

    diarization = output.speaker_diarization

    segments = []

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):
        segments.append(
            {
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            }
        )

    if output_dir is None:
        root = find_project_root()
        output_dir = root / "data" / "diarization"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_json = output_dir / f"{audio_path.stem}_diarization.json"
    output_csv = output_dir / f"{audio_path.stem}_diarization.csv"

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            segments,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        output_csv,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("start\tend\tspeaker\n")
        for seg in segments:
            f.write(f"{seg['start']}\t{seg['end']}\t{seg['speaker']}\n")

    print(f"Diarization JSON saved to: {output_json}")
    print(f"Diarization CSV saved to: {output_csv}")

    return segments


if __name__ == "__main__":
    import sys

    default_audio = find_project_root() / "data" / "preprocessed_audio" / "meeting_01.wav"
    audio_file = sys.argv[1] if len(sys.argv) > 1 else str(default_audio)

    print(f"Running speaker diarization on: {audio_file}")
    results = diarize_audio(audio_file)
    print(f"Completed! Generated {len(results)} speaker segments.")