import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import soundfile as sf
import torch


# Hugging Face model
MODEL_NAME = "pyannote/speaker-diarization-community-1"


def find_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def load_diarization_pipeline(hf_token: Optional[str] = None):
    """
    Load the pyannote speaker diarization pipeline using HF_TOKEN.
    Raises ValueError if HF_TOKEN is absent.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set your Hugging Face token first."
        )

    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            MODEL_NAME,
            token=token
        )
        if pipeline is None:
            raise RuntimeError("Failed to load pyannote pipeline.")
        return pipeline
    except Exception as e:
        print(f"[Diarization Warning] pyannote pipeline load error: {e}")
        raise RuntimeError(f"Failed to load pyannote pipeline: {e}")



def run_acoustic_fallback_diarization(
    waveform_mono: np.ndarray,
    sample_rate: int,
    num_speakers: int = 2,
    frame_duration_sec: float = 1.5,
    hop_duration_sec: float = 0.75
) -> List[Dict[str, Any]]:
    """
    High-accuracy local acoustic clustering fallback for speaker diarization.
    Used when Hugging Face token is not provided or network is unavailable.
    
    Extracts spectral energy, zero-crossing rate, spectral centroid, and spectral roll-off
    across sliding frames, then clusters speech turns using Agglomerative Clustering.
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.signal import medfilt

    total_samples = len(waveform_mono)
    total_duration = total_samples / sample_rate

    frame_len = int(frame_duration_sec * sample_rate)
    hop_len = int(hop_duration_sec * sample_rate)

    if total_samples < frame_len:
        return [{"start": 0.0, "end": round(total_duration, 2), "speaker": "SPEAKER_00"}]

    features = []
    frame_times = []

    # Voice activity energy threshold (15% of RMS energy)
    rms_global = np.sqrt(np.mean(waveform_mono ** 2) + 1e-9)
    energy_threshold = rms_global * 0.15

    for start_idx in range(0, total_samples - frame_len + 1, hop_len):
        chunk = waveform_mono[start_idx : start_idx + frame_len]
        rms_local = np.sqrt(np.mean(chunk ** 2) + 1e-9)

        mid_sec = (start_idx + frame_len / 2.0) / sample_rate
        frame_times.append(mid_sec)

        if rms_local < energy_threshold:
            # Silence / low energy marker
            features.append(None)
            continue

        # Extract acoustic spectral descriptors
        zcr = np.mean(np.abs(np.diff(np.sign(chunk))))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
        sum_spec = np.sum(spectrum) + 1e-9

        # Spectral Centroid
        centroid = np.sum(freqs * spectrum) / sum_spec

        # Spectral Spread / Variance
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / sum_spec)

        # Spectral Roll-off (85%)
        cum_spec = np.cumsum(spectrum)
        roll_idx = np.where(cum_spec >= 0.85 * sum_spec)[0]
        rolloff = freqs[roll_idx[0]] if len(roll_idx) > 0 else freqs[-1]

        # Sub-band energy ratios (low, mid, high)
        f_low = np.sum(spectrum[freqs < 1000]) / sum_spec
        f_mid = np.sum(spectrum[(freqs >= 1000) & (freqs < 3000)]) / sum_spec
        f_high = np.sum(spectrum[freqs >= 3000]) / sum_spec

        feature_vec = [
            rms_local / (rms_global + 1e-6),
            zcr,
            centroid / 4000.0,
            spread / 2000.0,
            rolloff / 4000.0,
            f_low,
            f_mid,
            f_high
        ]
        features.append(feature_vec)

    # Filter out silence frames for clustering
    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if len(valid_indices) < num_speakers:
        return [{"start": 0.0, "end": round(total_duration, 2), "speaker": "SPEAKER_00"}]

    valid_features = np.array([features[i] for i in valid_indices])

    # Standardize features
    mean = np.mean(valid_features, axis=0)
    std = np.std(valid_features, axis=0) + 1e-6
    valid_features_norm = (valid_features - mean) / std

    # Cluster using Agglomerative Clustering
    k = min(num_speakers, len(valid_features))
    clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = clusterer.fit_predict(valid_features_norm)

    # Map labels back to timeline
    timeline_labels = [-1] * len(features)
    for idx, lab in zip(valid_indices, labels):
        timeline_labels[idx] = lab

    # Forward/backward fill small gaps
    last_known = 0
    for i in range(len(timeline_labels)):
        if timeline_labels[i] == -1:
            timeline_labels[i] = last_known
        else:
            last_known = timeline_labels[i]

    # Median filtering to smooth rapid alternating labels
    if len(timeline_labels) > 5:
        smoothed = medfilt(timeline_labels, kernel_size=5).astype(int)
    else:
        smoothed = np.array(timeline_labels)

    # Form continuous turns
    segments = []
    current_spk = smoothed[0]
    seg_start = 0.0

    for i in range(1, len(smoothed)):
        if smoothed[i] != current_spk:
            seg_end = frame_times[i]
            segments.append({
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "speaker": f"SPEAKER_{current_spk:02d}"
            })
            current_spk = smoothed[i]
            seg_start = seg_end

    segments.append({
        "start": round(seg_start, 2),
        "end": round(total_duration, 2),
        "speaker": f"SPEAKER_{current_spk:02d}"
    })

    return segments


def diarize_audio(
    audio_path: Union[str, Path],
    pipeline=None,
    output_dir: Optional[Union[str, Path]] = None,
    num_speakers: int = 2,
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on an audio file.
    Supports official pyannote.audio when HF_TOKEN is configured,
    and falls back to robust local acoustic spectral clustering automatically.
    """
    audio_path = Path(audio_path)

    # Resolve audio file path
    if not audio_path.exists():
        root = find_project_root()
        candidates = [
            root / audio_path,
            root / "AI-MEETING-MINUTES-GENERATOR" / audio_path,
            root / "data" / "preprocessed_audio" / audio_path.name,
            root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "preprocessed_audio" / audio_path.name,
            root / "data" / "raw_audio" / audio_path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                audio_path = candidate
                break
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

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

    # Convert to mono numpy array
    mono_waveform = waveform.mean(axis=1)

    segments = []
    used_pyannote = False

    # Attempt pyannote if available or pipeline passed
    if pipeline is None:
        try:
            pipeline = load_diarization_pipeline(hf_token=hf_token)
        except (ValueError, RuntimeError) as e:
            print(f"[Diarization Notice] pyannote skipped ({e}). Using acoustic fallback.")
            pipeline = None

    if pipeline is not None:
        try:
            print("[Diarization] Running pyannote speaker diarization...")
            torch_waveform = torch.from_numpy(waveform.T).mean(dim=0, keepdim=True)
            output = pipeline(
                {
                    "waveform": torch_waveform,
                    "sample_rate": sample_rate
                },
                num_speakers=num_speakers
            )
            diarization = output.speaker_diarization
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker
                })
            used_pyannote = True
            print(f"[Diarization] pyannote generated {len(segments)} speaker turns.")
        except Exception as e:
            print(f"[Diarization Warning] pyannote execution failed: {e}. Falling back to acoustic clustering.")

    if not used_pyannote or not segments:
        print("[Diarization] Running acoustic clustering diarization fallback...")
        segments = run_acoustic_fallback_diarization(
            waveform_mono=mono_waveform,
            sample_rate=sample_rate,
            num_speakers=num_speakers
        )
        print(f"[Diarization] Acoustic clustering generated {len(segments)} speaker turns.")

    # Save output files
    if output_dir is None:
        root = find_project_root()
        output_dir = root / "data" / "diarization"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"{audio_path.stem}_diarization.json"
    output_csv = output_dir / f"{audio_path.stem}_diarization.csv"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\n")
        for seg in segments:
            f.write(f"{seg['start']}\t{seg['end']}\t{seg['speaker']}\n")

    print(f"[Diarization] Outputs saved to:\n - {output_json}\n - {output_csv}")
    return segments


if __name__ == "__main__":
    import sys
    default_audio = find_project_root() / "data" / "preprocessed_audio" / "meeting_01.wav"
    audio_file = sys.argv[1] if len(sys.argv) > 1 else str(default_audio)

    print(f"Running speaker diarization on: {audio_file}")
    results = diarize_audio(audio_file)
    print(f"Completed! Generated {len(results)} speaker segments.")