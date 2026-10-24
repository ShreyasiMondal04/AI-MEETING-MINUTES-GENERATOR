import os
import shutil
from pathlib import Path
from typing import Dict, Any, Union, Optional
from pydub import AudioSegment


def ensure_ffmpeg_in_path() -> Optional[str]:
    """
    Ensure ffmpeg is discoverable by pydub.
    Checks system PATH, common Windows download locations, and sets environment variables.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    # Check common download locations on this machine
    possible_paths = [
        Path(r"C:\Users\HP\Downloads\ffmpeg-2026-08-27-git-a6f573a1db-essentials_build\ffmpeg-2026-08-27-git-a6f573a1db-essentials_build\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
    ]

    for p in possible_paths:
        candidate = p / "ffmpeg.exe"
        if candidate.exists():
            ffmpeg_dir = str(p)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            AudioSegment.converter = str(candidate)
            ffprobe_candidate = p / "ffprobe.exe"
            if ffprobe_candidate.exists():
                AudioSegment.ffprobe = str(ffprobe_candidate)
            return str(candidate)

    return None


# Run ffmpeg discovery on module import
ensure_ffmpeg_in_path()


def get_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract metadata including duration, channels, sample rate, and size."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio = AudioSegment.from_file(str(file_path))
    duration_sec = round(len(audio) / 1000.0, 2)
    file_size_bytes = file_path.stat().st_size

    return {
        "filename": file_path.name,
        "format": file_path.suffix.lstrip(".").lower(),
        "duration_seconds": duration_sec,
        "formatted_duration": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
        "channels": audio.channels,
        "sample_rate_hz": audio.frame_rate,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
    }


def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize audio volume to a target dBFS level."""
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)


def preprocess_audio(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    normalize: bool = True
) -> Path:
    """
    Convert any meeting recording (WAV, MP3, MPEG, M4A, FLAC, OGG)
    into a clean, normalized 16kHz mono WAV file optimized for Whisper and pyannote.
    """
    ensure_ffmpeg_in_path()
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    if output_file is None:
        # Default destination: data/preprocessed_audio/<stem>.wav
        project_root = Path(__file__).resolve().parents[2]
        output_path = project_root / "data" / "preprocessed_audio" / f"{input_path.stem}.wav"
    else:
        output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Preprocessing] Loading audio from: {input_path.name}...")
    audio = AudioSegment.from_file(str(input_path))

    # Convert to mono if multi-channel
    if audio.channels != target_channels:
        print(f"[Preprocessing] Converting {audio.channels} channels -> {target_channels} (mono)...")
        audio = audio.set_channels(target_channels)

    # Set sample rate (16 kHz is optimal for Whisper and pyannote)
    if audio.frame_rate != target_sample_rate:
        print(f"[Preprocessing] Resampling {audio.frame_rate} Hz -> {target_sample_rate} Hz...")
        audio = audio.set_frame_rate(target_sample_rate)

    # Volume normalization
    if normalize and audio.dBFS != float("-inf"):
        print(f"[Preprocessing] Normalizing volume (current: {audio.dBFS:.1f} dBFS)...")
        audio = normalize_audio(audio, target_dbfs=-20.0)

    # Export to WAV 16-bit PCM
    audio.export(str(output_path), format="wav", parameters=["-acodec", "pcm_s16le"])
    print(f"[Preprocessing] Audio successfully saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parents[2]
    raw_meeting_1 = project_root / "data" / "raw_audio" / "meeting_01.mpeg"
    raw_meeting_2 = project_root / "data" / "raw_audio" / "meeting_02.mpeg"

    target = sys.argv[1] if len(sys.argv) > 1 else str(raw_meeting_2 if raw_meeting_2.exists() else raw_meeting_1)
    print(f"Testing preprocessing on: {target}")
    out = preprocess_audio(target)
    print("Preprocessed Output:", out)
    print("Metadata:", get_audio_metadata(out))