import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch
import whisper


def find_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


_LOADED_MODELS: Dict[str, Any] = {}


def get_whisper_model(model_size: str = "base"):
    """
    Load and cache Whisper model in memory.
    Automatically detects CUDA availability and selects CPU/CUDA device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = f"{model_size}_{device}"

    if cache_key not in _LOADED_MODELS:
        print(f"[Whisper] Loading '{model_size}' model onto {device.upper()}...")
        _LOADED_MODELS[cache_key] = whisper.load_model(model_size, device=device)
        print(f"[Whisper] Model '{model_size}' loaded successfully.")

    return _LOADED_MODELS[cache_key], device


def transcribe_audio(
    audio_file: Union[str, Path],
    model_size: str = "base",
    language: str = "en",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Transcribe speech to text using OpenAI Whisper.
    Returns structured result with full text and timestamped segments.
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        root = find_project_root()
        candidate = root / audio_file
        if candidate.exists():
            audio_path = candidate
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model, device = get_whisper_model(model_size=model_size)
    fp16 = device == "cuda"

    print(f"[Whisper] Transcribing '{audio_path.name}' (language={language}, fp16={fp16})...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        temperature=temperature,
        fp16=fp16,
        verbose=False
    )

    print(f"[Whisper] Transcription completed. Detected {len(result.get('segments', []))} segments.")
    return result


def save_transcript(
    result: Dict[str, Any],
    output_txt_file: Union[str, Path],
    output_json_file: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    Save transcription results in standard timestamped TXT format
    and structured JSON format.
    """
    txt_path = Path(output_txt_file)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    segments = result.get("segments", [])

    with txt_path.open("w", encoding="utf-8") as file:
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            file.write(f"[{start:.2f} - {end:.2f}] {text}\n")

    saved = {"txt": txt_path}

    if output_json_file:
        json_path = Path(output_json_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "text": result.get("text", "").strip(),
                "segments": [
                    {
                        "id": s.get("id"),
                        "start": round(s["start"], 2),
                        "end": round(s["end"], 2),
                        "text": s["text"].strip()
                    }
                    for s in segments
                ],
                "language": result.get("language", "en")
            }, f, indent=2, ensure_ascii=False)
        saved["json"] = json_path

    return saved


if __name__ == "__main__":
    import sys
    root = find_project_root()
    default_audio = root / "data" / "preprocessed_audio" / "meeting_01.wav"
    audio_arg = sys.argv[1] if len(sys.argv) > 1 else str(default_audio)
    stem = Path(audio_arg).stem

    out_txt = root / "data" / "transcripts" / f"{stem}.txt"
    out_json = root / "data" / "transcripts" / f"{stem}.json"

    res = transcribe_audio(audio_arg, model_size="base")
    saved = save_transcript(res, out_txt, out_json)

    print(f"[Whisper] Results saved to:\n - {saved['txt']}\n - {saved.get('json')}")