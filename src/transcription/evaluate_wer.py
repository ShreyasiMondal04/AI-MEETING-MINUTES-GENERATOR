import re
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from jiwer import wer, cer, process_words


def find_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def clean_reference_text(text: str) -> str:
    """Remove speaker tags like 'Speaker 1:' and normalize spacing."""
    # Remove Speaker tags
    cleaned = re.sub(r"^Speaker\s*\d+\s*:\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove timestamps like [0.00 - 6.00]
    cleaned = re.sub(r"\[\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\]", "", cleaned)
    # Remove punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def clean_hypothesis_text(text: str) -> str:
    """Remove timestamps, speaker tags, and normalize whitespace for transcription evaluation."""
    cleaned = re.sub(r"\[\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\]", "", text)
    cleaned = re.sub(r"^Speaker\s*\d+\s*:\s*", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def compute_wer(
    reference_text: str,
    hypothesis_text: str
) -> Dict[str, Any]:
    """
    Compute detailed Word Error Rate (WER) and Character Error Rate (CER)
    using jiwer.
    """
    ref_norm = clean_reference_text(reference_text)
    hyp_norm = clean_hypothesis_text(hypothesis_text)

    if not ref_norm:
        raise ValueError("Reference text is empty.")

    wer_score = wer(ref_norm, hyp_norm)
    cer_score = cer(ref_norm, hyp_norm)

    word_details = process_words(ref_norm, hyp_norm)

    ref_word_count = len(ref_norm.split())
    hyp_word_count = len(hyp_norm.split())

    return {
        "wer": round(float(wer_score), 4),
        "wer_percentage": f"{wer_score * 100:.2f}%",
        "cer": round(float(cer_score), 4),
        "cer_percentage": f"{cer_score * 100:.2f}%",
        "word_accuracy": round(max(0.0, 1.0 - wer_score), 4),
        "word_accuracy_percentage": f"{max(0.0, 1.0 - wer_score) * 100:.2f}%",
        "substitutions": word_details.substitutions,
        "deletions": word_details.deletions,
        "insertions": word_details.insertions,
        "hits": word_details.hits,
        "reference_words": ref_word_count,
        "hypothesis_words": hyp_word_count,
    }


def evaluate_meeting_wer(
    meeting_stem: str = "meeting_01",
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Evaluate WER for a given meeting comparing reference dialogue
    with Whisper generated transcript.
    """
    root = find_project_root()
    ref_path = root / "data" / "reference_minutes" / f"{meeting_stem}.txt"
    hyp_path = root / "data" / "transcripts" / f"{meeting_stem}.txt"

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hypothesis transcript not found: {hyp_path}")

    ref_content = ref_path.read_text(encoding="utf-8")
    hyp_content = hyp_path.read_text(encoding="utf-8")

    metrics = compute_wer(ref_content, hyp_content)
    metrics["meeting_stem"] = meeting_stem
    metrics["reference_file"] = str(ref_path)
    metrics["hypothesis_file"] = str(hyp_path)

    # Save evaluation report if directory provided or default
    if output_dir is None:
        output_dir = root / "data" / "evaluation"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{meeting_stem}_wer_report.json"
    report_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["report_file"] = str(report_file)

    return metrics


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else "meeting_01"
    print(f"Evaluating Word Error Rate (WER) for '{stem}'...")
    try:
        results = evaluate_meeting_wer(stem)
        print("=" * 50)
        print(f"WER Metric Results for: {stem}")
        print("=" * 50)
        print(f"Word Error Rate (WER) : {results['wer_percentage']}")
        print(f"Word Accuracy         : {results['word_accuracy_percentage']}")
        print(f"Character Error (CER) : {results['cer_percentage']}")
        print(f"Hits / Correct Words  : {results['hits']} / {results['reference_words']}")
        print(f"Substitutions         : {results['substitutions']}")
        print(f"Deletions             : {results['deletions']}")
        print(f"Insertions            : {results['insertions']}")
        print("=" * 50)
    except Exception as e:
        print(f"Evaluation error: {e}")