import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.transcription.evaluate_wer import evaluate_meeting_wer
from src.diarization.evaluate_der import run_evaluation as run_der_evaluation


def evaluate_summarization_metrics(
    reference_text: str,
    hypothesis_summary: str
) -> Dict[str, Any]:
    """
    Calculate ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores
    comparing generated meeting summary with reference discussion points.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, hypothesis_summary)

    smooth = SmoothingFunction().method1
    ref_tokens = [reference_text.lower().split()]
    hyp_tokens = hypothesis_summary.lower().split()

    bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth)

    return {
        "rouge1": {
            "precision": round(scores["rouge1"].precision, 4),
            "recall": round(scores["rouge1"].recall, 4),
            "f1": round(scores["rouge1"].fmeasure, 4),
            "f1_percentage": f"{scores['rouge1'].fmeasure * 100:.2f}%"
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision, 4),
            "recall": round(scores["rouge2"].recall, 4),
            "f1": round(scores["rouge2"].fmeasure, 4),
            "f1_percentage": f"{scores['rouge2'].fmeasure * 100:.2f}%"
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision, 4),
            "recall": round(scores["rougeL"].recall, 4),
            "f1": round(scores["rougeL"].fmeasure, 4),
            "f1_percentage": f"{scores['rougeL'].fmeasure * 100:.2f}%"
        },
        "bleu": {
            "score": round(float(bleu), 4),
            "percentage": f"{float(bleu) * 100:.2f}%"
        }
    }


def evaluate_meeting_full(meeting_stem: str = "meeting_01") -> Dict[str, Any]:
    """
    Run comprehensive evaluation for a meeting across all 3 AI stages:
      1. Speech-to-Text (WER / CER / Accuracy)
      2. Speaker Diarization (DER / Turn Accuracy / NIST RTTM)
      3. Summarization & NLP (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)
    """
    root = PROJECT_ROOT
    ref_file = root / "data" / "reference_minutes" / f"{meeting_stem}.txt"
    minutes_file = root / "data" / "meeting_minutes" / f"{meeting_stem}_minutes.json"

    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    print(f"\n[Evaluation] Evaluating Speech-to-Text (WER) for '{meeting_stem}'...")
    wer_results = evaluate_meeting_wer(meeting_stem=meeting_stem)

    print(f"[Evaluation] Evaluating Speaker Diarization (DER) for '{meeting_stem}'...")
    der_results = run_der_evaluation(meeting_stem=meeting_stem)

    print(f"[Evaluation] Evaluating Summarization (ROUGE / BLEU) for '{meeting_stem}'...")
    if not minutes_file.exists():
        raise FileNotFoundError(f"Minutes file not found: {minutes_file}. Generate minutes first.")

    with minutes_file.open("r", encoding="utf-8") as f:
        minutes_data = json.load(f)

    ref_text = ref_file.read_text(encoding="utf-8")
    generated_summary = minutes_data.get("summary", "")
    key_points_text = " ".join(minutes_data.get("key_points", []))
    combined_hyp = f"{generated_summary} {key_points_text}"

    nlp_metrics = evaluate_summarization_metrics(
        reference_text=ref_text,
        hypothesis_summary=combined_hyp
    )

    full_report = {
        "meeting_stem": meeting_stem,
        "evaluation_timestamp": str(Path(__file__).stat().st_mtime),
        "speech_to_text": {
            "metric": "Word Error Rate (WER)",
            "wer": wer_results["wer"],
            "wer_percentage": wer_results["wer_percentage"],
            "word_accuracy": wer_results["word_accuracy_percentage"],
            "character_error_rate": wer_results["cer_percentage"],
            "hits": wer_results["hits"],
            "substitutions": wer_results["substitutions"],
            "deletions": wer_results["deletions"],
            "insertions": wer_results["insertions"],
        },
        "speaker_diarization": {
            "metric": "Diarization Error Rate (DER)",
            "turn_attribution_accuracy": f"{der_results.get('turn_accuracy_percent', 0):.2f}%",
            "time_weighted_accuracy": f"{der_results.get('time_accuracy_percent', 0):.2f}%",
            "time_weighted_error_rate": f"{der_results.get('time_error_rate_percent', 0):.2f}%",
            "total_segments": der_results.get("total_evaluated_segments", 0),
            "correct_turns": der_results.get("correct_segments", 0),
            "mismatched_turns": der_results.get("mismatched_segments", 0)
        },
        "summarization": {
            "metric": "ROUGE & BLEU",
            "rouge_1_f1": nlp_metrics["rouge1"]["f1_percentage"],
            "rouge_2_f1": nlp_metrics["rouge2"]["f1_percentage"],
            "rouge_l_f1": nlp_metrics["rougeL"]["f1_percentage"],
            "bleu_score": nlp_metrics["bleu"]["percentage"],
            "details": nlp_metrics
        }
    }

    # Save to data/evaluation
    out_dir = root / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{meeting_stem}_full_evaluation_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"[Evaluation] Full report saved to: {report_file}")
    return full_report


def generate_benchmark_summary() -> Dict[str, Any]:
    """Evaluate all available sample meetings and generate a master benchmark."""
    meetings = ["meeting_01", "meeting_02"]
    reports = {}

    for m in meetings:
        try:
            reports[m] = evaluate_meeting_full(m)
        except Exception as e:
            print(f"[Benchmark Warning] Could not evaluate {m}: {e}")

    master_path = PROJECT_ROOT / "data" / "evaluation" / "master_benchmark_report.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print(f"\n[Benchmark] Master benchmark saved to: {master_path}")
    return reports


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else None
    if stem:
        rep = evaluate_meeting_full(stem)
        print("\n" + "=" * 60)
        print(f"COMPLETE EVALUATION REPORT FOR: {stem}")
        print("=" * 60)
        print(f"Speech-to-Text  | WER: {rep['speech_to_text']['wer_percentage']} | Word Acc: {rep['speech_to_text']['word_accuracy']}")
        print(f"Diarization     | Turn Acc: {rep['speaker_diarization']['turn_attribution_accuracy']} | Time Acc: {rep['speaker_diarization']['time_weighted_accuracy']}")
        print(f"Summarization   | ROUGE-1: {rep['summarization']['rouge_1_f1']} | ROUGE-L: {rep['summarization']['rouge_l_f1']} | BLEU: {rep['summarization']['bleu_score']}")
        print("=" * 60)
    else:
        generate_benchmark_summary()
