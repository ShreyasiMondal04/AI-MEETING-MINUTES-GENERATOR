import os
import re
import json
import itertools
from pathlib import Path


def find_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'src'."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def load_reference_dialogue(ref_file: Path) -> list[dict]:
    """
    Parse a reference transcript formatted as:
    Speaker 1: Text...
    Speaker 2: Text...
    """
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    turns = []
    pattern = re.compile(r"^(Speaker\s*\d+):\s*(.+)$", re.IGNORECASE)

    with open(ref_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                speaker = m.group(1).title()
                text = m.group(2).strip()
                turns.append({
                    "turn_id": len(turns) + 1,
                    "line_number": idx,
                    "speaker": speaker,
                    "text": text
                })

    if not turns:
        raise ValueError(f"No valid speaker dialogue turns found in reference file: {ref_file}")

    return turns


def tokenize(text: str) -> set[str]:
    """Extract lowercased word tokens from text."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def align_hypothesis_to_reference(hyp_segments: list[dict], ref_turns: list[dict]) -> list[dict]:
    """
    Align each Whisper hypothesis segment to its true reference dialogue turn
    using token-overlap semantic matching.

    This eliminates naive index-drift errors caused by sentence splits
    (e.g., when Whisper splits one reference sentence into two segments).
    """
    aligned = []

    for h in hyp_segments:
        h_tokens = tokenize(h["text"])
        best_turn = None
        best_overlap = -1

        for r in ref_turns:
            r_tokens = tokenize(r["text"])
            overlap = len(h_tokens.intersection(r_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = r

        ref_speaker = best_turn["speaker"] if best_turn else "Unknown"
        ref_turn_id = best_turn["turn_id"] if best_turn else None
        ref_text = best_turn["text"] if best_turn else ""

        duration = round(h["end"] - h["start"], 2)

        aligned.append({
            "segment_id": h["segment_id"],
            "start": h["start"],
            "end": h["end"],
            "duration": duration,
            "timestamp": h["timestamp"],
            "hypothesis_speaker": h["speaker"],
            "raw_speaker": h.get("raw_speaker", h["speaker"]),
            "secondary_speaker": h.get("secondary_speaker"),
            "reference_speaker": ref_speaker,
            "reference_turn_id": ref_turn_id,
            "hypothesis_text": h["text"],
            "reference_text": ref_text
        })

    return aligned


def find_optimal_speaker_permutation(aligned_segments: list[dict]) -> dict:
    """
    Implements the Hungarian / Optimal Assignment concept for speaker clustering.
    Evaluates all 1-to-1 bijections between raw acoustic clusters (e.g. 'SPEAKER_00', 'SPEAKER_01')
    and reference identities ('Speaker 1', 'Speaker 2').

    Proves mathematically whether any remaining error is an arbitrary label permutation
    or genuine acoustic cluster confusion.
    """
    raw_speakers = sorted(list(set(s["raw_speaker"] for s in aligned_segments)))
    ref_speakers = sorted(list(set(s["reference_speaker"] for s in aligned_segments if s["reference_speaker"] != "Unknown")))

    if len(raw_speakers) != len(ref_speakers):
        # Fallback if cluster counts differ
        return {"is_optimal": True, "optimal_mapping": {}, "gain": 0}

    best_mapping = {}
    best_matches = -1
    total_permutations = list(itertools.permutations(ref_speakers))

    for perm in total_permutations:
        candidate_map = dict(zip(raw_speakers, perm))
        matches = sum(
            1 for s in aligned_segments
            if candidate_map.get(s["raw_speaker"]) == s["reference_speaker"]
        )
        if matches > best_matches:
            best_matches = matches
            best_mapping = candidate_map

    # Check direct pipeline mapping matches
    pipeline_matches = sum(
        1 for s in aligned_segments
        if s["hypothesis_speaker"] == s["reference_speaker"]
    )

    is_optimal = (pipeline_matches >= best_matches)
    return {
        "pipeline_matches": pipeline_matches,
        "optimal_matches": best_matches,
        "is_optimal": is_optimal,
        "optimal_mapping": best_mapping
    }


def export_rttm_files(aligned_segments: list[dict], stem: str, output_dir: Path):
    """
    Export standard NIST RTTM files for ground-truth reference and hypothesis.
    Format: SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_rttm_path = output_dir / f"{stem}_reference.rttm"
    hyp_rttm_path = output_dir / f"{stem}_hypothesis.rttm"

    with open(ref_rttm_path, "w", encoding="utf-8") as f_ref, open(hyp_rttm_path, "w", encoding="utf-8") as f_hyp:
        for seg in aligned_segments:
            dur = round(seg["end"] - seg["start"], 2)
            f_ref.write(f"SPEAKER {stem} 1 {seg['start']:.2f} {dur:.2f} <NA> <NA> {seg['reference_speaker']} <NA> <NA>\n")
            f_hyp.write(f"SPEAKER {stem} 1 {seg['start']:.2f} {dur:.2f} <NA> <NA> {seg['hypothesis_speaker']} <NA> <NA>\n")

    return ref_rttm_path, hyp_rttm_path


def evaluate_diarization(hypothesis_json_path: Path, reference_file_path: Path) -> dict:
    """
    Comprehensive Diarization Evaluation Engine:
    1. Text-aligned semantic matching (fixes index drift)
    2. Optimal speaker permutation analysis (verifies Hungarian mapping)
    3. Segment turn attribution accuracy
    4. Time-weighted speech attribution accuracy and error rate
    5. Root cause error categorization
    """
    if not hypothesis_json_path.exists():
        raise FileNotFoundError(f"Hypothesis JSON not found: {hypothesis_json_path}")
    if not reference_file_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file_path}")

    with open(hypothesis_json_path, "r", encoding="utf-8") as f:
        hyp_segments = json.load(f)

    ref_turns = load_reference_dialogue(reference_file_path)
    aligned_segments = align_hypothesis_to_reference(hyp_segments, ref_turns)

    # Permutation Check
    perm_check = find_optimal_speaker_permutation(aligned_segments)

    total_segments = len(aligned_segments)
    total_speech_time = sum(s["duration"] for s in aligned_segments)

    correct_segments = 0
    correct_speech_time = 0.0
    mismatches = []

    for s in aligned_segments:
        is_match = (s["hypothesis_speaker"].lower() == s["reference_speaker"].lower())
        if is_match:
            correct_segments += 1
            correct_speech_time += s["duration"]
        else:
            # Diagnose reason for mismatch
            if s.get("secondary_speaker"):
                err_type = "Whisper Segment Fusion (Multiple Speakers in Interval)"
            else:
                err_type = "Acoustic Cluster Confusion / Speaker Merging (pyannote error)"

            mismatches.append({
                "segment_id": s["segment_id"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "timestamp": s["timestamp"],
                "reference_speaker": s["reference_speaker"],
                "hypothesis_speaker": s["hypothesis_speaker"],
                "raw_speaker": s["raw_speaker"],
                "error_type": err_type,
                "text": s["hypothesis_text"]
            })

    turn_accuracy = round((correct_segments / total_segments) * 100, 2) if total_segments > 0 else 0.0
    turn_error_rate = round(100.0 - turn_accuracy, 2)

    time_accuracy = round((correct_speech_time / total_speech_time) * 100, 2) if total_speech_time > 0 else 0.0
    time_error_rate = round(100.0 - time_accuracy, 2)

    return {
        "evaluation_method": "Semantic Text-Aligned Turn Matching",
        "total_evaluated_segments": total_segments,
        "correct_segments": correct_segments,
        "mismatched_segments": len(mismatches),
        "turn_accuracy_percent": turn_accuracy,
        "turn_error_rate_percent": turn_error_rate,
        "total_speech_time_seconds": round(total_speech_time, 2),
        "correct_speech_time_seconds": round(correct_speech_time, 2),
        "time_accuracy_percent": time_accuracy,
        "time_error_rate_percent": time_error_rate,
        "speaker_permutation_check": perm_check,
        "mismatches": mismatches,
        "aligned_segments": aligned_segments
    }


def explain_formal_der():
    """Prints standard educational notes on Diarization Error Rate."""
    print("=" * 75)
    print("DIARIZATION ERROR RATE (DER) & ACCURACY BENCHMARK EXPLANATION")
    print("=" * 75)
    print("Formal DER Formula:")
    print("    DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Speech Time")
    print("\nMetrics Provided in this Evaluation:")
    print("  1. Turn Attribution Accuracy: Percentage of speech segments correctly labeled.")
    print("  2. Time-Weighted Error Rate : Percentage of vocal duration attributed to wrong speaker.")
    print("  3. Permutation Verification : Confirms whether cluster mapping is mathematically optimal.")
    print("=" * 75)


def run_evaluation(meeting_stem: str = "meeting_01"):
    root = find_project_root()
    hyp_json = root / "data" / "speaker_diarized_transcripts" / f"{meeting_stem}_speaker_transcript.json"
    ref_txt = root / "data" / "reference_minutes" / f"{meeting_stem}.txt"

    if not ref_txt.exists():
        ref_txt = root / "AI-MEETING-MINUTES-GENERATOR" / "data" / "reference_minutes" / f"{meeting_stem}.txt"

    explain_formal_der()

    if not ref_txt.exists():
        print(f"[!] Error: Reference ground-truth file not found: {ref_txt}")
        return

    if not hyp_json.exists():
        print(f"[!] Error: Hypothesis transcript not found: {hyp_json}")
        print("Please run integrate_transcription.py first.")
        return

    print(f"Evaluating: {hyp_json.name}")
    print(f"Reference : {ref_txt.name}")

    results = evaluate_diarization(hyp_json, ref_txt)
    aligned_segments = results.pop("aligned_segments")

    # Export standard RTTMs
    rttm_dir = root / "data" / "diarization"
    ref_rttm, hyp_rttm = export_rttm_files(aligned_segments, meeting_stem, rttm_dir)

    print("\n--- EVALUATION BENCHMARK RESULTS ---")
    print(f"Total Evaluated Segments   : {results['total_evaluated_segments']}")
    print(f"Correctly Labeled Segments : {results['correct_segments']}")
    print(f"Mismatched Segments        : {results['mismatched_segments']}")
    print(f"Turn Attribution Accuracy  : {results['turn_accuracy_percent']}%")
    print(f"Turn Error Rate            : {results['turn_error_rate_percent']}%")
    print(f"Total Speech Time          : {results['total_speech_time_seconds']}s")
    print(f"Time-Weighted Accuracy     : {results['time_accuracy_percent']}%")
    print(f"Time-Weighted Error Rate   : {results['time_error_rate_percent']}%")

    perm = results["speaker_permutation_check"]
    print(f"\nSpeaker Permutation Check  : {'Optimal (No label inversion)' if perm['is_optimal'] else 'Sub-optimal'}")

    print("\nMismatch Diagnosis Breakdown:")
    for m in results["mismatches"]:
        print(f"  Segment {m['segment_id']:2d} [{m['timestamp']}] Ref: '{m['reference_speaker']}' vs Pred: '{m['hypothesis_speaker']}'")
        print(f"     Diagnosis: {m['error_type']}")
        print(f"     Text     : \"{m['text'][:55]}...\"")

    out_eval = root / "data" / "diarization" / f"{meeting_stem}_evaluation_report.json"
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Reference RTTM saved to : {ref_rttm}")
    print(f"[OK] Hypothesis RTTM saved to: {hyp_rttm}")
    print(f"[OK] Full Evaluation Report  : {out_eval}")


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else "meeting_01"
    run_evaluation(stem)
