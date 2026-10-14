import os
import re
import csv
import json
import tempfile
import unittest
from pathlib import Path

# Import Member 2 modules
from src.diarization.integrate_transcription import (
    find_project_root,
    load_diarization_segments,
    load_whisper_transcript,
    align_transcript_with_diarization,
    save_speaker_transcript,
    integrate_diarization_and_transcription
)
from src.diarization.evaluate_der import (
    load_reference_dialogue,
    align_hypothesis_to_reference,
    find_optimal_speaker_permutation,
    evaluate_diarization,
    export_rttm_files
)
from src.diarization.speaker_diarization import (
    load_diarization_pipeline,
    diarize_audio
)


class TestSpeakerDiarizationModule(unittest.TestCase):
    """Comprehensive test suite for Member 2 Speaker Diarization Module."""

    @classmethod
    def setUpClass(cls):
        cls.root = find_project_root()
        cls.audio_path = cls.root / "data" / "preprocessed_audio" / "meeting_01.wav"
        cls.diar_json = cls.root / "data" / "diarization" / "meeting_01_diarization.json"
        cls.diar_csv = cls.root / "data" / "diarization" / "meeting_01_diarization.csv"
        cls.whisper_txt = cls.root / "data" / "transcripts" / "meeting_01.txt"
        cls.ref_txt = cls.root / "data" / "reference_minutes" / "meeting_01.txt"

    def test_01_raw_diarization_output_schema_and_timestamps(self):
        """Verify diarization JSON/CSV outputs have valid, chronological, numeric timestamps and speaker IDs."""
        self.assertTrue(self.diar_json.exists(), f"Diarization JSON not found: {self.diar_json}")
        self.assertTrue(self.diar_csv.exists(), f"Diarization CSV not found: {self.diar_csv}")

        segments = load_diarization_segments(self.diar_json)
        self.assertGreater(len(segments), 0, "Diarization output is empty!")

        # Expected duration of meeting_01.wav is ~103.73 seconds
        max_audio_duration = 105.0
        prev_start = 0.0

        for idx, seg in enumerate(segments):
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("speaker", seg)

            start = seg["start"]
            end = seg["end"]
            speaker = seg["speaker"]

            self.assertIsInstance(start, float)
            self.assertIsInstance(end, float)
            self.assertIsInstance(speaker, str)
            self.assertTrue(speaker.startswith("SPEAKER_"), f"Unexpected raw speaker format: {speaker}")

            self.assertGreaterEqual(start, 0.0, f"Negative start time at segment {idx}")
            self.assertLessEqual(start, end, f"Start time exceeds end time at segment {idx}: {start} > {end}")
            self.assertLessEqual(end, max_audio_duration, f"End time exceeds audio duration: {end} > {max_audio_duration}")

            # Check that segments are ordered chronologically
            self.assertGreaterEqual(start, prev_start, f"Segments not chronological: {start} < {prev_start}")
            prev_start = start

    def test_02_integration_deliverables_and_speaker_labels(self):
        """Verify integrated transcript outputs (TXT, JSON, CSV) and consistent Speaker 1/2 labels."""
        result = integrate_diarization_and_transcription(meeting_stem="meeting_01")
        aligned = result["aligned_segments"]
        mapping = result["speaker_mapping"]
        output_files = result["output_files"]

        self.assertGreater(len(aligned), 0)
        self.assertIn("SPEAKER_00", mapping)
        self.assertIn("SPEAKER_01", mapping)

        timestamp_pattern = re.compile(r"^\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\]$")

        for seg in aligned:
            self.assertTrue(seg["speaker"].startswith("Speaker "), f"Invalid speaker label: {seg['speaker']}")
            self.assertTrue(timestamp_pattern.match(seg["timestamp"]), f"Malformed timestamp string: {seg['timestamp']}")
            self.assertGreater(len(seg["text"].strip()), 0, "Empty segment text")
            self.assertGreaterEqual(seg["confidence"], 0.0)
            self.assertLessEqual(seg["confidence"], 1.0)

        # Check output files exist and are non-empty
        for fmt, path in output_files.items():
            self.assertTrue(path.exists(), f"Deliverable {fmt} missing: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Deliverable {fmt} is empty: {path}")

    def test_03_error_handling_invalid_paths_and_empty_files(self):
        """Verify understandable errors for invalid paths and empty audio."""
        # Non-existent audio file
        with self.assertRaises(FileNotFoundError):
            diarize_audio("data/non_existent_audio_file_12345.wav")

        # Empty audio file (0 bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                diarize_audio(tmp_path)
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_04_missing_hf_token_produces_clear_error(self):
        """Verify load_diarization_pipeline produces a descriptive error when HF_TOKEN is absent."""
        original_token = os.environ.get("HF_TOKEN")
        try:
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            with self.assertRaises(ValueError) as ctx:
                load_diarization_pipeline()
            self.assertIn("HF_TOKEN", str(ctx.exception))
        finally:
            if original_token is not None:
                os.environ["HF_TOKEN"] = original_token

    def test_05_invalid_transcript_timestamps_produce_value_error(self):
        """Verify input validation triggers on invalid or inverted segment intervals."""
        invalid_segs = [{"start": 15.0, "end": 10.0, "text": "Inverted timestamps"}]
        diar_segs = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"}]
        with self.assertRaises(ValueError) as ctx:
            align_transcript_with_diarization(invalid_segs, diar_segs)
        self.assertIn("invalid", str(ctx.exception).lower())

    def test_06_evaluation_engine_and_hungarian_mapping(self):
        """Verify evaluation engine produces valid JSON, optimal Hungarian mapping, and valid RTTMs."""
        hyp_json = self.root / "data" / "speaker_diarized_transcripts" / "meeting_01_speaker_transcript.json"
        report_path = self.root / "data" / "diarization" / "meeting_01_evaluation_report.json"

        results = evaluate_diarization(hyp_json, self.ref_txt)

        self.assertIn("turn_accuracy_percent", results)
        self.assertIn("time_accuracy_percent", results)
        self.assertIn("speaker_permutation_check", results)
        self.assertIn("mismatches", results)

        # Confirm permutation check is optimal
        perm = results["speaker_permutation_check"]
        self.assertTrue(perm["is_optimal"], "Speaker mapping permutation is not optimal!")

        # Confirm metrics are numeric and in range
        self.assertGreater(results["turn_accuracy_percent"], 0.0)
        self.assertLessEqual(results["turn_accuracy_percent"], 100.0)
        self.assertGreater(results["time_accuracy_percent"], 0.0)
        self.assertLessEqual(results["time_accuracy_percent"], 100.0)

        # Check RTTM export
        rttm_dir = self.root / "data" / "diarization"
        ref_rttm, hyp_rttm = export_rttm_files(results["aligned_segments"], "meeting_01", rttm_dir)
        self.assertTrue(ref_rttm.exists() and ref_rttm.stat().st_size > 0)
        self.assertTrue(hyp_rttm.exists() and hyp_rttm.stat().st_size > 0)

        # Verify evaluation report is valid JSON on disk
        self.assertTrue(report_path.exists())
        with open(report_path, "r", encoding="utf-8") as f:
            disk_report = json.load(f)
        self.assertIn("turn_accuracy_percent", disk_report)


class TestDeterministicWhisperDiarizationIntegration(unittest.TestCase):
    """
    Deterministic tests using small in-memory fixtures.
    Does NOT require downloading any models or external network access.
    """

    def test_deterministic_clean_alternation_and_invariance(self):
        """Verify clean alternation, chronological mapping, and preservation of text and timestamps."""
        whisper_fixture = [
            {"start": 0.0, "end": 5.0, "text": "Good morning everyone."},
            {"start": 5.0, "end": 10.0, "text": "Hello, let's start the review."}
        ]
        diarization_fixture = [
            {"start": 0.4, "end": 4.8, "speaker": "SPEAKER_00"},
            {"start": 5.2, "end": 9.9, "speaker": "SPEAKER_01"}
        ]

        aligned, mapping = align_transcript_with_diarization(whisper_fixture, diarization_fixture)

        self.assertEqual(len(aligned), 2)
        # Check mapping
        self.assertEqual(mapping["SPEAKER_00"], "Speaker 1")
        self.assertEqual(mapping["SPEAKER_01"], "Speaker 2")

        # Check segment 1
        self.assertEqual(aligned[0]["speaker"], "Speaker 1")
        self.assertEqual(aligned[0]["raw_speaker"], "SPEAKER_00")
        self.assertEqual(aligned[0]["start"], 0.0)
        self.assertEqual(aligned[0]["end"], 5.0)
        self.assertEqual(aligned[0]["text"], "Good morning everyone.")  # Preserved untouched

        # Check segment 2
        self.assertEqual(aligned[1]["speaker"], "Speaker 2")
        self.assertEqual(aligned[1]["raw_speaker"], "SPEAKER_01")
        self.assertEqual(aligned[1]["start"], 5.0)
        self.assertEqual(aligned[1]["end"], 10.0)
        self.assertEqual(aligned[1]["text"], "Hello, let's start the review.")  # Preserved untouched

    def test_deterministic_overlapping_speech_secondary_speaker(self):
        """Verify that multi-speaker overlap identifies dominant speaker and records secondary speaker."""
        whisper_fixture = [
            {"start": 10.0, "end": 20.0, "text": "Can I add something to that point?"}
        ]
        diarization_fixture = [
            {"start": 10.0, "end": 18.0, "speaker": "SPEAKER_00"},  # 8.0s overlap (dominant)
            {"start": 14.0, "end": 19.0, "speaker": "SPEAKER_01"}   # 5.0s overlap (50% of segment, secondary)
        ]

        aligned, mapping = align_transcript_with_diarization(whisper_fixture, diarization_fixture)

        self.assertEqual(len(aligned), 1)
        seg = aligned[0]
        self.assertEqual(seg["speaker"], "Speaker 1")
        self.assertEqual(seg["raw_speaker"], "SPEAKER_00")
        self.assertEqual(seg["secondary_speaker"], "Speaker 2")
        self.assertGreaterEqual(seg["confidence"], 0.8)

    def test_deterministic_missing_speaker_and_large_gap(self):
        """Verify that segments occurring in large unassigned gaps are tagged as 'Unknown' with 0 confidence."""
        whisper_fixture = [
            {"start": 50.0, "end": 55.0, "text": "This occurs far after any diarized speech."}
        ]
        # Diarization only has speech from 0 to 10s (gap is 40s)
        diarization_fixture = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}
        ]

        aligned, mapping = align_transcript_with_diarization(whisper_fixture, diarization_fixture)

        self.assertEqual(len(aligned), 1)
        seg = aligned[0]
        self.assertEqual(seg["speaker"], "Unknown")
        self.assertEqual(seg["raw_speaker"], "UNKNOWN")
        self.assertEqual(seg["confidence"], 0.0)
        self.assertEqual(seg["text"], "This occurs far after any diarized speech.")

    def test_deterministic_empty_text_and_point_timestamps(self):
        """Verify handling of empty text and zero-duration point timestamps without errors."""
        whisper_fixture = [
            {"start": 2.0, "end": 2.0, "text": ""},  # Zero-duration empty segment
            {"start": 3.0, "end": 6.0, "text": "   "}  # Whitespace only
        ]
        diarization_fixture = [
            {"start": 1.0, "end": 7.0, "speaker": "SPEAKER_00"}
        ]

        aligned, mapping = align_transcript_with_diarization(whisper_fixture, diarization_fixture)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0]["start"], 2.0)
        self.assertEqual(aligned[0]["end"], 2.0)
        self.assertEqual(aligned[0]["text"], "")
        self.assertEqual(aligned[1]["text"], "   ")  # Original whitespace string preserved untouched

    def test_deterministic_in_memory_whisper_dict_integration(self):
        """Verify in-memory Whisper dictionary input (e.g. directly from model.transcribe())."""
        whisper_dict_result = {
            "text": "Full concatenated transcript text here.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.5, "text": "First in-memory sentence."},
                {"id": 1, "start": 4.0, "end": 8.0, "text": "Second in-memory sentence."}
            ]
        }
        diarization_list = [
            {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"}
        ]

        trans_segs = load_whisper_transcript(whisper_dict_result)
        diar_segs = load_diarization_segments(diarization_list)
        aligned, mapping = align_transcript_with_diarization(trans_segs, diar_segs)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0]["text"], "First in-memory sentence.")
        self.assertEqual(aligned[0]["speaker"], "Speaker 1")
        self.assertEqual(aligned[1]["text"], "Second in-memory sentence.")
        self.assertEqual(aligned[1]["speaker"], "Speaker 2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
