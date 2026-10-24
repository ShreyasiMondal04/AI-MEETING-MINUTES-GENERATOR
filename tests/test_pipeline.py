import os
import sys
import json
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.backend.app import app
from src.transcription.audio_preprocessing import get_audio_metadata, ensure_ffmpeg_in_path
from src.transcription.evaluate_wer import compute_wer
from src.summarization.meeting_summarizer import (
    load_transcript,
    generate_minutes,
    save_minutes,
    clean_filler_words,
    infer_meeting_title
)
from src.evaluation.evaluate_all import evaluate_summarization_metrics


class TestFullPipeline(unittest.TestCase):
    """End-to-end integration and unit tests for the complete AI Meeting Minutes Generator."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.root = PROJECT_ROOT
        cls.meeting_01_audio = cls.root / "data" / "preprocessed_audio" / "meeting_01.wav"
        cls.meeting_02_audio = cls.root / "data" / "preprocessed_audio" / "meeting_02.wav"
        cls.meeting_01_json = cls.root / "data" / "speaker_diarized_transcripts" / "meeting_01_speaker_transcript.json"

    def test_01_ffmpeg_detection_and_audio_metadata(self):
        """Verify ffmpeg discovery and audio metadata extraction."""
        ffmpeg = ensure_ffmpeg_in_path()
        self.assertIsNotNone(ffmpeg, "FFmpeg was not discovered!")

        if self.meeting_01_audio.exists():
            meta = get_audio_metadata(self.meeting_01_audio)
            self.assertEqual(meta["format"], "wav")
            self.assertEqual(meta["channels"], 1)
            self.assertEqual(meta["sample_rate_hz"], 16000)
            self.assertGreater(meta["duration_seconds"], 50)

    def test_02_wer_computation_engine(self):
        """Verify Word Error Rate (WER) computation."""
        ref = "Good morning everyone. Let's discuss the project deadline."
        hyp = "Good morning everyone. Let us discuss the project deadline."
        res = compute_wer(ref, hyp)

        self.assertIn("wer", res)
        self.assertIn("word_accuracy", res)
        self.assertGreater(res["word_accuracy"], 0.8)

    def test_03_filler_words_cleaning(self):
        """Verify removal of conversational filler words."""
        dirty = "Um, you know, we basically completed the module yesterday, like, without bugs."
        clean = clean_filler_words(dirty)
        self.assertNotIn("you know", clean.lower())
        self.assertNotIn("basically", clean.lower())
        self.assertIn("completed the module yesterday", clean)

    def test_04_nlp_minutes_generation_and_title_inference(self):
        """Verify meeting minutes synthesis, decisions, and action items extraction."""
        self.assertTrue(self.meeting_01_json.exists(), f"Missing {self.meeting_01_json}")
        segments = load_transcript(self.meeting_01_json)

        title = infer_meeting_title(segments)
        self.assertIn("AI Meeting Minutes", title)

        minutes = generate_minutes(segments)
        self.assertIn("summary", minutes)
        self.assertIn("key_points", minutes)
        self.assertIn("decisions", minutes)
        self.assertIn("action_items", minutes)
        self.assertIn("speaker_analytics", minutes)

        self.assertGreater(len(minutes["decisions"]), 0)
        self.assertGreater(len(minutes["action_items"]), 0)

        # Check action item structure
        first_action = minutes["action_items"][0]
        self.assertIn("task", first_action)
        self.assertIn("responsible_person", first_action)
        self.assertIn("deadline", first_action)
        self.assertIn("priority", first_action)

    def test_05_document_export_all_formats(self):
        """Verify PDF, Word (.docx), Markdown, Plain Text, and JSON generation."""
        segments = load_transcript(self.meeting_01_json)
        minutes = generate_minutes(segments)

        out_dir = self.root / "data" / "meeting_minutes"
        json_file, txt_file = save_minutes(minutes, output_dir=out_dir, meeting_stem="test_pipeline_run")

        pdf_file = out_dir / "test_pipeline_run_minutes.pdf"
        docx_file = out_dir / "test_pipeline_run_minutes.docx"
        md_file = out_dir / "test_pipeline_run_minutes.md"

        self.assertTrue(pdf_file.exists(), "PDF export file missing!")
        self.assertTrue(docx_file.exists(), "DOCX export file missing!")
        self.assertTrue(md_file.exists(), "Markdown export file missing!")
        self.assertTrue(txt_file.exists(), "Text export file missing!")
        self.assertTrue(json_file.exists(), "JSON export file missing!")

        self.assertGreater(pdf_file.stat().st_size, 1000)
        self.assertGreater(docx_file.stat().st_size, 1000)

        # Clean up temporary test files
        for f in [pdf_file, docx_file, md_file, txt_file, json_file]:
            if f.exists():
                f.unlink()

    def test_06_summarization_metrics_rouge_bleu(self):
        """Verify ROUGE-1/2/L and BLEU scoring engine."""
        ref = "We agreed to complete the frontend design by Friday."
        hyp = "The team will complete the frontend design by Friday."
        scores = evaluate_summarization_metrics(ref, hyp)

        self.assertIn("rouge1", scores)
        self.assertIn("rouge2", scores)
        self.assertIn("rougeL", scores)
        self.assertIn("bleu", scores)
        self.assertGreater(scores["rouge1"]["f1"], 0.5)

    def test_07_backend_api_health_and_samples(self):
        """Verify FastAPI /health and /api/samples endpoints."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

        res_samples = self.client.get("/api/samples")
        self.assertEqual(res_samples.status_code, 200)
        samples_data = res_samples.json()
        self.assertGreaterEqual(len(samples_data["samples"]), 2)

    def test_08_backend_api_process_sample(self):
        """Verify FastAPI /api/process-sample/meeting_01 endpoint."""
        res = self.client.post("/api/process-sample/meeting_01")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("minutes", data)
        self.assertIn("speaker_transcript", data)

    def test_09_backend_api_exports_download(self):
        """Verify downloading PDF, Word, Markdown, Text, and JSON exports."""
        for fmt in ["pdf", "docx", "txt", "json", "md"]:
            res = self.client.get(f"/api/export/meeting_01/{fmt}")
            self.assertEqual(res.status_code, 200, f"Export download failed for {fmt}")
            self.assertGreater(len(res.content), 50)

    def test_10_backend_api_evaluate_endpoint(self):
        """Verify /api/evaluate/meeting_01 endpoint returns WER, DER, ROUGE, BLEU."""
        res = self.client.get("/api/evaluate/meeting_01")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["has_reference"])
        self.assertIn("speech_to_text", data["metrics"])
        self.assertIn("speaker_diarization", data["metrics"])
        self.assertIn("summarization", data["metrics"])


if __name__ == "__main__":
    unittest.main()
