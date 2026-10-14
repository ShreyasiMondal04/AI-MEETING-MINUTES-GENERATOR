# Member 2: Speaker Diarization Module — Final Documentation & Handoff

## Module Role & Responsibilities
- **Role**: Member 2 — Speaker Diarization & Speaker-Transcript Integration
- **Input Dependencies**:
  1. Preprocessed meeting audio (`data/preprocessed_audio/<meeting>.wav`)
  2. Member 1's timestamped Whisper transcript (`data/transcripts/<meeting>.txt` or `.json`)
- **Key Outputs**:
  1. Raw speaker turns with start/end timestamps (`data/diarization/`)
  2. Speaker-labelled transcripts in TXT, JSON, and CSV (`data/speaker_diarized_transcripts/`)
  3. Diarization evaluation & error analysis (`data/diarization/<meeting>_evaluation_report.json`)

---

## Deliverables Summary

| Deliverable | Path | Description | Target Consumer |
| :--- | :--- | :--- | :--- |
| **Speaker Transcript (TXT)** | `data/speaker_diarized_transcripts/<name>_speaker_transcript.txt` | Human-readable dialogue with timestamps `[HH:MM:SS - HH:MM:SS] Speaker N: text` | **Member 3 (Summarization)** |
| **Speaker Transcript (JSON)** | `data/speaker_diarized_transcripts/<name>_speaker_transcript.json` | Structured array of segment objects with timestamps, clean speaker labels, raw IDs, confidence, and text | **Member 3 & Member 4** |
| **Speaker Transcript (CSV)** | `data/speaker_diarized_transcripts/<name>_speaker_transcript.csv` | Tabular format for spreadsheet analysis and database ingestion | **Team reference** |
| **Raw Diarization (JSON)** | `data/diarization/<name>_diarization.json` | 29 speaker turns from `pyannote.audio` | **Module pipeline** |
| **Raw Diarization (CSV)** | `data/diarization/<name>_diarization.csv` | Tab-delimited `start`, `end`, `speaker` | **Module pipeline** |
| **Evaluation Report** | `data/diarization/<name>_evaluation_report.json` | DER explanation, turn accuracy, and mismatch breakdown vs ground-truth | **Faculty / Viva / Report** |

---

## Output Data Schemas

### 1. Readable TXT Format
```text
[00:00:00 - 00:00:06] Speaker 1: Good morning, let's discuss the progress of our AI meeting mini generator project.
[00:00:06 - 00:00:13] Speaker 2: Good morning, I have completed the initial setup of the Whisper Space to Stick environment.
[00:00:13 - 00:00:16] Speaker 2: Great, were you able to test it with an audio file?
```

### 2. Structured JSON Schema (For Member 3 & Member 4)
```json
[
  {
    "segment_id": 1,
    "start": 0.0,
    "end": 6.0,
    "timestamp": "[00:00:00 - 00:00:06]",
    "speaker": "Speaker 1",
    "raw_speaker": "SPEAKER_00",
    "confidence": 0.71,
    "text": "Good morning, let's discuss the progress of our AI meeting mini generator project."
  }
]
```

### JSON Fields Explained:
- `segment_id`: 1-indexed sequential segment integer.
- `start`: Floating-point timestamp in seconds.
- `end`: Floating-point timestamp in seconds.
- `timestamp`: Formatted string `[HH:MM:SS - HH:MM:SS]`.
- `speaker`: Clean, standardized speaker label (`Speaker 1`, `Speaker 2`, etc.).
- `raw_speaker`: pyannote model identifier (`SPEAKER_00`, `SPEAKER_01`).
- `confidence`: Fraction of the Whisper segment duration overlapping with the assigned speaker.
- `text`: Transcribed speech content.

---

## How Other Members Use These Deliverables

### For Member 3 (Summarization & Action Items):
Member 3 can load the JSON file or TXT file directly into their LLM prompt or summarizer:
```python
import json

with open("data/speaker_diarized_transcripts/meeting_01_speaker_transcript.json", "r", encoding="utf-8") as f:
    transcript = json.load(f)

# Group text by speaker for individual contribution summary
speaker_contributions = {}
for seg in transcript:
    speaker = seg["speaker"]
    speaker_contributions.setdefault(speaker, []).append(seg["text"])

# Feed into summarization model / prompt
formatted_dialogue = "\n".join([f"{s['speaker']}: {s['text']}" for s in transcript])
```

### For Member 4 (Backend / FastAPI / Flask):
Member 4 can expose an endpoint `/api/transcripts/{meeting_id}` that serves the JSON payload directly to the frontend interface for synchronized audio playback and speaker filtering.

---

## Evaluation & Diarization Error Rate (DER)

- **Formal Definition**:
  $$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$
- **Evaluation on `meeting_01` (Semantic Text-Aligned Turn Matching)**:
  - Reference: `data/reference_minutes/meeting_01.txt`
  - Evaluated Segments: 20
  - Correctly Labeled: 13
  - Mismatched Segments: 7
  - Turn Attribution Accuracy: **65.00%**
  - Time-Weighted Speech Accuracy: **65.05%** (67.00s / 103.00s)
  - Time-Weighted Error Rate: **34.95%** (36.00s / 103.00s)
  - Speaker Permutation Check: **Optimal** (Hungarian assignment proves the pipeline mapping is the optimal 1-to-1 bijection).
  - Standard NIST RTTM files generated: `data/diarization/meeting_01_reference.rttm` & `meeting_01_hypothesis.rttm`.

---

## PowerShell Execution Commands

Execute from the workspace root (`C:\Users\bidyu\ai project\New folder\AI-MEETING-MINUTES GENERATOR`):

```powershell
# 1. Run integration on existing diarization & transcript:
.\.venv312\Scripts\python.exe src/diarization/integrate_transcription.py meeting_01

# 2. Run diarization accuracy evaluation:
.\.venv312\Scripts\python.exe src/diarization/evaluate_der.py meeting_01

# 3. Run the complete end-to-end master pipeline:
.\.venv312\Scripts\python.exe src/diarization/run_diarization_pipeline.py meeting_01
```
