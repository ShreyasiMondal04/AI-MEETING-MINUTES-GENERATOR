# MinuteAI — Executive AI Meeting Minutes Generator

An end-to-end, production-grade automated meeting intelligence suite that transforms multi-speaker conversational audio into structured, executive-ready meeting minutes, decisions, and action items.

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-green.svg)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-orange.svg)](https://github.com/openai/whisper)
[![pyannote.audio](https://img.shields.io/badge/Diarization-pyannote.audio-purple.svg)](https://github.com/pyannote/pyannote-audio)
[![Tests Passing](https://img.shields.io/badge/Tests-21%2F21%20Passing-brightgreen.svg)]()

---

## 🎓 B.Tech 3rd Year Student Quick Start & Viva Voce Guide

This project is tailored specifically for **B.Tech 3rd Year (Minor Project / Academic Defense)** standards. It provides two simple modes to run and demonstrate the project:

### 1. Terminal Demo (Recommended for Viva Voce & Low-Spec PCs)
Double-click [`start_terminal_demo.bat`](file:///c:/Users/HP/OneDrive/Desktop/AI_MEETING_MINUTES_GENERATOR/start_terminal_demo.bat) or run:
```powershell
python main.py
```
- Step-by-step console visualization of all 5 stages.
- Generates instant executive minutes, decisions, and action items table with talk-time analytics.
- Exports formatted **PDF**, **Word (.docx)**, and **Plain Text** files directly to `data/meeting_minutes/`.

### 2. Modern Web Dashboard (For PPT Presentation & Project Exhibition)
Double-click [`start_app.bat`](file:///c:/Users/HP/OneDrive/Desktop/AI_MEETING_MINUTES_GENERATOR/start_app.bat) or run:
```powershell
python run_app.py
```
- Opens browser automatically at `http://127.0.0.1:8000`.
- 1-click sample demo buttons, audio player seek sync on timestamp click, speaker renaming, and multi-format exports.

### 📚 Academic Defense & Preparation Resources
- 📖 [**Student Viva Voce Preparation Guide (Top 25 Questions & Model Answers)**](docs/STUDENT_VIVA_GUIDE.md)
- 📝 [**Whiteboard Code Cheatsheet (Algorithms to write on paper/board)**](docs/WHITEBOARD_CODE_CHEATSHEET.md)
- 📊 [**Academic Presentation Slides (Ready for College Seminar)**](docs/PRESENTATION_SLIDES.md)
- 📄 [**Full Academic Project Report (University Standard)**](docs/PROJECT_REPORT.md)

---

## Key Features

- **Multi-Format Audio Preprocessing:** Supports WAV, MP3, MPEG, M4A, FLAC, and OGG with automated FFmpeg discovery, mono conversion, 16 kHz resampling, and dynamic $-20\text{ dBFS}$ volume normalization.
- **OpenAI Whisper Speech-to-Text:** Generates timestamped utterances with automatic CUDA GPU acceleration and CPU fallback.
- **Speaker Diarization Engine:** Dual-mode architecture utilizing `pyannote.audio` with an intelligent acoustic spectral clustering fallback ensuring 100% offline uptime.
- **NLP Information Extraction:** Eliminates filler words (`"um"`, `"uh"`, `"you know"`, `"basically"`), clusters thematic discussion sections, detects explicit decisions, and extracts structured action items with assignees, deadlines, and priorities.
- **Multi-Format Document Exporter:** Instant one-click exports to:
  - **Executive PDF Report** (formatted corporate design with tables and transcript)
  - **Microsoft Word (.docx)** (editable Word document)
  - **GitHub Markdown (.md)** (clean GFM)
  - **Universal Plain Text (.txt)** (email & archive ready)
  - **Structured JSON (.json)** (database & API payload)
- **Executive Web Dashboard:** Glassmorphism interface with dark/light themes, integrated seekable audio player, real-time pipeline stepper, talk-time participation analytics, and interactive speaker renaming.
- **Academic Evaluation Suite:** Comprehensive scoring calculating Word Error Rate (WER), Diarization Error Rate (DER), ROUGE-1/2/L, and BLEU against human ground truth.

---

## Proposed System Architecture

```text
       Raw Audio Recording (WAV, MP3, MPEG, M4A)
                          ↓
               Audio Preprocessing (pydub + FFmpeg)
             [Mono 16 kHz PCM, -20 dBFS Normalization]
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
 Speech-to-Text (Whisper)         Speaker Diarization (pyannote)
 [Timestamped Utterances]          [Temporal Vocal Turns]
        └─────────────────┬─────────────────┘
                          ↓
           Interval Overlap Integration Engine
                          ↓
             Speaker-Diarized Transcript
                  [JSON, TXT, CSV]
                          ↓
           AI / NLP Entity Extraction Engine
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
Executive Summary   Decisions Made   Action Items Matrix
        └─────────────────┬─────────────────┘
                          ↓
           Document Exporter & REST API
        [PDF, Word DOCX, Markdown, Text, JSON]
                          ↓
              Executive Web Application
```

---

## Technology Stack

| Component | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Speech-to-Text** | OpenAI Whisper, PyTorch |
| **Speaker Diarization** | pyannote.audio, PyTorch, SoundFile, scikit-learn |
| **Audio Processing** | pydub, FFmpeg |
| **NLP & Summarization** | Regex NLP, `rouge-score`, NLTK (BLEU) |
| **Backend API** | FastAPI, Uvicorn, Python-Multipart |
| **Document Export** | ReportLab (PDF), python-docx (Word) |
| **Frontend UI** | Modern HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript |
| **Evaluation Metrics** | jiwer (WER/CER), pyannote RTTM (DER), ROUGE-1/2/L, BLEU |

---

## Performance Evaluation & Benchmarks

Empirical testing conducted on benchmark meeting recordings with human ground truth:

| Evaluation Dimension | Metric | Meeting 01 (Sprint) | Meeting 02 (Website) | Average |
| :--- | :--- | :---: | :---: | :---: |
| **Speech-to-Text (STT)** | **Word Error Rate (WER)** | **12.90%** | **3.80%** | **8.35%** |
| | **Word Accuracy (1 - WER)** | **87.10%** | **96.20%** | **91.65%** |
| | Character Error Rate (CER) | 7.84% | 1.17% | 4.51% |
| **Speaker Diarization** | **Turn Attribution Accuracy** | **65.00%** | **44.44%** | **54.72%** |
| | **Time-Weighted Accuracy** | **65.05%** | **38.31%** | **51.68%** |
| **Summarization & NLP** | **ROUGE-1 F1 Score** | **63.67%** | **63.84%** | **63.76%** |
| | **ROUGE-2 F1 Score** | **45.49%** | **50.13%** | **47.81%** |
| | **ROUGE-L F1 Score** | **53.88%** | **57.36%** | **55.62%** |
| | **BLEU Score** | **30.33%** | **43.06%** | **36.70%** |

---

## Team Division & Deliverables

| Member | Assigned Role | Modules & Deliverables |
| :--- | :--- | :--- |
| **Member 1 (Annyasha Bhul)** | Audio + Speech-to-Text | `src/transcription/audio_preprocessing.py`, `src/transcription/whisper_transcription.py`, `src/transcription/evaluate_wer.py` |
| **Member 2 (Subhadip Santra)** | Speaker Diarization | `src/diarization/speaker_diarization.py`, `src/diarization/integrate_transcription.py`, `src/diarization/evaluate_der.py` |
| **Member 3 (Sonali Jana)** | NLP & Summarization | `src/summarization/meeting_summarizer.py`, topic clustering, decision & action-item extraction |
| **Member 4 (Shreyasi Mondal)** | Backend + Frontend + Integration | `src/backend/app.py`, `src/frontend/` (HTML/CSS/JS), `src/utils/document_exporter.py` |

---

## Quick Start & Installation

### 1. Activate Environment & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/AI-MEETING-MINUTES-GENERATOR.git
cd AI-MEETING-MINUTES-GENERATOR

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or: .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
# Start the FastAPI server on port 8000
python -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 3. Run Automated Tests & Evaluation
```bash
# Run unit & integration test suites (21 tests)
python -m unittest tests/test_diarization.py
python -m unittest tests/test_pipeline.py

# Run master evaluation benchmark (WER, DER, ROUGE, BLEU)
python src/evaluation/evaluate_all.py
```

---

## REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System diagnostics, hardware device, and FFmpeg detection |
| `GET` | `/api/samples` | List preloaded benchmark meeting recordings |
| `POST` | `/api/process-meeting` | Upload audio and execute end-to-end processing pipeline |
| `POST` | `/api/process-sample/{id}` | Execute or retrieve pipeline results for sample meetings |
| `GET` | `/api/meetings/{id}` | Retrieve full meeting data (minutes, transcript, audio) |
| `GET` | `/api/export/{id}/{format}` | Download export document (`pdf`, `docx`, `txt`, `json`, `md`) |
| `POST` | `/api/rename-speakers/{id}` | Dynamically rename speakers and re-export minutes |
| `GET` | `/api/evaluate/{id}` | Compute WER, DER, ROUGE, and BLEU against ground truth |
| `GET` | `/api/audio/{filename}` | Stream meeting audio for web playback |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
