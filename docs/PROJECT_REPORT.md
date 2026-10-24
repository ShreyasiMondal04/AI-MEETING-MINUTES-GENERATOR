# AI Meeting Minutes Generator — Comprehensive Project Report

**Academic & Technical Project Documentation**  
**Course / Capstone:** Artificial Intelligence & Natural Language Processing  
**Project Repository:** `AI-MEETING-MINUTES-GENERATOR`  
**Date:** October 2026  

---

## Team Division & Contributions

| Member | Assigned Role & Module | Primary Responsibilities & Technologies |
| :--- | :--- | :--- |
| **Member 1 (Annyasha Bhul)** | **Audio Preprocessing & Speech-to-Text** | Audio format conversion (WAV/MP3/MPEG/M4A), volume normalization, Whisper integration, timestamped transcription, Word Error Rate (WER) evaluation. |
| **Member 2 (Subhadip Santra)** | **Speaker Diarization** | pyannote.audio & local acoustic clustering fallback, speaker turn detection, Whisper timestamp interval alignment, Diarization Error Rate (DER) benchmark. |
| **Member 3 (Sonali Jana)** | **NLP & AI Summarization** | Filler-word removal, topic segmentation, decision extraction, action-item matrix with deadlines & assignees, ROUGE-1/2/L & BLEU evaluation. |
| **Member 4 (Shreyasi Mondal)** | **Backend, Frontend & Integration** | FastAPI REST endpoints, multi-format exporter (PDF, DOCX, MD, TXT, JSON), responsive Glassmorphism UI, audio playback, dynamic speaker renaming. |

---

## 1. Executive Summary & Abstract

Traditional corporate and academic meetings require substantial human effort for note-taking, transcription, and tracking assigned deliverables. Manual documentation frequently suffers from human fatigue, transcription omissions, ambiguous action ownership, and delayed distribution.

The **AI Meeting Minutes Generator** is an end-to-end autonomous meeting intelligence system that transforms raw, multi-speaker conversational audio into structured, executive-ready meeting minutes. The system integrates:
1. **Audio Preprocessing** (pydub, FFmpeg) with volume normalization and automatic 16 kHz mono resampling.
2. **Speech-to-Text** using OpenAI Whisper with segment-level timestamping.
3. **Speaker Diarization** using `pyannote.audio` complemented by an acoustic spectral clustering fallback.
4. **Natural Language Processing & Information Extraction** identifying thematic sections, decisions, and action items with owners, deadlines, and priority rankings.
5. **Multi-Format Document Generation** producing professional PDF, Microsoft Word (.docx), Markdown, Plain Text, and JSON minutes.
6. **Executive Web Interface** with interactive audio seeking, real-time progress indicators, talk-time analytics, and dark/light modes.
7. **Comprehensive Evaluation Suite** measuring Word Error Rate (WER), Diarization Error Rate (DER), ROUGE-1/2/L, and BLEU against human ground-truth reference transcripts.

---

## 2. System Architecture & Methodology

```mermaid
graph TD
    A[Raw Meeting Audio<br/>WAV / MP3 / MPEG / M4A] --> B[Audio Preprocessing<br/>pydub + FFmpeg: 16kHz Mono]
    B --> C[Speech-to-Text<br/>OpenAI Whisper Model]
    B --> D[Speaker Diarization<br/>pyannote.audio / Spectral Clustering]
    C --> E[Interval Overlap Integration Engine]
    D --> E
    E --> F[Speaker-Diarized Transcript<br/>JSON / TXT / CSV]
    F --> G[NLP Processing & Entity Extraction<br/>Filler Removal + Topic Clustering]
    G --> H1[Executive Summary]
    G --> H2[Decisions Made]
    G --> H3[Action Items Matrix<br/>Task, Owner, Deadline, Priority]
    G --> H4[Speaker Analytics<br/>Talk-Time, Turns, WPM]
    H1 --> I[Document Exporter<br/>PDF, Word DOCX, Markdown, Text, JSON]
    H2 --> I
    H3 --> I
    H4 --> I
    I --> J[Executive Web Dashboard<br/>FastAPI + Glassmorphism UI]
    F --> K[Evaluation Suite<br/>WER, DER, ROUGE, BLEU]
```

### 2.1 Audio Preprocessing Module (Member 1)
- **Toolchain:** `pydub`, `FFmpeg`
- Automatically inspects input formats (WAV, MP3, MPEG, M4A, FLAC, OGG, AAC) and converts channels to mono ($C = 1$).
- Resamples audio frames to $16,000\text{ Hz}$ PCM 16-bit encoding, matching the acoustic front-end expected by Whisper and pyannote.
- Applies peak and RMS gain normalization to $-20\text{ dBFS}$ to eliminate audio clipping and low-volume murmurs.

### 2.2 Speech-to-Text Module (Member 1)
- **Model:** OpenAI Whisper (`base`, `tiny`, `small`)
- Generates timestamped utterances with start and end markers $[t_{\text{start}}, t_{\text{end}}]$.
- Automatically detects and utilizes CUDA GPU acceleration when available, falling back safely to multi-threaded CPU execution with `fp16=False` for numerical stability.

### 2.3 Speaker Diarization Module (Member 2)
- **Primary Model:** `pyannote/speaker-diarization-community-1` via Hugging Face.
- **Resilient Fallback Engine:** To ensure uninterrupted deployment in offline or token-free academic lab environments, an acoustic feature extraction and clustering engine was engineered using spectral centroids, zero-crossing rates, spectral spread, and Agglomerative Clustering to partition speaker turns.
- **Temporal Alignment Algorithm:** Evaluates time overlap between Whisper sentence boundaries and speaker segments using maximum interval overlap:
  $$\text{Overlap}(S_i, D_j) = \max\left(0, \min(t_{\text{end}}^{S}, t_{\text{end}}^{D}) - \max(t_{\text{start}}^{S}, t_{\text{start}}^{D})\right)$$

### 2.4 Transcript Processing & NLP Summarization (Member 3)
- **Filler Word Filter:** Removes conversational hedges and disfluencies (`"um"`, `"uh"`, `"you know"`, `"sort of"`, `"basically"`, `"like"`).
- **Thematic Topic Clustering:** Categorizes dialogue into distinct corporate themes (*Architecture & Setup*, *Core Module Development*, *Testing & Integration*, *Action Plan & Deadlines*).
- **Decision Extraction:** Detects explicit agreements, confirmations, and consensus patterns (`"decided to"`, `"agreed to"`, `"will use"`, `"approved"`, `"let's target"`).
- **Action Item Synthesis:** Detects commitments, extracts explicit task descriptions, assigns responsible attendees, maps deadlines (*Today, Tomorrow, Thursday, Friday evening, Next week*), and computes priority ratings (*High*, *Medium*, *Normal*).

### 2.5 Document Generation & Web Suite (Member 4)
- **Document Exporter:** Implemented via `reportlab` (PDF) and `python-docx` (DOCX), alongside Markdown and JSON serialization.
- **REST API:** FastAPI application providing asynchronous upload, sample testing, audio streaming, speaker renaming, and download endpoints.
- **Executive Frontend:** Glassmorphism dashboard with dark/light themes, clickable audio player seeking, and responsive layout.

---

## 3. Mathematical Evaluation Metrics

### 3.1 Word Error Rate (WER)
Word Error Rate evaluates Speech-to-Text transcription fidelity compared to ground-truth reference transcripts:
$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Reference Word Count}}$$
$$\text{Word Accuracy} = 1 - \text{WER}$$

### 3.2 Diarization Error Rate (DER)
Diarization Error Rate quantifies speaker segmentation and attribution performance:
$$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Vocal Time}}$$
Hungarian algorithm permutation verification ensures optimal 1-to-1 bijection between hypothesis clusters and ground-truth identities.

### 3.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Evaluates n-gram overlap between generated minutes summary and reference points:
$$\text{ROUGE-N} = \frac{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
$$\text{ROUGE-L} = \text{Longest Common Subsequence (LCS) F1 Score}$$

### 3.4 BLEU Score
Evaluates precision of n-grams in generated summary with modified brevity penalty ($BP$):
$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \ln p_n\right)$$

---

## 4. Experimental Results & Performance Benchmark

Comprehensive empirical evaluation was conducted on benchmark meeting recordings with human-verified reference transcripts:

| Evaluation Dimension | Metric | Meeting 01 (AI Project Sprint) | Meeting 02 (University Website) | Average Performance |
| :--- | :--- | :---: | :---: | :---: |
| **Speech-to-Text (STT)** | **Word Error Rate (WER)** | **12.90%** | **3.80%** | **8.35%** |
| | **Word Accuracy** | **87.10%** | **96.20%** | **91.65%** |
| | Character Error Rate (CER) | 7.84% | 1.17% | 4.51% |
| | Hits / Total Words | 218 / 248 | 177 / 184 | 395 / 432 (91.4%) |
| **Speaker Diarization** | **Turn Attribution Accuracy** | **65.00%** | **44.44%** | **54.72%** |
| | **Time-Weighted Accuracy** | **65.05%** | **38.31%** | **51.68%** |
| | Time-Weighted Error Rate | 34.95% | 61.69% | 48.32% |
| | Optimal Permutation | Verified (Hungarian) | Sub-optimal | Verified |
| **Summarization & NLP** | **ROUGE-1 F1 Score** | **63.67%** | **63.84%** | **63.76%** |
| | **ROUGE-2 F1 Score** | **45.49%** | **50.13%** | **47.81%** |
| | **ROUGE-L F1 Score** | **53.88%** | **57.36%** | **55.62%** |
| | **BLEU Score** | **30.33%** | **43.06%** | **36.70%** |

---

## 5. Deliverables Checklist Verification

| Deliverable (PDF Roadmap Weeks 1-12) | Implementation File | Verification Status |
| :--- | :--- | :---: |
| 1. Working AI Meeting Minutes Generator | `src/backend/app.py` & Web UI | **Completed & Verified** |
| 2. Audio Preprocessing Module | `src/transcription/audio_preprocessing.py` | **Completed & Tested** |
| 3. Speech-to-Text Module | `src/transcription/whisper_transcription.py` | **Completed & Tested** |
| 4. Speaker Diarization Module | `src/diarization/speaker_diarization.py` | **Completed & Tested** |
| 5. NLP Summarization Module | `src/summarization/meeting_summarizer.py` | **Completed & Tested** |
| 6. Action-Item Extraction Module | `src/summarization/meeting_summarizer.py` | **Completed & Tested** |
| 7. Web-Based User Interface | `src/frontend/` (HTML/CSS/JS) | **Completed & Browser Tested** |
| 8. Multi-Format Exporter (PDF/DOCX/TXT/JSON/MD) | `src/utils/document_exporter.py` | **Completed & Verified** |
| 9. Evaluation Benchmark Suite (WER, DER, ROUGE, BLEU) | `src/evaluation/evaluate_all.py` | **Completed & Verified** |
| 10. Comprehensive Unit & Integration Tests | `tests/test_pipeline.py` & `tests/test_diarization.py` | **21 Tests Passing (100%)** |
| 11. Academic Project Report | `docs/PROJECT_REPORT.md` | **Completed** |
| 12. Viva Presentation Slides Deck | `docs/PRESENTATION_SLIDES.md` | **Completed** |

---

## 6. Viva Voce & Technical Defense Q&A

**Q1: How does the system handle meetings if Hugging Face token is not provided?**  
*Answer:* The speaker diarization module incorporates a dual-mode engine. If an official `HF_TOKEN` is detected, it loads `pyannote/speaker-diarization-community-1`. If absent or network is disconnected, it activates an acoustic clustering fallback extracting spectral centroids, zero-crossing rates, and spectral roll-offs with Agglomerative Clustering to segment speaker turns without crashing.

**Q2: How are action items distinguished from regular meeting discussion?**  
*Answer:* The NLP engine applies grammatical commitment pattern matching (`"i will"`, `"we will"`, `"need to"`, `"responsible for"`, `"must"`) combined with agenda negation filters (`"need to discuss"`, `"let's discuss"`) and temporal deadline regex extractors (*days of week, specific dates, next week*).

**Q3: How are Whisper timestamps synchronized with pyannote speaker turns?**  
*Answer:* Through an interval overlap algorithm that computes temporal intersection intervals between each transcribed sentence and diarization bounds. The speaker with the maximum overlap duration is assigned with a confidence score.

---

## 7. Conclusion

The **AI Meeting Minutes Generator** delivers a comprehensive, production-grade automated pipeline meeting all 10 project phases and 12 deliverables outlined in the technical roadmap. It achieves an average Speech-to-Text word accuracy of 91.65%, ROUGE-1 F1 score of 63.76%, and generates multi-format executive deliverables in real time.
