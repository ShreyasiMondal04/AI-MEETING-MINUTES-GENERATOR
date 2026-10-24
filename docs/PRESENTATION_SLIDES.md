# AI Meeting Minutes Generator — Presentation Slide Deck

**Presentation & Viva Defense Slides Outline**  
**Project:** AI Meeting Minutes Generator  
**Team Members:**  
- Annyasha Bhul (Member 1: Audio & Speech-to-Text)  
- Subhadip Santra (Member 2: Speaker Diarization)  
- Sonali Jana (Member 3: NLP & Summarization)  
- Shreyasi Mondal (Member 4: Backend, Frontend & Integration)  

---

### Slide 1: Title Slide
- **Title:** AI Meeting Minutes Generator
- **Subtitle:** An Autonomous End-to-End Pipeline for Audio Transcription, Speaker Diarization, and Executive Minutes Synthesis
- **Department:** Computer Science & Engineering / Artificial Intelligence
- **Presentation Date:** October 2026

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:**
  - Corporate and academic meetings generate vital information that is often lost or poorly recorded.
  - Manual note-taking is subjective, slow, prone to transcription errors, and distracts participants.
  - Ambiguous action items and unclear task assignments lead to accountability bottlenecks.
- **Our Vision:**
  - Develop an intelligent, zero-friction AI system that ingests raw conversational meeting audio and automatically synthesizes structured, executive-ready meeting minutes in seconds.

---

### Slide 3: Project Objectives & Scope
- **Core Objectives:**
  1. Support diverse audio formats (WAV, MP3, MPEG, M4A, FLAC) with volume normalization.
  2. Perform high-accuracy Speech-to-Text transcription with temporal timestamps via OpenAI Whisper.
  3. Distinguish between multiple speakers using Speaker Diarization.
  4. Automatically extract decisions, key topics, and action items with assignees and deadlines.
  5. Generate multi-format deliverables: PDF, Word (.docx), Markdown, Plain Text, and JSON.
  6. Provide an interactive web dashboard with audio seeking, talk-time analytics, and dark/light themes.
  7. Quantitatively evaluate the pipeline using WER, DER, ROUGE-1/2/L, and BLEU metrics.

---

### Slide 4: System Architecture & Workflow Pipeline
- **End-to-End Pipeline:**
  $$\text{Meeting Audio} \rightarrow \text{Preprocessing} \rightarrow \text{Whisper STT} \rightarrow \text{pyannote Diarization} \rightarrow \text{Interval Overlap} \rightarrow \text{NLP Synthesis} \rightarrow \text{Executive Minutes}$$
- **Modular Micro-Architecture:**
  - Independent decoupled modules allowing unit testing and separate component optimization.
  - RESTful FastAPI backend connecting the AI processing pipeline with the web user interface.

---

### Slide 5: Module 1 — Audio Preprocessing (Member 1)
- **Role:** Annyasha Bhul
- **Technologies:** Python, `pydub`, `FFmpeg`
- **Key Functions:**
  - Auto-detection of system `FFmpeg` binaries across system PATH and environment folders.
  - Multi-format ingestion: WAV, MP3, MPEG, M4A, OGG, FLAC.
  - Conversion to standard 16 kHz Mono PCM 16-bit encoding.
  - Dynamic gain normalization to $-20\text{ dBFS}$ to eliminate volume disparity between speakers.
- **Deliverable:** Clean, normalized audio ready for acoustic modeling.

---

### Slide 6: Module 2 — Speech-to-Text Transcription (Member 1)
- **Role:** Annyasha Bhul
- **Technologies:** OpenAI Whisper (`base`, `tiny`, `small`), PyTorch
- **Key Functions:**
  - Automatic hardware acceleration: detects CUDA GPU; falls back to CPU with `fp16=False`.
  - Sentence-level timestamping $[t_{\text{start}}, t_{\text{end}}]$.
  - Generation of structured JSON transcripts and standard timestamped text logs.
- **Evaluation:** Evaluated with Word Error Rate (WER) using `jiwer`.
  - *Result on Sample 1:* **87.10% Word Accuracy** (12.90% WER).
  - *Result on Sample 2:* **96.20% Word Accuracy** (3.80% WER).

---

### Slide 7: Module 3 — Speaker Diarization (Member 2)
- **Role:** Subhadip Santra
- **Technologies:** `pyannote.audio`, PyTorch, SoundFile, scikit-learn
- **Dual-Mode Engine:**
  - **Online Mode:** Utilizes `pyannote/speaker-diarization-community-1` via Hugging Face token.
  - **Resilient Fallback Mode:** In offline lab environments without tokens, activates an acoustic clustering engine using spectral centroid, zero-crossing rates, and Agglomerative Clustering.
- **Key Functions:**
  - Identifies distinct speaker vocal turns: `SPEAKER_00`, `SPEAKER_01`, etc.
  - Exports standard NIST RTTM files and tabular CSV logs.
- **Deliverable:** Temporal speaker segmentation logs.

---

### Slide 8: Module 4 — Transcript Integration (Member 2)
- **Role:** Subhadip Santra
- **Core Algorithm:** Interval Overlap Alignment
  - For each Whisper segment, computes temporal intersection with all speaker turns:
    $$\text{Overlap} = \max\left(0, \min(t_{\text{end}}^{W}, t_{\text{end}}^{D}) - \max(t_{\text{start}}^{W}, t_{\text{start}}^{D})\right)$$
  - Assigns dominant speaker label (`Speaker 1`, `Speaker 2`) with confidence score.
  - Handles secondary overlapping speech and temporal transition gaps.
- **Deliverables:** Integrated JSON, TXT, and CSV speaker transcripts.

---

### Slide 9: Module 5 — NLP Processing & Summarization (Member 3)
- **Role:** Sonali Jana
- **Technologies:** Python, Regex NLP, `rouge-score`, `nltk`
- **Key Functions:**
  - **Filler-Word Filter:** Eliminates disfluencies (`"um"`, `"uh"`, `"you know"`, `"sort of"`, `"basically"`).
  - **Topic Clustering:** Segments discussion into thematic pillars (*Architecture, Modules, Testing, Deadlines*).
  - **Executive Summary:** Generates concise multi-sentence overview of meeting outcomes.
- **Evaluation:**
  - *ROUGE-1 F1:* **63.76%** average semantic recall.
  - *ROUGE-L F1:* **55.62%** longest common sequence score.

---

### Slide 10: Module 6 — Action Items & Decision Extraction (Member 3)
- **Role:** Sonali Jana
- **Decision Engine:**
  - Detects explicit consensus (`"agreed to"`, `"decided to"`, `"will use"`, `"approved"`).
- **Action Item Matrix:**
  - Identifies commitments (`"i will"`, `"we will"`, `"need to"`, `"responsible for"`).
  - Assigns responsible attendee based on first-person alignment.
  - Extracts explicit deadlines (*Today, Tomorrow, Thursday, Friday evening, Next week*).
  - Assigns priority ratings: **High**, **Medium**, **Normal**.

---

### Slide 11: Module 7 — Document Exporter & Backend API (Member 4)
- **Role:** Shreyasi Mondal
- **Technologies:** FastAPI, `reportlab`, `python-docx`, Uvicorn
- **Multi-Format Export Engine:**
  - **PDF:** Corporate styled document with headers, metadata badge grid, decisions callout, action items table, and full transcript.
  - **Word (.docx):** Editable Microsoft Word document with styled tables.
  - **Markdown & TXT:** Clean plain text and GFM files.
  - **JSON:** Complete machine-readable payload.
- **REST Endpoints:** `/health`, `/api/samples`, `/api/process-meeting`, `/api/export/{id}/{format}`, `/api/rename-speakers/{id}`.

---

### Slide 12: Module 8 — Executive Web Interface (Member 4)
- **Role:** Shreyasi Mondal
- **Technologies:** HTML5, CSS3 (Vanilla), Modern JavaScript
- **Key Features:**
  - **Glassmorphism Design:** Curated color palette, glowing ambient backdrop, dark/light theme switch.
  - **Dual Input Hub:** Drag-and-drop file upload zone + one-click sample demonstration cards.
  - **Integrated Audio Player:** Plays meeting audio; clicking any transcript timestamp seeks audio player instantly!
  - **5 Interactive Tabs:**
    1. Executive Minutes
    2. Diarized Transcript (with speaker search & filter)
    3. Participation Analytics (talk-time distribution bar & stats)
    4. Academic Benchmarks (WER, DER, ROUGE, BLEU scorecards)
    5. Export Center (one-click download buttons)
  - **Dynamic Speaker Renaming Modal:** Renames "Speaker 1" to actual names across all minutes in real time.

---

### Slide 13: Experimental Results & Benchmarks
- **Comprehensive Benchmark Table:**

| Metric | Meeting 01 (Sprint) | Meeting 02 (Website) | Average |
| :--- | :---: | :---: | :---: |
| **STT Word Accuracy (1 - WER)** | **87.10%** | **96.20%** | **91.65%** |
| **Word Error Rate (WER)** | 12.90% | 3.80% | 8.35% |
| **Diarization Turn Accuracy** | **65.00%** | **44.44%** | **54.72%** |
| **Time-Weighted Speech Accuracy**| **65.05%** | **38.31%** | **51.68%** |
| **ROUGE-1 F1 Score** | **63.67%** | **63.84%** | **63.76%** |
| **ROUGE-2 F1 Score** | **45.49%** | **50.13%** | **47.81%** |
| **ROUGE-L F1 Score** | **53.88%** | **57.36%** | **55.62%** |
| **BLEU Score** | **30.33%** | **43.06%** | **36.70%** |

---

### Slide 14: Engineering Challenges & Innovations
1. **Challenge:** Dependency on Hugging Face token for pyannote speaker diarization.
   - *Solution:* Built an acoustic spectral clustering fallback using MFCC/spectral features that guarantees 100% offline uptime.
2. **Challenge:** Whisper splitting single sentences into multiple intervals causing alignment drift.
   - *Solution:* Implemented token-overlap semantic alignment and maximum temporal intersection.
3. **Challenge:** Formatting multi-page executive reports without external binary tools.
   - *Solution:* Programmed a pure-Python `reportlab` PDF flowable generator with corporate typography, metadata grids, and table styling.

---

### Slide 15: Conclusion & Future Scope
- **Project Achievements:**
  - Successfully delivered all 10 roadmap phases and 12 required deliverables.
  - Achieved high speech-to-text accuracy (91.65% avg) and robust NLP synthesis.
  - Fully verified with 21 unit & integration tests and automated browser testing.
- **Future Enhancements:**
  - Real-time live audio streaming transcription via WebSockets.
  - Multi-language diarization and cross-lingual translation.
  - Integration with Zoom, Google Meet, and Microsoft Teams bots.
- **Thank You! Questions & Discussion.**
