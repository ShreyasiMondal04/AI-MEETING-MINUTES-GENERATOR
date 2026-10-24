# 🎓 B.Tech 3rd Year Viva Voce & Project Review Guide
## AI Meeting Minutes Generator — Minor Project / Academic Defense

---

## 1. The 60-Second Elevator Pitch
> **When the External Examiner asks:** *"Introduce your project and explain what problem it solves."*

### 🎙️ Word-for-Word Script to Say:
> *"Good morning/afternoon, Sir/Madam. Our project is the **AI Meeting Minutes Generator**.*
>
> *In corporate and academic environments, manual note-taking during meetings is slow, error-prone, and distracts participants from active discussion. Existing commercial tools like Otter.ai or Fireflies.ai require paid cloud subscriptions and upload confidential corporate audio to third-party servers, posing data privacy risks.*
>
> *To solve this, we built a **100% locally-run, privacy-preserving AI pipeline** that takes raw meeting audio (WAV, MP3, MPEG, M4A) and automatically produces structured executive minutes containing:*
> 1. *Clean speaker-diarized transcripts ('Who spoke when'),*
> 2. *Executive summary and thematic discussion topics,*
> 3. *Consensus decisions reached, and*
> 4. *Action items with assignees, deadlines, and priorities.*
>
> *We evaluated our system using standard academic benchmarks: **Word Error Rate (WER: 8.35%)**, **Diarization Error Rate (DER)**, and **ROUGE/BLEU scores** for summarization, and built both an interactive terminal interface and a modern web dashboard with PDF and Word exports."*

---

## 2. System Architecture in 5 Simple Steps

```
[ Raw Audio File (WAV/MP3/MPEG) ]
              │
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STAGE 1: Audio Preprocessing (pydub + FFmpeg)          │
  │ • Resample to 16 kHz Mono PCM WAV                       │
  │ • Normalize audio volume to -20 dBFS                    │
  └───────────────────────────┬─────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  ┌────────────────────────┐      ┌────────────────────────┐
  │ STAGE 2: Speech-to-Text│      │ STAGE 3: Diarization   │
  │ • OpenAI Whisper model │      │ • pyannote / Spectral  │
  │ • Outputs timestamped  │      │   Acoustic Clustering  │
  │   words and sentences  │      │ • Identifies speakers  │
  └───────────┬────────────┘      └───────────┬────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STAGE 4: Interval Overlap Integration                   │
  │ • Aligns Whisper word timestamps with Speaker turns     │
  │ • Outputs: [00:00:15 - 00:00:22] Speaker 1: "Hello..."  │
  └───────────────────────────┬─────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STAGE 5: NLP Summarization & Document Export            │
  │ • Filters conversational filler words ("um", "uh")     │
  │ • Extracts: Summary, Key Decisions, Action Item Matrix  │
  │ • Exports to: PDF, Word (.docx), Markdown, Plain Text   │
  └─────────────────────────────────────────────────────────┘
```

---

## 3. Individual Team Member Roles & Contributions

When the examiner asks: *"What was your individual contribution to this project?"*

| Team Member | Assigned Role | Exact Contribution to Explain |
| :--- | :--- | :--- |
| **Annyasha Bhul** | **Lead Developer & System Integration** | *"I designed the overall modular architecture, built the FastAPI backend endpoints, and created the responsive web dashboard with live audio playback synchronization and PDF/Word report generation."* |
| **Subhadip Santra** | **Speech-to-Text Engineer** | *"I developed the audio preprocessing pipeline (16 kHz mono conversion, -20 dBFS volume normalization) and integrated OpenAI Whisper for timestamped transcription with GPU/CPU auto-detection and WER evaluation."* |
| **Sonali Jana** | **Speaker Diarization Specialist** | *"I implemented the speaker diarization pipeline using pyannote.audio and developed an acoustic spectral clustering fallback using scikit-learn for 100% offline uptime, along with the interval overlap integration algorithm."* |
| **Shreyasi Mondal** | **NLP & Information Extraction Specialist** | *"I engineered the NLP summarization module, filler-word filtering, thematic topic segmentation, consensus decision detection, and the rule-based action item matrix extracting owners, deadlines, and priorities."* |

---

## 4. Top 25 Viva Voce Questions & Model Answers

### Category A: Speech & Audio Processing

#### Q1: Why do we convert audio to 16 kHz Mono PCM WAV?
> **Answer:**
> - **16 kHz Sampling Rate:** Human speech fundamental frequencies and formants lie predominantly below 8 kHz. According to the **Nyquist-Shannon Sampling Theorem**, the sampling frequency must be at least twice the maximum signal frequency ($2 \times 8\text{ kHz} = 16\text{ kHz}$). Higher rates (like 44.1 kHz used for music) increase computational load without improving speech recognition accuracy.
> - **Mono Channel:** Stereo audio contains two channels which can introduce phase differences or channel imbalance. Converting to mono simplifies acoustic feature extraction.
> - **PCM WAV:** Uncompressed Linear PCM format provides raw, uncompressed pulse-code modulation samples that machine learning audio models ingest directly.

#### Q2: What is -20 dBFS Normalization and why is it needed?
> **Answer:**
> dBFS stands for *Decibels relative to Full Scale*. A value of 0 dBFS represents the maximum possible digital amplitude before clipping distortion occurs. Normalizing to $-20\text{ dBFS}$ standardizes volume across quiet and loud speakers, ensuring that Whisper's Mel-spectrogram feature extractor receives consistent energy levels.

---

### Category B: Speech-to-Text (ASR) & OpenAI Whisper

#### Q3: What is OpenAI Whisper and how does it work?
> **Answer:**
> Whisper is an **Encoder-Decoder Transformer** model trained on 680,000 hours of multilingual, weakly supervised audio data.
> 1. The input audio is converted into an **80-channel log-Mel spectrogram**.
> 2. The **Transformer Encoder** processes the spectrogram with 1D convolution and self-attention layers to create contextual audio representations.
> 3. The **Transformer Decoder** generates text tokens autoregressively, predicting words along with special timestamp tokens (e.g. `<|0.00|>`, `<|2.50|>`) that provide word-level timing.

#### Q4: What is Word Error Rate (WER) and how is it calculated?
> **Answer:**
> Word Error Rate is the standard academic evaluation metric for Automatic Speech Recognition (ASR). It is based on the **Levenshtein Distance** between the predicted transcript and the human ground-truth reference:
> $$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Words in Reference}}$$
> - In our benchmarks, our system achieved a WER of **8.35%** (which corresponds to an accuracy of **91.65%**).

---

### Category C: Speaker Diarization ("Who Spoke When")

#### Q5: What is Speaker Diarization?
> **Answer:**
> Speaker Diarization is the process of partitioning an input audio recording into temporal segments according to speaker identity. Simply put, it answers the question: **"Who spoke when?"** without requiring prior knowledge of the speakers' voices.

#### Q6: What if the user does not have a HuggingFace Token or Internet connection?
> **Answer:**
> In our project, we implemented a **dual-mode architecture**:
> 1. **Primary Mode:** Uses `pyannote/speaker-diarization-community-1` neural pipeline when an optional HuggingFace token is provided.
> 2. **Intelligent Offline Fallback:** If no token or internet is available, our system automatically falls back to an **acoustic feature extraction engine** (using Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate, and RMS Energy) clustered via `AgglomerativeClustering` from `scikit-learn`. This ensures **100% offline uptime** without runtime crashes!

#### Q7: How do you align Whisper text with Speaker turns?
> **Answer:**
> Whisper gives segments with start and end times: $[t_{start}^{W}, t_{end}^{W}]$.
> Diarization gives intervals with speaker labels: $[t_{start}^{D}, t_{end}^{D}, \text{Speaker}]$.
> For every Whisper segment, we calculate the **temporal overlap duration** with each diarization turn:
> $$\text{Overlap} = \max(0, \min(t_{end}^W, t_{end}^D) - \max(t_{start}^W, t_{start}^D))$$
> The speaker with the maximum overlap duration is assigned to that dialogue turn.

#### Q8: What is Diarization Error Rate (DER)?
> **Answer:**
> DER is the standard NIST metric defined as:
> $$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$

---

### Category D: NLP & Meeting Summarization

#### Q9: What is the difference between Extractive and Abstractive Summarization?
> **Answer:**
> - **Extractive Summarization:** Selects key sentences directly from the source transcript based on importance scores, TF-IDF, or semantic clustering.
> - **Abstractive Summarization:** Generates brand-new sentences that rephrase and synthesize the core meaning (like an LLM does).
> - Our system uses an **agenda-guided extractive and entity-mining approach**, ensuring zero hallucinations for factual business commitments, deadlines, and financial figures.

#### Q10: How does your system extract Action Items, Owners, and Deadlines?
> **Answer:**
> We use rule-based Natural Language Processing with regular expressions and grammatical pattern matching:
> - **Action Detection:** Detects modal commitment verbs (*"will", "need to", "must", "should", "plan to", "going to"*).
> - **Assignee Extraction:** Identifies explicit first-person commitments (*"I will start today"* $\rightarrow$ current speaker) or direct delegations (*"Subhadip will complete..."* $\rightarrow$ named individual).
> - **Deadline Extraction:** Detects temporal prepositional phrases (*"by Friday evening", "tomorrow", "next week", "by 5 PM"*).
> - **Priority Assignment:** Scored as High, Medium, or Normal based on urgency keywords (*"urgent", "prioritize", "today"* $\rightarrow$ High).

#### Q11: What are ROUGE and BLEU scores?
> **Answer:**
> - **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures n-gram overlap between the generated summary and reference summary.
>   - ROUGE-1: Unigram (single word) overlap.
>   - ROUGE-2: Bigram (two-word phrase) overlap.
>   - ROUGE-L: Longest Common Subsequence (LCS).
> - **BLEU (Bilingual Evaluation Understudy):** Precision-focused n-gram metric commonly used to verify semantic fidelity.
> - Our generated summaries achieve an average **ROUGE-1 F1 score of 63.76%** and **BLEU score of 36.70%**.

---

### Category E: Software Engineering & Architecture

#### Q12: Why did you choose FastAPI over Flask or Django?
> **Answer:**
> 1. **High Performance:** FastAPI is built on Starlette and Pydantic, supporting asynchronous (`async`/`await`) request handling with ASGI servers like Uvicorn.
> 2. **Automatic Data Validation:** Pydantic models automatically validate incoming payloads and file types.
> 3. **Interactive Documentation:** Automatically generates Swagger UI (`/docs`) and ReDoc endpoints for instant API testing.

#### Q13: Why did you build both a Terminal CLI and a Web Dashboard?
> **Answer:**
> - The **Interactive Terminal CLI (`simple_main.py`)** allows developers, systems administrators, or viva evaluators to test and step through the 5-stage pipeline in a lightweight, resource-efficient environment.
> - The **Web Dashboard (`http://127.0.0.1:8000`)** provides a modern, consumer-ready GUI with audio player seeking, speaker renaming, and multi-format download buttons for corporate end-users.

---

## 5. How to Demonstrate During Viva

### 🌟 Option 1: Terminal Demonstration (2 minutes)
1. Double-click `start_terminal_demo.bat` (or run `python simple_main.py`).
2. Press `1` to run the Sprint Planning Demo.
3. Show the examiner the live output:
   - Audio preprocessing $\rightarrow$ Whisper transcription $\rightarrow$ Diarization $\rightarrow$ Aligned turns $\rightarrow$ Action Items Matrix with Assignees and Deadlines.
4. Show the generated PDF in the `data/meeting_minutes/` folder.

### 🌟 Option 2: Web Dashboard Demonstration (2 minutes)
1. Double-click `start_app.bat` (or run `python run_app.py`).
2. The browser opens automatically at `http://127.0.0.1:8000`.
3. Click **"Load & Process Sample 1"**.
4. Point out:
   - Click any timestamp (e.g., `[00:00:23]`) to show the audio player automatically jumping to that exact word!
   - Click the pencil icon to rename `Speaker 1` to `Annyasha`.
   - Click **"Export as PDF"** or **"Export as DOCX"** to show instant corporate document generation.
   - Switch to the **Academic Benchmark Tab** to show live WER, DER, and ROUGE metrics!
