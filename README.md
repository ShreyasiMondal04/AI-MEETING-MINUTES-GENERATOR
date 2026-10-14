AI Meeting Minutes Generator

Overview

The AI Meeting Minutes Generator is an intelligent audio-processing system designed to automatically convert meeting recordings into structured meeting minutes.

The system aims to:

* Convert meeting audio into text using speech-to-text technology.
* Differentiate between multiple speakers using speaker diarization.
* Generate concise meeting summaries.
* Extract important decisions and action items.
* Identify assigned tasks and deadlines where possible.

Proposed Workflow

Audio Recording
↓
Audio Preprocessing
↓
Speech-to-Text (Whisper)
↓
Speaker Diarization (pyannote.audio)
↓
Speaker-Labeled Transcript
↓
AI/NLP Processing
↓
Summary + Decisions + Action Items
↓
Structured Meeting Minutes

 Technologies

* Python
* OpenAI Whisper
* pyannote.audio
* pydub
* PyTorch
* NLP/LLM-based summarization

Project Structure

```text
src/
├── transcription/
├── diarization/
├── summarization/
└── utils/
```

Current Status

🚧 Project initialization and research phase.

 Completed

* Project idea and architecture finalized.
* Initial repository structure created.
* Core technologies identified.

 In Progress

* Dataset collection.
* Whisper speech-to-text implementation.
* Speaker diarization research.

Team

Add the names and GitHub profiles of your team members here.
