# 📝 Whiteboard Code Cheatsheet for B.Tech Viva Voce
## AI Meeting Minutes Generator — Core Algorithms to Write on Board/Paper

Examiners frequently ask students to write short snippets on a blackboard or sheet of paper to prove they understand the underlying Python logic. Here are the 4 most commonly requested snippets:

---

### Snippet 1: Audio Preprocessing with `pydub` (10 Lines)
> **Examiner:** *"Show me how you convert any incoming audio into 16 kHz Mono PCM WAV."*

```python
from pydub import AudioSegment

def preprocess_audio(input_file, output_wav):
    # 1. Load audio in any format (WAV, MP3, MPEG, M4A)
    audio = AudioSegment.from_file(input_file)
    
    # 2. Convert to Mono (1 channel) and 16000 Hz sample rate
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # 3. Volume Normalization to -20 dBFS
    target_dbfs = -20.0
    gain_change = target_dbfs - audio.dBFS
    normalized_audio = audio.apply_gain(gain_change)
    
    # 4. Export as uncompressed 16-bit PCM WAV
    normalized_audio.export(output_wav, format="wav")
    return output_wav
```

---

### Snippet 2: OpenAI Whisper Transcription (10 Lines)
> **Examiner:** *"Write down the Python code to run Whisper on GPU/CPU and get timestamps."*

```python
import whisper
import torch

def transcribe(audio_path, model_size="base"):
    # 1. Auto-detect CUDA GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load model into memory
    model = whisper.load_model(model_size, device=device)
    
    # 3. Transcribe audio with timestamped segments
    result = model.transcribe(str(audio_path), language="en", fp16=(device == "cuda"))
    
    # 4. Loop over segments
    for segment in result["segments"]:
        start, end, text = segment["start"], segment["end"], segment["text"]
        print(f"[{start:.2f}s - {end:.2f}s]: {text}")
        
    return result
```

---

### Snippet 3: Aligning Whisper Words with Diarization Turns (Interval Overlap)
> **Examiner:** *"How do you combine who spoke with what was said mathematically?"*

```python
def assign_speaker_to_segment(whisper_start, whisper_end, diarization_turns):
    """
    Finds the speaker with maximum temporal intersection duration.
    """
    best_speaker = "Unknown"
    max_overlap = 0.0
    
    for turn in diarization_turns:
        # Calculate intersection between [whisper_start, whisper_end] and [turn.start, turn.end]
        overlap_start = max(whisper_start, turn["start"])
        overlap_end = min(whisper_end, turn["end"])
        overlap_duration = max(0.0, overlap_end - overlap_start)
        
        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = turn["speaker"]
            
    return best_speaker
```

---

### Snippet 4: Rule-Based Action Item Extraction (NLP)
> **Examiner:** *"How do you extract action items, assignees, and deadlines without high-cost cloud LLMs?"*

```python
import re

def extract_action_items(transcript_sentences):
    action_items = []
    # Keywords indicating future task commitment
    commitment_pattern = r"\b(will|need to|must|should|going to|plan to)\b"
    # Prepositions indicating deadlines
    deadline_pattern = r"\b(by Friday|by Monday|tomorrow|next week|today)\b"
    
    for item in transcript_sentences:
        text = item["text"]
        speaker = item["speaker"]
        
        if re.search(commitment_pattern, text, re.IGNORECASE):
            deadline_match = re.search(deadline_pattern, text, re.IGNORECASE)
            deadline = deadline_match.group(0) if deadline_match else "Not specified"
            
            # Assign owner: if 1st person commitment ("I will..."), assign to current speaker
            owner = speaker if re.search(r"\b(i will|i'll|i can)\b", text, re.IGNORECASE) else "Team"
            
            action_items.append({
                "task": text,
                "owner": owner,
                "deadline": deadline
            })
            
    return action_items
```

---

### Snippet 5: Word Error Rate (WER) Calculation Formula
> **Examiner:** *"What is the mathematical formula for Word Error Rate?"*

$$\text{WER} = \frac{S + D + I}{N}$$

Where:
- $S$ = Number of **Substitutions** (words wrongly transcribed, e.g. "car" instead of "bar")
- $D$ = Number of **Deletions** (words omitted by the model)
- $I$ = Number of **Insertions** (words added that were not spoken)
- $N$ = Total number of words in the reference ground-truth text ($N = S + D + C$, where $C$ is correct words).
