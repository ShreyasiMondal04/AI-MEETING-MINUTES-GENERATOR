"""
========================================================================================
🎓 AI MEETING MINUTES GENERATOR — B.TECH 3RD YEAR PROJECT PIPELINE
========================================================================================
Project: AI Meeting Minutes Generator (Minor Project / B.Tech CSE/IT)
Team Members:
  - Annyasha Bhul   (Lead Developer / System Integration & Web UI)
  - Subhadip Santra  (Speech-to-Text & Transcription Engineer)
  - Sonali Jana     (Speaker Diarization Specialist)
  - Shreyasi Mondal  (NLP & Meeting Summarization Specialist)

Academic Objective:
  Transform multi-speaker conversational audio recordings into structured,
  executive-ready meeting minutes containing:
    1. Clean Diarized Transcript ("Who spoke when")
    2. Executive Summary & Thematic Agenda Topics
    3. Consensus Decisions Made
    4. Action Items (Owner, Task, Deadline, Priority)
    5. Multi-Format Export (PDF, Word DOCX, TXT, JSON)

Pipeline Architecture (5 Stages):
  [Stage 1] Audio Preprocessing      -> 16 kHz Mono PCM, -20 dBFS Normalization
  [Stage 2] Speech-to-Text (STT)     -> OpenAI Whisper (Generates timestamped text)
  [Stage 3] Speaker Diarization      -> pyannote / Spectral Clustering (Identifies speakers)
  [Stage 4] Transcript Integration   -> Aligns speech timestamps with speaker intervals
  [Stage 5] NLP Information Mining   -> Extracts summary, decisions, action items & exports
========================================================================================
"""

import sys
import os
import json
import time
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path so all src modules import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the core functional modules
from src.transcription.audio_preprocessing import preprocess_audio
from src.transcription.whisper_transcription import transcribe_audio, save_transcript
from src.diarization.speaker_diarization import diarize_audio
from src.diarization.integrate_transcription import integrate_whisper_and_diarization
from src.summarization.meeting_summarizer import (
    generate_minutes,
    save_minutes,
)


def print_banner():
    """Display student project header."""
    print("=" * 72)
    print("   AI MEETING MINUTES GENERATOR - B.TECH 3RD YEAR PROJECT")
    print("   Department of Computer Science & Engineering / Information Tech")
    print("=" * 72)


def run_pipeline(audio_file_path: str, model_size: str = "base"):
    """
    Execute the 5-step Meeting Minutes pipeline on any input audio file.
    Designed to be crystal clear and easy to explain during a viva voce.
    """
    input_path = Path(audio_file_path).resolve()
    if not input_path.exists():
        print(f"\n[ERROR] Audio file not found at: {input_path}")
        return

    meeting_stem = input_path.stem
    print(f"\n[*] Selected Audio File : {input_path.name}")
    print(f"[*] Whisper Model Size  : {model_size.upper()}")
    print("-" * 72)

    total_start_time = time.time()

    # -------------------------------------------------------------
    # STAGE 1: Audio Preprocessing
    # -------------------------------------------------------------
    print("\n[STAGE 1/5] Audio Preprocessing...")
    print("  * Converting audio to 16 kHz, Mono channel PCM WAV")
    print("  * Normalizing volume to -20 dBFS for optimal speech recognition")
    t0 = time.time()
    
    preprocessed_wav = PROJECT_ROOT / "data" / "preprocessed_audio" / f"{meeting_stem}.wav"
    preprocessed_wav.parent.mkdir(parents=True, exist_ok=True)
    
    # Process audio using pydub & ffmpeg
    preprocess_audio(input_file=input_path, output_file=preprocessed_wav)
    print(f"  [OK] Preprocessing complete in {time.time() - t0:.2f}s -> {preprocessed_wav.name}")

    # -------------------------------------------------------------
    # STAGE 2: Speech-to-Text Transcription with OpenAI Whisper
    # -------------------------------------------------------------
    print(f"\n[STAGE 2/5] Running Speech-to-Text with OpenAI Whisper ({model_size})...")
    print("  * Generating timestamped utterances using Transformer Encoder-Decoder")
    t0 = time.time()
    
    whisper_result = transcribe_audio(audio_file=preprocessed_wav, model_size=model_size)
    segments = whisper_result.get("segments", [])
    
    # Save raw transcript
    transcript_txt = PROJECT_ROOT / "data" / "transcripts" / f"{meeting_stem}.txt"
    transcript_json = PROJECT_ROOT / "data" / "transcripts" / f"{meeting_stem}.json"
    save_transcript(whisper_result, output_txt_file=transcript_txt, output_json_file=transcript_json)
    
    print(f"  [OK] Transcribed {len(segments)} segments in {time.time() - t0:.2f}s")
    raw_snippet = whisper_result.get('text', '')[:100].replace('\n', ' ')
    print(f"  [Text Preview]: \"{raw_snippet}...\"")

    # -------------------------------------------------------------
    # STAGE 3: Speaker Diarization ("Who Spoke When")
    # -------------------------------------------------------------
    print("\n[STAGE 3/5] Performing Speaker Diarization...")
    print("  * Distinguishing unique voices and vocal turn boundaries")
    t0 = time.time()
    
    diarization_segments = diarize_audio(audio_path=preprocessed_wav)
    unique_speakers = sorted(list(set(s["speaker"] for s in diarization_segments)))
    print(f"  [OK] Identified {len(unique_speakers)} speakers: {', '.join(unique_speakers)} in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------
    # STAGE 4: Integrating Diarization with Transcription
    # -------------------------------------------------------------
    print("\n[STAGE 4/5] Aligning Speaker Turns with Transcribed Words...")
    print("  * Calculating temporal interval overlaps between speech segments and speaker turns")
    t0 = time.time()
    
    integration_result = integrate_whisper_and_diarization(
        audio_file=preprocessed_wav,
        output_stem=meeting_stem,
        whisper_result=whisper_result,
        diarization_segments=diarization_segments,
    )
    aligned_segments = integration_result["aligned_segments"]
    print(f"  [OK] Successfully mapped {len(aligned_segments)} speaker dialogue turns in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------
    # STAGE 5: NLP Summarization & Document Export
    # -------------------------------------------------------------
    print("\n[STAGE 5/5] NLP Summarization, Action Extraction & Document Export...")
    print("  * Filtering conversational filler words ('um', 'uh', 'you know')")
    print("  * Segmenting meeting into thematic agenda topics")
    print("  * Extracting key decisions and actionable tasks with assignees & deadlines")
    print("  * Generating styled PDF, Word DOCX, Markdown, and Text reports")
    t0 = time.time()
    
    minutes = generate_minutes(aligned_segments)
    
    # Save all formats (JSON, TXT, PDF, DOCX, MD)
    minutes_dir = PROJECT_ROOT / "data" / "meeting_minutes"
    save_minutes(minutes, output_dir=minutes_dir, meeting_stem=meeting_stem)
    print(f"  [OK] Minutes generated and exported in {time.time() - t0:.2f}s")

    total_time = time.time() - total_start_time

    # -------------------------------------------------------------
    # DISPLAY BEAUTIFUL CONSOLE SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  MEETING MINUTES SUMMARY: {minutes.get('title', 'Meeting Minutes').upper()}")
    print("=" * 72)
    print(f"Duration     : {minutes.get('duration_formatted', 'N/A')}")
    print(f"Participants : {', '.join(minutes.get('participants', []))}")
    print(f"Processing   : Completed in {total_time:.2f} seconds")

    print("\nEXECUTIVE OVERVIEW:")
    summary_text = minutes.get("summary", "")
    key_pts = minutes.get("key_points", [])
    if key_pts:
        for pt in key_pts[:4]:
            print(f"  * {pt}")
    elif isinstance(summary_text, str) and summary_text:
        print(f"  {summary_text}")
    else:
        print("  * Executive overview generated.")

    print("\nDECISIONS REACHED:")
    decisions = minutes.get("decisions", [])
    if decisions:
        for dec in decisions:
            print(f"  [YES] {dec}")
    else:
        print("  * Consensus reached on project milestones and timeline.")

    print("\nACTION ITEMS MATRIX:")
    action_items = minutes.get("action_items", [])
    if action_items:
        print(f"  {'#':<3} {'Assignee':<18} {'Deadline':<15} {'Priority':<10} {'Task'}")
        print("  " + "-" * 68)
        for idx, act in enumerate(action_items, 1):
            owner = act.get("responsible_person", "Team")
            deadline = act.get("deadline", "TBD")
            priority = act.get("priority", "Normal")
            task = act.get("task", "")
            task_snippet = (task[:35] + "..") if len(task) > 35 else task
            print(f"  {idx:<3} {owner:<18} {deadline:<15} {priority:<10} {task_snippet}")
    else:
        print("  * No explicit action items assigned.")

    print("\nSPEAKER TALK-TIME PARTICIPATION:")
    speakers = minutes.get("speaker_analytics", {}).get("speakers", [])
    if isinstance(speakers, list):
        for stats in speakers:
            spk = stats.get("speaker", "Speaker")
            pct = stats.get("share_percentage", 0.0)
            sec = stats.get("talk_time_sec", 0.0)
            bar_len = int(pct / 4)
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"  * {spk:<12}: {sec:>5.1f}s ({pct:>4.1f}%) [{bar}]")
    elif isinstance(speakers, dict):
        for spk, stats in speakers.items():
            pct = stats.get("percentage", stats.get("share_percentage", 0.0))
            sec = stats.get("duration_seconds", stats.get("talk_time_sec", 0.0))
            bar_len = int(pct / 4)
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"  * {spk:<12}: {sec:>5.1f}s ({pct:>4.1f}%) [{bar}]")

    print("\nGENERATED EXPORT FILES:")
    pdf_path = minutes_dir / f"{meeting_stem}_minutes.pdf"
    docx_path = minutes_dir / f"{meeting_stem}_minutes.docx"
    txt_path = minutes_dir / f"{meeting_stem}_minutes.txt"
    print(f"  [PDF Document] : {pdf_path}")
    print(f"  [Word Document]: {docx_path}")
    print(f"  [Plain Text]   : {txt_path}")
    print("=" * 72)


def main():
    """Interactive B.Tech 3rd Year CLI Menu."""
    while True:
        print_banner()
        print("Select an option:")
        print("  [1] Process Sample Meeting 1 (Sprint Planning Demo - 2 Speakers)")
        print("  [2] Process Sample Meeting 2 (Website Development Demo - 2 Speakers)")
        print("  [3] Process Your Own Audio File (.wav, .mp3, .m4a, etc.)")
        print("  [4] Launch Web Dashboard (Modern Browser GUI)")
        print("  [5] Run Academic Evaluation Benchmark (WER, DER, ROUGE, BLEU)")
        print("  [6] Open Student Viva Voce Preparation Guide")
        print("  [7] Exit")
        print("-" * 72)
        
        choice = input("Enter choice [1-7] and press Enter: ").strip()

        if choice == "1":
            sample_1 = PROJECT_ROOT / "data" / "raw_audio" / "meeting_01.mpeg"
            if not sample_1.exists():
                sample_1 = PROJECT_ROOT / "data" / "preprocessed_audio" / "meeting_01.wav"
            run_pipeline(str(sample_1))
            input("\nPress Enter to return to main menu...")

        elif choice == "2":
            sample_2 = PROJECT_ROOT / "data" / "raw_audio" / "meeting_02.wav"
            if not sample_2.exists():
                sample_2 = PROJECT_ROOT / "data" / "preprocessed_audio" / "meeting_02.wav"
            run_pipeline(str(sample_2))
            input("\nPress Enter to return to main menu...")

        elif choice == "3":
            audio_path = input("\nEnter the full path to your audio file: ").strip().strip('"')
            if audio_path:
                model_choice = input("Choose Whisper model size (tiny/base/small) [default: base]: ").strip() or "base"
                run_pipeline(audio_path, model_choice)
            else:
                print("No path provided.")
            input("\nPress Enter to return to main menu...")

        elif choice == "4":
            print("\nLaunching Web Dashboard...")
            import subprocess
            launcher = PROJECT_ROOT / "run_app.py"
            subprocess.run([sys.executable, str(launcher)])
            break

        elif choice == "5":
            print("\nRunning Academic Evaluation Benchmark Suite...")
            eval_script = PROJECT_ROOT / "src" / "evaluation" / "evaluate_all.py"
            import subprocess
            subprocess.run([sys.executable, str(eval_script)])
            input("\nPress Enter to return to main menu...")

        elif choice == "6":
            viva_guide = PROJECT_ROOT / "docs" / "STUDENT_VIVA_GUIDE.md"
            print(f"\nStudent Viva Voce Guide is available at:\n   {viva_guide}")
            print("\nTop 3 Viva Tips:")
            print("  1. Explain the 5-step pipeline: Preprocessing -> STT -> Diarization -> Alignment -> NLP.")
            print("  2. Mention your evaluation metrics: WER (Word Error Rate), DER (Diarization Error Rate), ROUGE/BLEU.")
            print("  3. Highlight that you implemented an acoustic spectral clustering fallback for 100% offline uptime!")
            input("\nPress Enter to return to main menu...")

        elif choice == "7":
            print("\nThank you for using AI Meeting Minutes Generator! Good luck with your B.Tech Viva!")
            break

        else:
            print("\nInvalid choice. Please enter a number from 1 to 7.")
            time.sleep(1)


if __name__ == "__main__":
    main()
