import json
import re
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "speaker_diarized_transcripts"
    / "meeting_01_speaker_transcript.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "meeting_minutes"


# --------------------------------------------------
# LOAD SPEAKER-DIARIZED TRANSCRIPT
# --------------------------------------------------

def load_transcript(file_path):
    """Load speaker-labelled transcript segments from JSON."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {file_path}\n"
            "Check that Member 2's JSON file is in the correct folder."
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a list of segments or a dictionary containing them.
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        segments = data.get(
            "segments",
            data.get("transcript", data.get("results", []))
        )
    else:
        raise ValueError("Unexpected transcript JSON format.")

    if not isinstance(segments, list):
        raise ValueError(
            "Transcript segments must be stored as a JSON list."
        )

    cleaned_segments = []

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        cleaned_segments.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "timestamp": segment.get("timestamp", ""),
            "speaker": segment.get("speaker", "Unknown speaker"),
            "text": text,
        })

    if not cleaned_segments:
        raise ValueError(
            "No transcript segments were found in the input JSON."
        )

    return cleaned_segments


# --------------------------------------------------
# CLEAN AND SPLIT TRANSCRIPT TEXT
# --------------------------------------------------

def clean_text(text):
    """Normalize whitespace and remove common transcription artifacts."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    """Split transcript text into sentence-like units."""

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# BUILD SENTENCE RECORDS
# --------------------------------------------------

def build_sentence_records(segments):
    """Preserve speaker information when splitting sentences."""

    records = []

    for segment in segments:
        text = clean_text(segment["text"])

        for sentence in split_sentences(text):
            records.append({
                "speaker": segment["speaker"],
                "text": sentence,
                "start": segment["start"],
                "end": segment["end"],
                "timestamp": segment["timestamp"],
            })

    return records


# --------------------------------------------------
# GENERATE MEETING MINUTES
# --------------------------------------------------

def generate_minutes(segments):
    """Generate rule-based meeting minutes from transcript segments."""

    records = build_sentence_records(segments)

    # Exclude greetings and very short conversational fragments
    # from the main summary and key-point selection.
    greeting_patterns = [
        r"^(good morning|good afternoon|good evening)[,.! ]*$",
        r"^(hello|hi|hey)[,.! ]*$",
        r"^(great|perfect|exactly|agreed|sounds good)[,.! ]*$",
    ]

    def is_greeting(text):
        return any(
            re.match(pattern, text.lower())
            for pattern in greeting_patterns
        )

    useful = [
        item for item in records
        if len(item["text"].split()) >= 4
        and not is_greeting(item["text"])
    ]

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    summary_terms = (
        "completed",
        "successfully generated",
        "calculate the word error rate",
        "accuracy",
        "integrate",
        "identify the different speakers",
        "start working",
        "initial implementation",
        "backend",
        "fronted",
        "frontend",
        "sample meeting recording",
        "testing the complete pipeline",
        "test transcription first",
        "complete prototype",
        "send you the diarization",
        "reference minutes",
        "summarization",
    )

    summary_candidates = [
        item["text"]
        for item in useful
        if any(
            term in item["text"].lower()
            for term in summary_terms
        )
    ]

    # Remove duplicate sentences while keeping their original order.
    summary_candidates = list(dict.fromkeys(summary_candidates))

    summary = " ".join(summary_candidates[:5])

    if not summary:
        summary = " ".join(
            item["text"] for item in useful[:3]
        )

    # --------------------------------------------------
    # KEY POINTS
    # --------------------------------------------------

    key_point_terms = (
        "completed",
        "tested",
        "transcript",
        "accuracy",
        "word error rate",
        "diarization",
        "speaker",
        "integrate",
        "backend",
        "frontend",
        "recording",
        "testing",
        "prototype",
        "reference minutes",
        "summarization",
    )

    key_points = []

    for item in useful:
        sentence = item["text"]

        if any(
            term in sentence.lower()
            for term in key_point_terms
        ):
            if sentence not in key_points:
                key_points.append(sentence)

    # --------------------------------------------------
    # DECISIONS
    # --------------------------------------------------

    decision_pattern = re.compile(
        r"\b("
        r"decided to|decision is|agreed to|approved|selected|chose|"
        r"will use|let's target|target the|"
        r"we will use|we agreed"
        r")\b",
        re.IGNORECASE,
    )

    decisions = []

    for item in useful:
        sentence = item["text"]

        if decision_pattern.search(sentence):
            if sentence not in decisions:
                decisions.append(sentence)

    # --------------------------------------------------
    # ACTION ITEMS
    # --------------------------------------------------

    action_pattern = re.compile(
        r"\b("
        r"i will|i'll|we will|we'll|need to|needs to|must|"
        r"assigned to|responsible for|should prioritize|"
        r"let's create|let's target|start working|"
        r"complete the initial implementation|"
        r"calculate the word error rate|"
        r"send you|record one sample|"
        r"test transcription"
        r")\b",
        re.IGNORECASE,
    )

    deadline_pattern = re.compile(
        r"\b("
        r"today|tomorrow|"
        r"by monday|by tuesday|by wednesday|by thursday|"
        r"by friday|by saturday|by sunday|"
        r"next week|by next week"
        r")\b",
        re.IGNORECASE,
    )

    action_items = []

    for item in useful:
        sentence = item["text"]

        if not action_pattern.search(sentence):
            continue

        deadline_match = deadline_pattern.search(sentence)

        deadline = (
            deadline_match.group(0)
            if deadline_match
            else "Not identified"
        )

        # Attribute the task to the diarized speaker only when
        # the sentence explicitly uses first-person commitment.
        explicit_owner = re.search(
            r"\b(i will|i'll)\b",
            sentence,
            re.IGNORECASE,
        )

        owner = (
            item["speaker"]
            if explicit_owner
            else "Not identified"
        )

        action = {
            "task": sentence,
            "responsible_person": owner,
            "deadline": deadline,
        }

        # Avoid duplicate action items.
        if action not in action_items:
            action_items.append(action)

    # --------------------------------------------------
    # SPEAKER CONTRIBUTIONS
    # --------------------------------------------------

    speaker_contributions = []

    for item in records:
        speaker = item["speaker"]
        sentence = item["text"]

        if sentence:
            speaker_contributions.append({
                "speaker": speaker,
                "text": sentence,
            })

    # --------------------------------------------------
    # RETURN MINUTES
    # --------------------------------------------------

    return {
        "summary": summary,
        "key_points": key_points,
        "decisions": decisions,
        "action_items": action_items,
        "speaker_contributions": speaker_contributions,
        "note": (
            "Rule-based draft. Verify the summary, decisions, "
            "task ownership, and deadlines against the original "
            "transcript. Speaker labels may contain diarization errors."
        ),
    }


# --------------------------------------------------
# SAVE OUTPUT FILES
# --------------------------------------------------

def save_minutes(minutes, output_dir):
    """Save meeting minutes as JSON and readable TXT files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "meeting_01_minutes.json"
    text_path = output_dir / "meeting_01_minutes.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(minutes, file, indent=4, ensure_ascii=False)

    with text_path.open("w", encoding="utf-8") as file:
        file.write("MEETING MINUTES\n")
        file.write("=" * 40 + "\n\n")

        file.write("SUMMARY\n")
        file.write(minutes["summary"] + "\n\n")

        file.write("KEY POINTS\n")
        if minutes["key_points"]:
            for point in minutes["key_points"]:
                file.write(f"- {point}\n")
        else:
            file.write("- No key points identified.\n")

        file.write("\nDECISIONS\n")
        if minutes["decisions"]:
            for decision in minutes["decisions"]:
                file.write(f"- {decision}\n")
        else:
            file.write("- No explicit decisions identified.\n")

        file.write("\nACTION ITEMS\n")
        if minutes["action_items"]:
            for item in minutes["action_items"]:
                file.write(f"- Task: {item['task']}\n")
                file.write(
                    f"  Responsible person: "
                    f"{item['responsible_person']}\n"
                )
                file.write(f"  Deadline: {item['deadline']}\n")
        else:
            file.write("- No action items identified.\n")

        file.write("\nSPEAKER CONTRIBUTIONS\n")
        if minutes["speaker_contributions"]:
            for item in minutes["speaker_contributions"]:
                file.write(
                    f"- {item['speaker']}: {item['text']}\n"
                )
        else:
            file.write("- No speaker contributions found.\n")

        file.write("\nNOTE\n")
        file.write(minutes["note"] + "\n")

    return json_path, text_path


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Loading speaker-diarized transcript...")

    segments = load_transcript(INPUT_FILE)

    print(f"Loaded {len(segments)} transcript segments.")

    print("Generating meeting minutes...")

    minutes = generate_minutes(segments)

    json_path, text_path = save_minutes(
        minutes,
        OUTPUT_DIR,
    )

    print("\nMeeting minutes generated successfully.")
    print(f"JSON output: {json_path}")
    print(f"Text output: {text_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"\nERROR: {error}")