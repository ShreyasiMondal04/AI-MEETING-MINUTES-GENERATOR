import whisper
from pathlib import Path


def transcribe_audio(audio_file):
    print("Loading Whisper model...")

    model = whisper.load_model("base")

    print("Transcribing audio...")

    result = model.transcribe(
        audio_file,
        language="en"
    )

    return result


def save_transcript(result, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:

        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            file.write(
                f"[{start:.2f} - {end:.2f}] {text}\n"
            )


if __name__ == "__main__":

    audio_file = "data/preprocessed_audio/meeting_01.wav"

    output_file = "data/transcripts/meeting_01.txt"

    result = transcribe_audio(audio_file)

    save_transcript(result, output_file)

    print("Transcription completed!")
    print(f"Transcript saved to: {output_file}")
    