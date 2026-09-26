from pydub import AudioSegment
from pathlib import Path


def preprocess_audio(input_file, output_file):

    print("Loading audio...")

    audio = AudioSegment.from_file(input_file)

    print("Converting to mono...")

    audio = audio.set_channels(1)

    print("Setting sample rate to 16 kHz...")

    audio = audio.set_frame_rate(16000)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    audio.export(output_file, format="wav")

    print("Audio preprocessing completed!")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":

    input_file = "data/raw_audio/meeting_01.mpeg"
    output_file = "data/preprocessed_audio/meeting_01.wav"

    preprocess_audio(input_file, output_file)