from jiwer import wer


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


reference_file = "data/reference_minutes/meeting_01.txt"
hypothesis_file = "data/transcripts/meeting_01.txt"

reference = read_file(reference_file)
hypothesis = read_file(hypothesis_file)

error = wer(reference, hypothesis)

print(f"WER: {error:.2%}")