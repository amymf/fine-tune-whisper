from datasets import load_dataset
from collections import defaultdict
import numpy as np
import soundfile as sf
import os
import json


dataset = load_dataset("edinburghcstr/ami", "ihm")
ds = dataset["test"]

# Group utterances by meeting
meeting_groups = defaultdict(list)
for row in ds:
    meeting_groups[row["meeting_id"]].append(row)

def create_sequences(
    utterances, max_speakers=2, max_utts_per_speaker=3, max_duration=15.0
):
    sequences = []
    current_seq = []
    speaker_counts = defaultdict(int)
    duration = 0.0

    for utt in utterances:
        speaker = utt["speaker_id"]
        dur = utt["end_time"] - utt["begin_time"]

        if (
            (
                speaker not in speaker_counts.keys()
                or speaker_counts[speaker] < max_utts_per_speaker
            )
            and len(speaker_counts) < max_speakers
            and duration + dur <= max_duration
        ):
            current_seq.append(utt)
            speaker_counts[speaker] += 1
            duration += dur
            continue

        # If we get here, current utt did not fit - store current sequence
        if len(current_seq) > 0:
            sequences.append(current_seq)
        # Start a new sequence with the one that didn't fit
        current_seq = [utt]
        speaker_counts = defaultdict(int)
        speaker_counts[speaker] = 1
        duration = dur

    # Store the last sequence if it has any utterances
    if len(current_seq) > 0:
        sequences.append(current_seq)

    return sequences


all_sequences = []
for meeting_id, utterances in meeting_groups.items():
    utterances.sort(key=lambda x: x["begin_time"])
    sequences = create_sequences(utterances)
    all_sequences.extend(sequences)

def build_transcript(seq):
    speaker_map = {}
    current_index = 1

    for utt in seq:
        spk = utt["speaker_id"]
        if spk not in speaker_map:
            speaker_map[spk] = f"[SPK{current_index}]"
            current_index += 1

    lines = [f"{speaker_map[utt['speaker_id']]} {utt['text']}" for utt in seq]
    return " ".join(lines)

combined_sequences = []
for i, seq in enumerate(all_sequences):
    audio = []
    sr = seq[0]["audio"]["sampling_rate"] # same for all utterances
    for utt in seq:
        audio.append(utt["audio"]["array"])
    audio = np.concatenate(audio)
    text = build_transcript(seq)
    combined_sequences.append({ "audio": { "array": audio, "sampling_rate": sr }, "text": text })



output_dir = "test-data/audio_sequences"
os.makedirs(output_dir, exist_ok=True)

manifest = []

for i, sample in enumerate(combined_sequences):
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    text = sample["text"]

    audio_path = os.path.join(output_dir, f"seq_{i}.wav")
    
    # Save as 16-bit PCM WAV
    sf.write(audio_path, audio_array, samplerate=sampling_rate, subtype='PCM_16')

    # Add to manifest
    manifest.append({
        "audio_path": audio_path,
        "text": text
    })

# Save manifest
with open("test-data/manifest.jsonl", "w") as f:
    for entry in manifest:
        f.write(json.dumps(entry) + "\n")
