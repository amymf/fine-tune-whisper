import json
import torchaudio
import torch


class AMIDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, tokenizer):
        self.entries = [json.loads(line) for line in open(manifest_path)]
        self.tokenizer = tokenizer
        self.sampling_rate = 16000

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform, _ = torchaudio.load(entry["audio_path"])
        tokens = self.tokenizer.encode(entry["text"])
        return {
            "audio": {
                "array": waveform.squeeze(0),
                "sampling_rate": self.sampling_rate,
            },  # [T]
            "text_tokens": torch.tensor(tokens, dtype=torch.long),
        }
