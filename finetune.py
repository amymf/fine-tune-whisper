import torch
import whisper
from dataset import AMIDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import wandb
import torch.nn.functional as F

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

new_tokens = ["[SPK1]", "[SPK2]"]
existing_tokens = tokenizer.additional_special_tokens or []

# Avoid duplicates
additional_tokens = list(set(existing_tokens + new_tokens))
tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
model.resize_token_embeddings(len(tokenizer))

model.to(device)

train_dataset = AMIDataset("train-data/manifest.jsonl", tokenizer)
val_dataset = AMIDataset("val-data/manifest.jsonl", tokenizer)

def pad_or_trim(mel: torch.Tensor, target_length=3000):
    if mel.size(-1) > target_length:
        mel = mel[:, :target_length]
    elif mel.size(-1) < target_length:
        mel = F.pad(mel, (0, target_length - mel.size(-1)))
    return mel


def collate_fn(batch):
    mels = []
    for item in batch:
        mel = whisper.log_mel_spectrogram(item["audio"]["array"])  # [n_mel, num_frames]
        mel = pad_or_trim(mel, target_length=3000) 
        mels.append(mel)

    audio_batch = torch.stack(mels)

    input_tokens = [item["text_tokens"][:-1] for item in batch]
    target_tokens = [item["text_tokens"][1:] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    input_tokens = pad_sequence(
        input_tokens, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=-100)

    return audio_batch, input_tokens, target_tokens, audio_paths


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

NUM_EPOCHS = 10

# freeze the encoder
for param in model.model.encoder.parameters():
    param.requires_grad = False


def train():
    model.train()
    train_loss = 0
    train_acc = 0
    for epoch in range(NUM_EPOCHS):
        # train loop
        for i, (audio, input, target) in enumerate(train_loader):
            audio = audio.to(device) # (batch_size, n_mel, num_frames)
            input = input.to(device) # (batch_size, seq_len)
            target = target.to(device) # (batch_size, seq_len)

            optimizer.zero_grad()
            logits = model(input_features=audio, decoder_input_ids=input).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # accuracy
            pred = logits.argmax(dim=-1)
            target = target.view(-1) # (batch_size * seq_len)
            pred = pred.view(-1) # (batch_size * seq_len)
            mask = target != -100 # ignore padding
            correct_tokens += (pred[mask] == target[mask]).sum().item()
            total_tokens += mask.sum().item()
            train_acc += correct_tokens / total_tokens

            torch.cuda.empty_cache()

        # val loop
        model.eval()
        val_loss = 0
        val_acc = 0
        for i, (audio, input, target) in enumerate(val_loader):
            audio = audio.to(device)
            input = input.to(device)
            target = target.to(device)

            with torch.no_grad():
                logits = model(input_features=audio, decoder_input_ids=input).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100
                )

                val_loss += loss.item()

                # accuracy
                pred = logits.argmax(dim=-1)
                target = target.view(-1)
                pred = pred.view(-1)
                mask = target != -100
                correct_tokens += (pred[mask] == target[mask]).sum().item()
                total_tokens += mask.sum().item()
                val_acc += correct_tokens / total_tokens
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_acc = 100 * correct_tokens / total_tokens
        val_loss /= len(val_loader)
        val_acc = 100 * correct_tokens / total_tokens
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        # Save the model every epoch
        wandb.save("model.pt")

    wandb.save("model.pt")
    torch.save(model.state_dict(), "model.pt")
    wandb.finish()


if __name__ == "__main__":
    wandb.init(project="whisper-finetune")
    train()
