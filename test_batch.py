import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from dataset import AMIDataset
from finetune import collate_fn

# Load model and tokenizer
tokenizer = WhisperTokenizer.from_pretrained("amyf/whisper-fine-tune-ami")
model = WhisperForConditionalGeneration.from_pretrained("amyf/whisper-fine-tune-ami")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = AMIDataset("test-data/manifest.jsonl", tokenizer)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

model.eval()

max_len = 20  # increase to 448 if on gpu

for audio_batch, _, targets, paths in dataloader:
    audio_batch = audio_batch.to(device)

    batch_size = audio_batch.size(0)
    generated = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(input_features=audio_batch, decoder_input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab]
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [B]

        next_tokens[finished] = tokenizer.pad_token_id  # Stop updating finished sequences
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        finished |= next_tokens == tokenizer.eos_token_id
        if finished.all():
            break

    # Decode and print results
    for i in range(batch_size):
        original_text = tokenizer.decode(targets[i], skip_special_tokens=False)
        # Remove BOS, then stop at first EOS
        gen = generated[i, 1:].tolist()
        if tokenizer.eos_token_id in gen:
            gen = gen[:gen.index(tokenizer.eos_token_id)]
        generated_text = tokenizer.decode(gen, skip_special_tokens=False)

        print(f"Original text: {original_text}")
        print(f"Generated text: {generated_text}")
        print(f"Audio path: {paths[i]}")
        print("-" * 100)
        break
