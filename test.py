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
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

model.eval()

max_len = 448 # whisper max length

for audio_batch, _, targets, paths in dataloader:
    for i in range(len(audio_batch)):
        mel = audio_batch[i].unsqueeze(0).to(device)
        generated = [tokenizer.bos_token_id]
        for _ in range(max_len):
            with torch.no_grad():
                input_ids = torch.tensor([generated], device=device)
                outputs = model(input_features=mel, decoder_input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break
        original_text = tokenizer.decode(targets[i], skip_special_tokens=False)
        generated_text = tokenizer.decode(generated[1:], skip_special_tokens=False)
        print(f"Original text: {original_text}")
        print(f"Generated text: {generated_text}")
        # print(f"Audio path: {paths[i]}")
        print("-" * 100)