import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import matplotlib.pyplot as plt
from dataset import AMIDataset
from finetune import collate_fn

tokenizer = WhisperTokenizer.from_pretrained("amyf/whisper-finetuned-ami-160525")
model = WhisperForConditionalGeneration.from_pretrained(
    "amyf/whisper-finetuned-ami-160525"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = AMIDataset("test-data/manifest.jsonl", tokenizer)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

max_len = 448
plotted = 0

model.eval()

spk2_attentions = []

for audio_batch, _, targets, paths in dataloader:
    for i in range(len(audio_batch)):
        mel = audio_batch[i].unsqueeze(0).to(device)
        generated = [tokenizer.bos_token_id]
        spk2_indices = []
        spk2_attn_vectors = []

        for step in range(max_len):
            with torch.no_grad():
                input_ids = torch.tensor([generated], device=device)
                outputs = model(
                    input_features=mel,
                    decoder_input_ids=input_ids,
                    output_attentions=True,
                    return_dict=True,
                )
                cross_attn = outputs.cross_attentions  # tuple: layers x heads x tgt_len x src_len
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                # Check if next token is SPK2
                if next_token == tokenizer.convert_tokens_to_ids("[SPK2]"):
                    # Average attention over layers and heads for the current decoding step
                    # cross_attn is tuple of length num_layers, each is tensor (batch=1, heads, tgt_len, src_len)
                    # We want the last token's attention: tgt_len = step + 1, so index -1
                    attn_step = torch.stack([layer[0, :, -1, :] for layer in cross_attn])  # shape: layers x heads x src_len
                    avg_attn = attn_step.mean(dim=(0,1)).cpu()  # avg over layers and heads, shape: src_len
                    spk2_attn_vectors.append(avg_attn)

                if next_token == tokenizer.eos_token_id:
                    break

        if spk2_attn_vectors:
            # Sum or average if multiple SPK2 tokens in one sample
            # Here just take the first occurrence for simplicity
            spk2_attentions.append(spk2_attn_vectors[0].numpy())

        if len(spk2_attentions) >= 10: 
            break
    break

# Plot all SPK2 attention curves overlaid
plt.figure(figsize=(14,6))
for i, attn_vec in enumerate(spk2_attentions):
    plt.plot(attn_vec, label=f"Sample {i+1}")
plt.title("Cross-attention weights over encoder frames at first [SPK2] token generation")
plt.xlabel("Encoder time steps (audio frames)")
plt.ylabel("Attention weight")
plt.legend()
plt.grid(True)
plt.show()
