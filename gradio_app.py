import gradio as gr
import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torchaudio
import whisper 
from finetune import pad_or_trim

# Load tokenizer and model
tokenizer = WhisperTokenizer.from_pretrained("amyf/whisper-finetuned-ami-160525")
model = WhisperForConditionalGeneration.from_pretrained("amyf/whisper-finetuned-ami-160525")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocess a single audio file into a batch of input features
def preprocess(audio_path):
    waveform, sr = torchaudio.load(audio_path) # shape [channels, samples] - assume mono (1 channel)
    # If stereo, convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0).to(device)  # [samples]
    mel = whisper.log_mel_spectrogram(waveform)  # [n_mel, num_frames]
    mel = pad_or_trim(mel, target_length=3000)  
    mel = mel.unsqueeze(0)  # [1, n_mel, num_frames]
    mel = mel.to(device)
    return mel

# Generate transcription from audio input
def transcribe(audio_path):
    input_features = preprocess(audio_path)
    batch_size = 1
    generated = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_len = 448

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(input_features=input_features, decoder_input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab]
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [1]

        next_tokens[finished] = tokenizer.pad_token_id
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        finished |= next_tokens == tokenizer.eos_token_id
        if finished.all():
            break

    decoded = generated[0, 1:].tolist()
    if tokenizer.eos_token_id in decoded:
        decoded = decoded[:decoded.index(tokenizer.eos_token_id)]

    return tokenizer.decode(decoded, skip_special_tokens=False)

# Gradio interface
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs="text",
    title="Whisper AMI Fine-Tuned Transcriber"
).launch()
