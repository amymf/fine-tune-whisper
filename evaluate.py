import torch
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torchaudio
import whisper
from finetune import pad_or_trim
from pydub import AudioSegment

# Load model and tokenizer
tokenizer = WhisperTokenizer.from_pretrained("amyf/whisper-finetuned-ami-160525")
model = WhisperForConditionalGeneration.from_pretrained(
    "amyf/whisper-finetuned-ami-160525"
)

old_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
old_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

audio_path = "audio-1.wav"
# audio = AudioSegment.from_file(audio_path)
# audio = audio.set_channels(1)  # Convert to mono
# audio.set_frame_rate(16000)  # Set to 16kHz
# audio.export(audio_path, format="wav")
# save .wav file
# audio.export("converted.wav", format="wav")

waveform, sr = torchaudio.load(
    audio_path
)  # shape [channels, samples] - assume mono (1 channel)
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

generated = [tokenizer.bos_token_id]
for _ in range(448):
    with torch.no_grad():
        input_ids = torch.tensor([generated], device=device)
        outputs = model(input_features=mel, decoder_input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
generated_text = tokenizer.decode(generated[1:], skip_special_tokens=False)

# Compare with old model
generated = [
    old_tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    old_tokenizer.convert_tokens_to_ids("<|en|>"),
    old_tokenizer.convert_tokens_to_ids("<|transcribe|>"),
    old_tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
]
for i in range(448):
    with torch.no_grad():
        input_ids = torch.tensor([generated], device=device)
        outputs = old_model(input_features=mel, decoder_input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == old_tokenizer.eos_token_id:
            break
        generated.append(next_token)

original_text = old_tokenizer.decode(generated[1:], skip_special_tokens=False)
print(f"Old model prediction: {original_text}")
print(f"New model prediction: {generated_text}")
