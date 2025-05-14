from huggingface_hub import snapshot_download

snapshot_dir = snapshot_download("amyf/whisper-fine-tune-ami", repo_type="dataset", local_dir="data", local_dir_use_symlinks=False)
