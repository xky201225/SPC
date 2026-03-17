import os
from huggingface_hub import snapshot_download

def download_tokenizer():
    base_dir = "checkpoints"
    os.makedirs(base_dir, exist_ok=True)
    
    print("Downloading Qwen2.5-7B-Instruct Tokenizer...")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_dir=os.path.join(base_dir, "Qwen2.5-7B-Instruct"),
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"], # 忽略所有大权重文件，只下载 Tokenizer 配置
        local_dir_use_symlinks=False
    )
    print("Tokenizer download complete! Saved to checkpoints/Qwen2.5-7B-Instruct")

if __name__ == "__main__":
    download_tokenizer()