from huggingface_hub import snapshot_download
import os

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
exclude_files = ["v1-5-pruned-emaonly.ckpt", "v1-5-pruned-emaonly.safetensors",
                 "v1-5-pruned.ckpt", "v1-5-pruned.safetensors",
                 "model.safetensors", "pytorch_model.bin", "pytorch_model.fp16.bin",
                 "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.bin",
                 "diffusion_pytorch_model.bin", "diffusion_pytorch_model.non_ema.safetensors",
                 "diffusion_pytorch_model.non_ema.bin"]

local_dir = "D:/TJU/3.2/IntelliAnalyze/S2-SR/models/stable-diffusion-v1.5"

# 下载除了指定文件以外的所有文件
snapshot_download(
    repo_id,
    local_dir=local_dir,
    ignore_patterns=exclude_files,
    resume_download=True
)