import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import requests
from typing import List, Dict, Optional
import argparse


# =============================================
# 1. 数据集：支持 control_image + target_image + prompt
# =============================================

class SuperResolutionRefinementDataset(Dataset):
    """
    用于“语义引导细节增强”的数据集
    输入: HR_base（control image） + text_tags_prompt
    输出: HR_original（真实高清图）
    """

    def __init__(self,
                 control_dir: str,           # HAT-L 超分后的 HR_base 图像目录
                 target_dir: str,            # 原始高清图像目录（如 DIV2K HR）
                 prompt_file: str,
                 target_size: tuple = (512, 512),
                 transform=None):
        """
        Args:
            control_dir: 控制图目录（HR_base）
            target_dir: 目标图目录（HR_original）
            prompt_file: 标签文件，格式: filename,tag1,tag2,...
            target_size: 统一尺寸
            transform: 可选数据增强
        """
        self.control_dir = control_dir
        self.target_dir = target_dir
        self.target_size = target_size
        self.transform = transform or transforms.ToTensor()

        # 加载提示词
        self.prompts = self._load_prompts(prompt_file)

        # 获取图像列表（只保留 control_dir 和 target_dir 都存在的图像）
        self.image_files = []
        for f in os.listdir(control_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                target_path = os.path.join(target_dir, f)
                if os.path.exists(target_path):
                    self.image_files.append(f)

        print(f"加载了 {len(self.image_files)} 对训练图像 (control → target)")

    def _load_prompts(self, prompt_file: str) -> Dict[str, str]:
        prompts = {}
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        filename, prompt = parts
                        prompts[filename.strip()] = prompt.strip()
        return prompts

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        # 加载 control image (HR_base)
        control_path = os.path.join(self.control_dir, filename)
        control_img = Image.open(control_path).convert('RGB').resize(self.target_size)

        # 加载 target image (HR_original)
        target_path = os.path.join(self.target_dir, filename)
        target_img = Image.open(target_path).convert('RGB').resize(self.target_size)

        # 获取 prompt
        prompt = self.prompts.get(filename, "high quality, detailed")

        # 转换为 tensor
        if self.transform:
            control_tensor = self.transform(control_img)  # [3, H, W]
            target_tensor = self.transform(target_img)    # [3, H, W]

        return {
            'control_image': control_tensor,
            'target_image': target_tensor,
            'prompt': prompt,
            'filename': filename
        }


# =============================================
# 2. LoRA + ControlNet 训练器（核心修改）
# =============================================

class DetailEnhancementTrainer:
    """专用于“照片细节增强”的 ControlNet + LoRA 联合训练器"""

    def __init__(self,
                 base_model_path: str = "models/weights/stable-diffusion-v1-5",
                 output_dir: str = "checkpoints",
                 device: str = "cuda",
                 use_lora: bool = True):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.device = device
        self.use_lora = use_lora
        os.makedirs(output_dir, exist_ok=True)

        # 分别加载组件
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.controlnet = None

        self._load_models()
        self._setup_training_components()

    def _load_models(self):
        print("正在加载 Stable Diffusion 组件...")

        # 1. UNet（LoRA 微调）
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_model_path,
            subfolder="unet",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # 2. 使用 UNet 初始化 ControlNet（可训练）
        self.controlnet = ControlNetModel.from_unet(
            self.unet
        ).to(self.device)

        # 3. 其他（冻结）
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path,
            subfolder="vae",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.vae.requires_grad_(False)

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.text_encoder.requires_grad_(False)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer"
        )

        print("模型组件加载完成")

    def _setup_training_components(self):
        """设置可训练部分：ControlNet + UNet LoRA"""
        print("配置可训练模块...")

        # LoRA for UNet
        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"],  # SD1.5 UNet 注意力层
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()

        # ControlNet 全参数微调（或也可加 LoRA）
        self.controlnet.train()
        self.unet.train()

        print("训练组件配置完成")

    def _compute_loss(self, batch):
        """计算重建 + 感知损失"""
        control_images = batch['control_image'].to(self.device)  # [B, 3, H, W]
        target_images = batch['target_image'].to(self.device)    # [B, 3, H, W]
        prompts = batch['prompt']

        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # 编码文本
        with torch.no_grad():
            text_emb = self.text_encoder(text_inputs)[0]  # [B, 77, 768]

        # 编码图像
        with torch.no_grad():
            latents = self.vae.encode(target_images * 2 - 1).latent_dist.sample() * 0.18215  # [-4,4] → latent
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()

        # 添加噪声
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # ControlNet 输出
        down_samples, mid_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_emb,
            controlnet_cond=control_images,
            return_dict=False
        )

        # UNet 预测噪声
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample
        ).sample

        # 损失：预测噪声 vs 真实噪声
        loss = nn.MSELoss()(model_pred, noise)

        return loss

    def _add_noise(self, latents, noise, timesteps):
        """手动添加噪声（模拟扩散过程）"""
        scheduler = self._get_noise_scheduler()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents

    def _get_noise_scheduler(self):
        """使用简单的 DDPM 调度器"""
        from diffusers import DDPMScheduler
        return DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )

    def train(self,
              train_dataset,
              num_epochs=10,
              batch_size=1,
              learning_rate=1e-5,
              save_steps=500,
              gradient_accumulation_steps=4):

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # 优化器：只优化 ControlNet 和 LoRA
        optimizer = torch.optim.AdamW(
            [
                {'params': self.controlnet.parameters()},
                {'params': self.unet.parameters(), 'lr': learning_rate * 0.5}  # LoRA 学习率略低
            ],
            lr=learning_rate,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * num_epochs
        )

        global_step = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                loss = self._compute_loss(batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                if global_step % save_steps == 0:
                    self._save_checkpoint(global_step)

            print(f"Epoch {epoch + 1} 平均损失: {epoch_loss / len(train_loader):.4f}")

        self._save_final_model()

    def _save_checkpoint(self, step):
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)

        # 保存 ControlNet
        self.controlnet.save_pretrained(os.path.join(path, "controlnet"))

        # 保存 LoRA
        if self.use_lora:
            self.unet.save_pretrained(os.path.join(path, "lora_unet"))

        print(f"检查点已保存: {path}")

    def _save_final_model(self):
        final_path = os.path.join(self.output_dir, "final_detail_enhancer")
        self.controlnet.save_pretrained(os.path.join(final_path, "controlnet"))
        if self.use_lora:
            self.unet.save_pretrained(os.path.join(final_path, "lora_unet"))
        print(f"最终模型已保存: {final_path}")


# =============================================
# 3. 数据准备工具（无需修改）
# =============================================

def prepare_training_data(image_dir: str,
                         output_prompt_file: str,
                         ram_model_path: str = None):
    """使用 RAM 为图像生成语义标签"""
    print("正在准备训练数据...")

    if ram_model_path is None:
        # 使用默认提示
        default_prompts = ["high quality, detailed, sharp"]
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with open(output_prompt_file, 'w') as f:
            for img in image_files:
                f.write(f"{img},{np.random.choice(default_prompts)}\n")
    else:
        # 伪代码：调用 RAM 模型
        print("使用 RAM 生成语义标签...")
        # 实际需集成 RAMModel
        pass

    print(f"提示词文件已生成: {output_prompt_file}")


# =============================================
# 4. 使用示例
# =============================================

if __name__ == "__main__":
    # 示例训练流程
    dataset = SuperResolutionRefinementDataset(
        control_dir="data/hr_base",      # HAT-L 输出
        target_dir="data/hr_original",   # DIV2K/Flickr2K 原图
        prompt_file="data/prompts.csv"
    )

    trainer = DetailEnhancementTrainer(
        base_model_path="runwayml/stable-diffusion-v1-5",
        output_dir="checkpoints/detail_enhancer",
        device="cuda"
    )

    trainer.train(dataset, num_epochs=10, batch_size=2, learning_rate=5e-6)