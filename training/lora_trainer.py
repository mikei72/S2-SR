import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import os
import json
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm

class SuperResolutionDataset(Dataset):
    """超分辨率训练数据集"""
    
    def __init__(self, 
                 image_dir: str,
                 prompt_file: str,
                 transform=None,
                 target_size: tuple = (512, 512)):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录
            prompt_file: 提示词文件路径
            transform: 数据变换
            target_size: 目标尺寸
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        
        # 加载提示词
        self.prompts = self._load_prompts(prompt_file)
        
        # 获取图像文件列表
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"加载了 {len(self.image_files)} 张训练图像")
    
    def _load_prompts(self, prompt_file: str) -> Dict[str, str]:
        """加载提示词"""
        prompts = {}
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ',' in line:
                        filename, prompt = line.strip().split(',', 1)
                        prompts[filename] = prompt.strip()
        return prompts
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size)
        
        # 获取提示词
        prompt = self.prompts.get(image_file, "high quality, detailed")
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'prompt': prompt,
            'filename': image_file
        }

class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(self, 
                 base_model_path: str,
                 output_dir: str = "checkpoints",
                 device: str = "cuda"):
        """
        初始化LoRA训练器
        
        Args:
            base_model_path: 基础模型路径
            output_dir: 输出目录
            device: 计算设备
        """
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.device = device
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型
        self._load_base_model()
        self._setup_lora()
    
    def _load_base_model(self):
        """加载基础模型"""
        print("正在加载基础Stable Diffusion模型...")
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipeline = self.pipeline.to(self.device)
        print("基础模型加载完成")
    
    def _setup_lora(self):
        """设置LoRA配置"""
        print("正在配置LoRA...")
        
        # LoRA配置
        lora_config = LoraConfig(
            r=16,  # LoRA秩
            lora_alpha=32,  # LoRA缩放参数
            target_modules=["q_proj", "v_proj"],  # 目标模块
            lora_dropout=0.1,  # Dropout率
            bias="none",  # 偏置处理
            task_type="CAUSAL_LM"  # 任务类型
        )
        
        # 应用LoRA到模型
        self.model = get_peft_model(self.pipeline.unet, lora_config)
        self.model.print_trainable_parameters()
        
        print("LoRA配置完成")
    
    def train(self, 
              train_dataset: SuperResolutionDataset,
              num_epochs: int = 10,
              batch_size: int = 1,
              learning_rate: float = 1e-4,
              save_steps: int = 500,
              gradient_accumulation_steps: int = 4):
        """
        训练LoRA
        
        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            save_steps: 保存步数
            gradient_accumulation_steps: 梯度累积步数
        """
        print("开始LoRA训练...")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * num_epochs
        )
        
        # 训练循环
        global_step = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 获取数据
                images = batch['image'].to(self.device)
                prompts = batch['prompt']
                
                # 前向传播
                loss = self._compute_loss(images, prompts)
                
                # 反向传播
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
                # 保存检查点
                if global_step % save_steps == 0:
                    self._save_checkpoint(global_step)
            
            # 打印epoch结果
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
            
            # 保存epoch检查点
            self._save_checkpoint(f"epoch_{epoch + 1}")
        
        # 保存最终模型
        self._save_final_model()
        print("LoRA训练完成")
    
    def _compute_loss(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """计算损失"""
        # 这里需要根据具体的训练策略实现损失计算
        # 可以使用重建损失、感知损失等
        
        # 简化的损失计算（示例）
        batch_size = images.size(0)
        
        # 使用图像重建损失
        reconstructed_images = self.model(images)  # 这里需要根据实际模型输出调整
        loss = nn.MSELoss()(reconstructed_images, images)
        
        return loss
    
    def _save_checkpoint(self, step: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"lora_checkpoint_{step}")
        
        # 保存LoRA权重
        self.model.save_pretrained(checkpoint_path)
        
        # 保存训练配置
        config = {
            'step': step,
            'base_model_path': self.base_model_path,
            'lora_config': {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.1
            }
        }
        
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型"""
        final_path = os.path.join(self.output_dir, "lora_final")
        self.model.save_pretrained(final_path)
        print(f"最终模型已保存: {final_path}")
    
    def load_trained_lora(self, lora_path: str):
        """加载训练好的LoRA"""
        print(f"正在加载训练好的LoRA: {lora_path}")
        
        # 加载LoRA权重
        self.pipeline.load_lora_weights(lora_path)
        
        print("LoRA加载完成")

def prepare_training_data(image_dir: str, 
                         output_prompt_file: str,
                         ram_model_path: str = None):
    """
    准备训练数据
    
    Args:
        image_dir: 图像目录
        output_prompt_file: 输出提示词文件
        ram_model_path: RAM模型路径（用于生成提示词）
    """
    print("正在准备训练数据...")
    
    # 如果没有提供RAM模型，使用默认提示词
    if ram_model_path is None:
        default_prompts = [
            "high quality, detailed, sharp, clear",
            "professional photography, high resolution",
            "crystal clear, ultra detailed, sharp focus",
            "high quality image, detailed texture",
            "professional grade, sharp details"
        ]
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        with open(output_prompt_file, 'w', encoding='utf-8') as f:
            for image_file in image_files:
                # 随机选择一个提示词
                prompt = np.random.choice(default_prompts)
                f.write(f"{image_file},{prompt}\n")
        
        print(f"已为 {len(image_files)} 张图像生成默认提示词")
    
    else:
        # 使用RAM模型生成提示词
        from models.ram_model import RAMModel
        
        ram_model = RAMModel(ram_model_path)
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        with open(output_prompt_file, 'w', encoding='utf-8') as f:
            for image_file in tqdm(image_files, desc="生成提示词"):
                image_path = os.path.join(image_dir, image_file)
                prompt = ram_model.generate_prompt(image_path)
                f.write(f"{image_file},{prompt}\n")
        
        print(f"已为 {len(image_files)} 张图像生成RAM提示词")
    
    print(f"训练数据准备完成，提示词文件: {output_prompt_file}") 