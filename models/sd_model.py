import torch
from PIL import Image
import numpy as np
from typing import Union, Optional
from diffusers import StableDiffusionImg2ImgPipeline
import os

class SDModel:
    """Stable Diffusion模型包装器，用于生成式精修"""
    
    def __init__(self, model_path: str, device: str = "cuda", 
                 lora_path: Optional[str] = None):
        """
        初始化SD模型
        
        Args:
            model_path: SD模型路径
            device: 计算设备
            lora_path: LoRA权重路径（可选）
        """
        self.device = device
        self.model_path = model_path
        self.lora_path = lora_path
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载SD模型"""
        try:
            print(f"正在加载Stable Diffusion模型: {self.model_path}")
            
            # 加载img2img pipeline
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # 加载LoRA权重（如果提供）
            if self.lora_path and os.path.exists(self.lora_path):
                print(f"正在加载LoRA权重: {self.lora_path}")
                self.pipeline.load_lora_weights(self.lora_path)
            
            # 移动到指定设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            print("Stable Diffusion模型加载完成")
            
        except Exception as e:
            print(f"Stable Diffusion模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        预处理输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的PIL图像
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        return image
    
    def refine_image(self, 
                    base_image: Union[str, Image.Image, np.ndarray],
                    prompt: str,
                    strength: float = 0.15,
                    guidance_scale: float = 7.5,
                    num_inference_steps: int = 50,
                    negative_prompt: str = "") -> Image.Image:
        """
        对基础图像进行生成式精修
        
        Args:
            base_image: 基础图像（HAT超分结果）
            prompt: 文本提示词
            strength: 重绘强度 (0.0-1.0)
            guidance_scale: 分类器自由引导尺度
            num_inference_steps: 推理步数
            negative_prompt: 负面提示词
            
        Returns:
            精修后的图像
        """
        # 预处理基础图像
        base_image = self.preprocess_image(base_image)
        
        # 执行img2img生成
        result = self.pipeline(
            prompt=prompt,
            image=base_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt
        )
        
        return result.images[0]
    
    def batch_refine(self, 
                    base_images: list,
                    prompts: list,
                    strength: float = 0.15,
                    guidance_scale: float = 7.5,
                    num_inference_steps: int = 50,
                    negative_prompt: str = "") -> list:
        """
        批量精修图像
        
        Args:
            base_images: 基础图像列表
            prompts: 提示词列表
            strength: 重绘强度
            guidance_scale: 分类器自由引导尺度
            num_inference_steps: 推理步数
            negative_prompt: 负面提示词
            
        Returns:
            精修后的图像列表
        """
        results = []
        
        for base_image, prompt in zip(base_images, prompts):
            refined_image = self.refine_image(
                base_image=base_image,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt
            )
            results.append(refined_image)
        
        return results
    
    def __call__(self, 
                base_image: Union[str, Image.Image, np.ndarray],
                prompt: str,
                strength: float = 0.15,
                **kwargs) -> Image.Image:
        """便捷调用接口"""
        return self.refine_image(base_image, prompt, strength, **kwargs) 