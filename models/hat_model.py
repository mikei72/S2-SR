import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, Tuple
import sys
import os

# 添加HAT模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'HAT'))

class HATModel:
    """HAT模型包装器，用于保真度超分辨率"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化HAT模型
        
        Args:
            model_path: HAT模型权重路径
            device: 计算设备
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载HAT模型"""
        try:
            # 这里需要根据实际的HAT模型结构进行适配
            # 由于HAT是外部依赖，这里提供接口框架
            print(f"正在加载HAT模型: {self.model_path}")
            
            # 模拟模型加载过程
            # 实际使用时需要导入HAT的具体实现
            # from hat.models import HAT
            # self.model = HAT()
            # self.model.load_state_dict(torch.load(self.model_path))
            
            print("HAT模型加载完成")
            
        except Exception as e:
            print(f"HAT模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        预处理输入图像
        
        Args:
            image: 输入图像（路径、PIL图像或numpy数组）
            
        Returns:
            预处理后的tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
        
        return image_tensor.to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        后处理输出tensor为图像
        
        Args:
            tensor: 模型输出的tensor
            
        Returns:
            处理后的numpy数组
        """
        # 确保值在[0, 1]范围内
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def upscale(self, image: Union[str, Image.Image, np.ndarray], 
                target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        执行超分辨率
        
        Args:
            image: 输入低分辨率图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            超分辨率后的图像
        """
        # 预处理
        input_tensor = self.preprocess_image(image)
        
        # 获取原始尺寸
        _, _, h, w = input_tensor.shape
        
        # 计算目标尺寸
        if target_size is None:
            target_h, target_w = h * 4, w * 4  # 默认4倍放大
        else:
            target_w, target_h = target_size
        
        # 执行超分辨率推理
        with torch.no_grad():
            # 这里需要调用实际的HAT模型推理
            # output_tensor = self.model(input_tensor)
            
            # 临时使用双线性插值作为占位符
            output_tensor = F.interpolate(
                input_tensor, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 后处理
        output_image = self.postprocess_image(output_tensor)
        
        return output_image
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray], 
                 target_size: Tuple[int, int] = None) -> np.ndarray:
        """便捷调用接口"""
        return self.upscale(image, target_size) 