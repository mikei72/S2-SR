import torch
from PIL import Image
import numpy as np
from typing import Union, List, Tuple
import sys
import os

# 添加RAM模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'RAM'))

class RAMModel:
    """RAM模型包装器，用于语义标签生成"""
    
    def __init__(self, model_path: str, device: str = "cuda", 
                 confidence_threshold: float = 0.5, max_tags: int = 10):
        """
        初始化RAM模型
        
        Args:
            model_path: RAM模型权重路径
            device: 计算设备
            confidence_threshold: 标签置信度阈值
            max_tags: 最大标签数量
        """
        self.device = device
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_tags = max_tags
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载RAM模型"""
        try:
            # 这里需要根据实际的RAM模型结构进行适配
            # 由于RAM是外部依赖，这里提供接口框架
            print(f"正在加载RAM模型: {self.model_path}")
            
            # 模拟模型加载过程
            # 实际使用时需要导入RAM的具体实现
            # from ram.models import ram
            # self.model = ram(pretrained=self.model_path)
            # self.model.to(self.device)
            # self.model.eval()
            
            print("RAM模型加载完成")
            
        except Exception as e:
            print(f"RAM模型加载失败: {e}")
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
        
        # 调整图像尺寸为模型输入要求
        image = image.resize((224, 224))  # RAM模型通常使用224x224输入
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
        
        # 标准化 (ImageNet标准)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.to(self.device)
    
    def generate_tags(self, image: Union[str, Image.Image, np.ndarray]) -> List[str]:
        """
        生成图像标签
        
        Args:
            image: 输入图像
            
        Returns:
            标签列表
        """
        # 预处理
        input_tensor = self.preprocess_image(image)
        
        # 执行推理
        with torch.no_grad():
            # 这里需要调用实际的RAM模型推理
            # predictions = self.model(input_tensor)
            
            # 模拟标签生成过程
            # 实际使用时需要根据RAM的输出格式进行解析
            mock_tags = [
                "person", "building", "sky", "tree", "car", 
                "road", "grass", "cloud", "window", "door"
            ]
            
            # 模拟置信度
            mock_confidences = torch.rand(len(mock_tags))
            
            # 根据置信度阈值和最大标签数筛选
            valid_indices = torch.where(mock_confidences > self.confidence_threshold)[0]
            valid_indices = valid_indices[:self.max_tags]
            
            selected_tags = [mock_tags[i] for i in valid_indices]
        
        return selected_tags
    
    def generate_prompt(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """
        生成文本提示词
        
        Args:
            image: 输入图像
            
        Returns:
            逗号分隔的标签字符串
        """
        tags = self.generate_tags(image)
        return ", ".join(tags)
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """便捷调用接口"""
        return self.generate_prompt(image) 