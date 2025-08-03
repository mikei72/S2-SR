import numpy as np
import torch
from typing import Union, List, Dict
from PIL import Image
import os

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: 未安装lpips，LPIPS指标将不可用")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: 未安装skimage，PSNR和SSIM指标将不可用")

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化指标计算器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.lpips_fn = None
        
        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算PSNR
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            PSNR值
        """
        if not SKIMAGE_AVAILABLE:
            return self._calculate_psnr_manual(img1, img2)
        
        # 确保图像类型一致
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 使用skimage计算PSNR
        return psnr(img1, img2, data_range=255)
    
    def _calculate_psnr_manual(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """手动计算PSNR"""
        # 确保图像类型一致
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 计算MSE
        mse = np.mean((img1 - img2) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # 计算PSNR
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr_value
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算SSIM
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            SSIM值
        """
        if not SKIMAGE_AVAILABLE:
            return 0.0
        
        # 确保图像类型一致
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 使用skimage计算SSIM
        return ssim(img1, img2, multichannel=True, data_range=255)
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算LPIPS
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            LPIPS值
        """
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return 0.0
        
        # 转换为tensor
        img1_tensor = self._numpy_to_tensor(img1)
        img2_tensor = self._numpy_to_tensor(img2)
        
        # 计算LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1_tensor, img2_tensor)
        
        return lpips_value.item()
    
    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """numpy数组转tensor"""
        # 确保图像在[0, 1]范围内
        if img.max() > 1.0:
            img = img / 255.0
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img).float()
        
        # 调整维度: HWC -> CHW
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        
        # 移动到设备
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
    
    def calculate_all_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 计算PSNR
        metrics['psnr'] = self.calculate_psnr(img1, img2)
        
        # 计算SSIM
        metrics['ssim'] = self.calculate_ssim(img1, img2)
        
        # 计算LPIPS
        metrics['lpips'] = self.calculate_lpips(img1, img2)
        
        return metrics
    
    def batch_calculate_metrics(self, 
                              img1_list: List[np.ndarray], 
                              img2_list: List[np.ndarray]) -> Dict[str, List[float]]:
        """
        批量计算指标
        
        Args:
            img1_list: 图像1列表
            img2_list: 图像2列表
            
        Returns:
            包含所有指标列表的字典
        """
        if len(img1_list) != len(img2_list):
            raise ValueError("图像列表长度不匹配")
        
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
        for img1, img2 in zip(img1_list, img2_list):
            metrics = self.calculate_all_metrics(img1, img2)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        return all_metrics
    
    def calculate_average_metrics(self, 
                                img1_list: List[np.ndarray], 
                                img2_list: List[np.ndarray]) -> Dict[str, float]:
        """
        计算平均指标
        
        Args:
            img1_list: 图像1列表
            img2_list: 图像2列表
            
        Returns:
            包含平均指标的字典
        """
        batch_metrics = self.batch_calculate_metrics(img1_list, img2_list)
        
        avg_metrics = {}
        for key, values in batch_metrics.items():
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def save_metrics_report(self, 
                          metrics: Dict[str, float], 
                          output_path: str) -> None:
        """
        保存指标报告
        
        Args:
            metrics: 指标字典
            output_path: 输出路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("图像超分辨率评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            for metric_name, value in metrics.items():
                if metric_name.endswith('_std'):
                    continue
                
                std_key = f"{metric_name}_std"
                if std_key in metrics:
                    f.write(f"{metric_name.upper()}: {value:.4f} ± {metrics[std_key]:.4f}\n")
                else:
                    f.write(f"{metric_name.upper()}: {value:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("报告生成完成\n")
        
        print(f"评估报告已保存到: {output_path}")

def evaluate_super_resolution_results(hr_gt_list: List[np.ndarray],
                                    hr_pred_list: List[np.ndarray],
                                    output_dir: str = "evaluation_results") -> Dict[str, float]:
    """
    评估超分辨率结果
    
    Args:
        hr_gt_list: 真实高分辨率图像列表
        hr_pred_list: 预测高分辨率图像列表
        output_dir: 输出目录
        
    Returns:
        评估指标字典
    """
    # 创建评估器
    calculator = MetricsCalculator()
    
    # 计算平均指标
    avg_metrics = calculator.calculate_average_metrics(hr_gt_list, hr_pred_list)
    
    # 保存报告
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    calculator.save_metrics_report(avg_metrics, report_path)
    
    # 打印结果
    print("\n超分辨率评估结果:")
    print("=" * 30)
    for metric_name, value in avg_metrics.items():
        if not metric_name.endswith('_std'):
            std_key = f"{metric_name}_std"
            if std_key in avg_metrics:
                print(f"{metric_name.upper()}: {value:.4f} ± {avg_metrics[std_key]:.4f}")
            else:
                print(f"{metric_name.upper()}: {value:.4f}")
    
    return avg_metrics 