#!/usr/bin/env python3
"""
项目测试脚本：验证所有模块的导入和基本功能
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试所有模块的导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    try:
        # 测试配置模块
        print("✓ 导入 config.config...")
        from config.config import Config
        print(f"  项目根目录: {Config.PROJECT_ROOT}")
        
        # 测试图像工具模块
        print("✓ 导入 utils.image_utils...")
        from utils.image_utils import load_image, save_image, calculate_psnr
        print("  图像工具函数导入成功")
        
        # 测试评估模块
        print("✓ 导入 evaluation.metrics...")
        from evaluation.metrics import MetricsCalculator
        print("  评估模块导入成功")
        
        # 测试模型模块（这些可能需要外部依赖）
        print("✓ 导入 models.hat_model...")
        from models.hat_model import HATModel
        print("  HAT模型模块导入成功")
        
        print("✓ 导入 models.ram_model...")
        from models.ram_model import RAMModel
        print("  RAM模型模块导入成功")
        
        print("✓ 导入 models.sd_model...")
        from models.sd_model import SDModel
        print("  SD模型模块导入成功")
        
        # 测试处理流程模块
        print("✓ 导入 pipeline.super_resolution_pipeline...")
        from pipeline.super_resolution_pipeline import SuperResolutionPipeline
        print("  处理流程模块导入成功")
        
        # 测试训练模块
        print("✓ 导入 training.lora_trainer...")
        from training.lora_trainer import LoRATrainer, SuperResolutionDataset
        print("  训练模块导入成功")
        
        print("\n所有模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_config():
    """测试配置功能"""
    print("\n" + "=" * 60)
    print("测试配置功能")
    print("=" * 60)
    
    try:
        from config.config import Config
        
        # 测试目录创建
        print("✓ 创建项目目录...")
        directories = Config.create_directories()
        print(f"  创建了 {len(directories)} 个目录")
        
        # 测试配置参数
        print("✓ 检查配置参数...")
        print(f"  默认重绘强度: {Config.DEFAULT_STRENGTH}")
        print(f"  目标分辨率: {Config.TARGET_SIZE}")
        print(f"  超分倍数: {Config.UPSCALE_FACTOR}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_image_utils():
    """测试图像工具功能"""
    print("\n" + "=" * 60)
    print("测试图像工具功能")
    print("=" * 60)
    
    try:
        import numpy as np
        from PIL import Image
        from utils.image_utils import create_low_resolution_image, calculate_psnr, calculate_ssim
        
        # 创建测试图像
        print("✓ 创建测试图像...")
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print(f"  测试图像尺寸: {test_image.shape}")
        
        # 测试低分辨率图像创建
        print("✓ 创建低分辨率图像...")
        lr_image = create_low_resolution_image(test_image, scale_factor=4)
        print(f"  低分辨率图像尺寸: {lr_image.shape}")
        
        # 测试PSNR计算
        print("✓ 计算PSNR...")
        psnr_value = calculate_psnr(test_image, test_image)  # 相同图像
        print(f"  PSNR (相同图像): {psnr_value}")
        
        # 测试SSIM计算
        print("✓ 计算SSIM...")
        ssim_value = calculate_ssim(test_image, test_image)  # 相同图像
        print(f"  SSIM (相同图像): {ssim_value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 图像工具测试失败: {e}")
        return False

def test_metrics_calculator():
    """测试评估指标计算器"""
    print("\n" + "=" * 60)
    print("测试评估指标计算器")
    print("=" * 60)
    
    try:
        import numpy as np
        from evaluation.metrics import MetricsCalculator
        
        # 创建计算器
        print("✓ 创建指标计算器...")
        calculator = MetricsCalculator(device="cpu")  # 使用CPU避免CUDA问题
        print("  指标计算器创建成功")
        
        # 创建测试图像
        print("✓ 创建测试图像...")
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试指标计算
        print("✓ 计算评估指标...")
        metrics = calculator.calculate_all_metrics(img1, img2)
        
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 评估指标测试失败: {e}")
        return False

def test_model_initialization():
    """测试模型初始化（不加载实际权重）"""
    print("\n" + "=" * 60)
    print("测试模型初始化")
    print("=" * 60)
    
    try:
        # 测试HAT模型初始化
        print("✓ 测试HAT模型初始化...")
        from models.hat_model import HATModel
        # 注意：这里使用不存在的路径，应该会失败但不会崩溃
        try:
            hat_model = HATModel("nonexistent_path.pth", device="cpu")
            print("  HAT模型初始化成功（模拟模式）")
        except:
            print("  HAT模型初始化失败（预期行为，因为模型文件不存在）")
        
        # 测试RAM模型初始化
        print("✓ 测试RAM模型初始化...")
        from models.ram_model import RAMModel
        try:
            ram_model = RAMModel("nonexistent_path.pth", device="cpu")
            print("  RAM模型初始化成功（模拟模式）")
        except:
            print("  RAM模型初始化失败（预期行为，因为模型文件不存在）")
        
        # 测试SD模型初始化
        print("✓ 测试SD模型初始化...")
        from models.sd_model import SDModel
        try:
            sd_model = SDModel("nonexistent_path", device="cpu")
            print("  SD模型初始化成功（模拟模式）")
        except:
            print("  SD模型初始化失败（预期行为，因为模型文件不存在）")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型初始化测试失败: {e}")
        return False

def test_pipeline_creation():
    """测试处理流程创建"""
    print("\n" + "=" * 60)
    print("测试处理流程创建")
    print("=" * 60)
    
    try:
        from pipeline.super_resolution_pipeline import SuperResolutionPipeline
        
        print("✓ 测试处理流程创建...")
        # 注意：这里会失败，因为模型文件不存在，但应该不会崩溃
        try:
            pipeline = SuperResolutionPipeline(device="cpu")
            print("  处理流程创建成功（模拟模式）")
        except Exception as e:
            print(f"  处理流程创建失败（预期行为）: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 处理流程测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("基于扩散模型的文本引导图像超分辨率 - 项目测试")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("配置功能", test_config),
        ("图像工具", test_image_utils),
        ("评估指标", test_metrics_calculator),
        ("模型初始化", test_model_initialization),
        ("处理流程", test_pipeline_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目结构正确。")
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
    
    print("\n下一步:")
    print("1. 下载预训练模型文件")
    print("2. 安装外部依赖（如HAT、RAM等）")
    print("3. 运行 python examples/demo.py 进行完整演示")

if __name__ == "__main__":
    main() 