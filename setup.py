#!/usr/bin/env python3
"""
项目安装脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """安装项目依赖"""
    print("正在安装项目依赖...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("正在创建项目目录...")
    
    directories = [
        "models/hat",
        "models/ram", 
        "models/stable-diffusion-v1-5",
        "outputs",
        "temp",
        "logs",
        "checkpoints",
        "examples/outputs",
        "examples/test_data",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")
    
    print("✓ 目录创建完成")

def download_models():
    """下载预训练模型（可选）"""
    print("\n预训练模型下载（可选）:")
    print("1. HAT模型: https://github.com/XPixelGroup/HAT/releases/download/v1.0/hat_l.pth")
    print("2. RAM模型: https://huggingface.co/xdecoder/RAM/resolve/main/ram_swin_large_14m.pth")
    print("3. SD模型: git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/stable-diffusion-v1-5")
    
    response = input("\n是否现在下载模型？(y/n): ").lower().strip()
    
    if response == 'y':
        print("开始下载模型...")
        
        # 这里可以添加自动下载逻辑
        # 由于模型文件较大，建议用户手动下载
        print("请手动下载模型文件到相应目录")
        print("下载完成后运行 python test_project.py 验证安装")

def main():
    """主安装函数"""
    print("基于扩散模型的文本引导图像超分辨率 - 安装程序")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("✗ 需要Python 3.8或更高版本")
        return False
    
    print(f"✓ Python版本: {sys.version}")
    
    # 安装依赖
    if not install_requirements():
        return False
    
    # 创建目录
    create_directories()
    
    # 下载模型
    download_models()
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    
    print("\n下一步:")
    print("1. 下载预训练模型文件")
    print("2. 运行 python test_project.py 验证安装")
    print("3. 运行 python examples/demo.py 查看演示")
    
    return True

if __name__ == "__main__":
    main() 