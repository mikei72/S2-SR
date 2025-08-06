# 基于扩散模型的文本引导图像超分辨率

本项目实现了一个基于扩散模型的文本引导图像超分辨率系统，采用"高保真先行，生成式精修"的核心理念，通过三个关键步骤实现高质量的图像超分辨率。

## 核心理念

**高保真先行，生成式精修**

- 首先使用HAT模型进行保真度超分，确保基础图像质量
- 然后使用RAM模型生成语义标签，提供文本引导
- 最后使用Stable Diffusion进行生成式精修，增加真实感细节

## 核心工作流程

### 步骤1：保真度超分 (Fidelity Upscaling)
- **任务**: 将低分辨率图像（LR）进行4倍放大
- **工具**: 预训练的HAT-L模型
- **输出**: 高保真度的基础图像 HR_base

### 步骤2：语义标签生成 (Semantic Tagging)
- **任务**: 为图像精准打上一系列关键词标签
- **工具**: 预训练的RAM (Recognize Anything Model)
- **输出**: 逗号分隔的关键词 text_tags_prompt

### 步骤3：生成式精修 (Generative Refinement)
- **任务**: 在HR_base基础上，根据关键词标签增加真实感细节
- **工具**: 预训练的Stable Diffusion v1.5
- **输出**: 最终的高分辨率图像 HR_final

## 关键参数

- **strength (重绘强度)**: 控制对HR_base的修改程度，平衡PSNR与视觉效果
- **建议取值**: 0.1-0.25的低区间内

## 项目结构

```
S2-SR/
├── config/
│   └── config.py              # 配置文件
├── examples/
│   ├── outputs/               # 测试输出
├── models/
│   ├── HAT                    # HAT模型
│   ├── ram                    # RAM模型
│   ├── weights                # 权重
│   ├── hat_model.py           # HAT模型包装器
│   └── sd_model.py            # SD模型包装器
├── pipeline/
│   └── super_resolution_pipeline.py  # 核心处理流程
├── training/
│   └── lora_trainer.py        # LoRA训练模块
├── utils/
│   ├── image_utils.py         # 图像处理工具
│   └── metrics_utils          # 评估指标
├── demo.py                    # 测试demo
├── download.py                # 下载权重脚本
├── main.py                    # 主程序入口
├── README.md                  # 项目说明
└── requirements.txt           # 依赖包列表
```

## 安装指南

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- 至少8GB显存

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/mikei72/S2-SR
cd S2-SR

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载预训练模型

通过网盘分享的文件：数据集
链接: https://pan.baidu.com/s/1_uwMaxgZ3QYKXvbHp8D2og?pwd=3ryp 提取码: 3ryp 
--来自百度网盘超级会员v3的分享

下载其中的HAT、RAM、和Stable Diffusion权重
放在 models/weights 目录下

## 使用方法
★★★★★★★★★★当前运行demo.py











### 1. 单张图像处理

```bash
python main.py --input input_image.jpg --output output_image.png --strength 0.15
```

### 2. 批量处理

```bash
python main.py --input_dir input_folder --output_dir output_folder --strength 0.15
```

### 3. 使用自定义LoRA

```bash
python main.py --input input_image.jpg --output output_image.png --lora_path path/to/lora
```

### 4. 训练LoRA

```bash
# 准备训练数据
python main.py --create_test_data --test_image high_res_image.jpg --test_output test_data

# 训练LoRA
python main.py --train_lora --train_data train_folder --epochs 10 --lr 1e-4
```

### 5. 评估结果

```bash
python main.py --evaluate --gt_dir ground_truth_folder --pred_dir prediction_folder
```

## 参数说明

### 基本参数
- `--device`: 计算设备 (默认: cuda)
- `--strength`: 重绘强度 (默认: 0.15)

### 模型路径
- `--hat_model`: HAT模型路径
- `--ram_model`: RAM模型路径
- `--sd_model`: SD模型路径
- `--lora_path`: LoRA权重路径

### 训练参数
- `--epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批次大小 (默认: 1)
- `--lr`: 学习率 (默认: 1e-4)

## 示例用法

### 示例1：基础超分辨率
```python
from pipeline.super_resolution_pipeline import SuperResolutionPipeline

# 创建处理流程
pipeline = SuperResolutionPipeline(device="cuda")

# 处理图像
hr_final, info = pipeline.process(
    lr_image="input.jpg",
    strength=0.15,
    output_path="output.png"
)

print(f"处理完成: {info}")
```

### 示例2：自定义参数
```python
from config.config import Config

# 修改配置
Config.DEFAULT_STRENGTH = 0.2
Config.TARGET_SIZE = (2048, 2048)

# 创建处理流程
pipeline = SuperResolutionPipeline(
    device="cuda",
    lora_path="checkpoints/lora_final"
)

# 处理图像
hr_final, info = pipeline.process("input.jpg", strength=0.2)
```

### 示例3：批量处理
```python
import os
from pipeline.super_resolution_pipeline import SuperResolutionPipeline

# 创建处理流程
pipeline = SuperResolutionPipeline(device="cuda")

# 获取输入文件列表
input_dir = "input_images"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

input_files = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 批量处理
for filename in input_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"sr_{filename}")
    
    hr_final, info = pipeline.process(
        lr_image=input_path,
        strength=0.15,
        output_path=output_path
    )
    
    print(f"处理完成: {filename}")
```

## 性能优化

### 1. 内存优化
- 使用`--device cpu`在CPU上运行（较慢但内存需求低）
- 调整batch_size减少内存使用

### 2. 速度优化
- 使用xformers加速注意力计算
- 启用混合精度训练
- 使用更小的图像尺寸进行测试

### 3. 质量优化
- 调整strength参数平衡保真度和生成质量
- 使用训练好的LoRA提升特定场景效果
- 优化RAM生成的提示词质量

## 评估指标

项目支持以下评估指标：
- **PSNR**: 峰值信噪比，衡量图像保真度
- **SSIM**: 结构相似性指数，衡量视觉质量
- **LPIPS**: 学习型感知图像块相似度，衡量感知质量

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用CPU模式
   - 降低图像分辨率

2. **模型加载失败**
   - 检查模型文件路径
   - 确保模型文件完整
   - 检查依赖包版本

3. **处理速度慢**
   - 使用GPU加速
   - 启用xformers
   - 减少推理步数

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
python -u main.py --input test.jpg --output test_out.png 2>&1 | tee log.txt
```

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8

# 代码格式化
black .
flake8 .
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- [HAT](https://github.com/XPixelGroup/HAT) - 超分辨率模型
- [RAM](https://github.com/xdecoder/RAM) - 图像识别模型
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 扩散模型
- [Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{s2sr2024,
  title={基于扩散模型的文本引导图像超分辨率},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/S2-SR}
}
``` 