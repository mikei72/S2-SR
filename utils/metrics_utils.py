import argparse
import os
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim


# ==============================================================================
# 核心计算函数 (无需修改)
# ==============================================================================
def calculate_metrics(sr_path: str, gt_path: str, crop_border: int, test_y_channel: bool):
    """
    计算 SR 图像和 GT 图像之间的 PSNR 和 SSIM 指标。
    脚本内置了对旋转和尺寸不匹配问题的鲁棒自动校正。
    """
    ROTATION_TOLERANCE = 5

    try:
        sr_img_pil = Image.open(sr_path).convert("RGB")
        gt_img_pil = Image.open(gt_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return

    sr_img = np.array(sr_img_pil, dtype=np.uint8)
    gt_img = np.array(gt_img_pil, dtype=np.uint8)

    h_sr, w_sr, _ = sr_img.shape
    h_gt, w_gt, _ = gt_img.shape

    if h_sr != h_gt or w_sr != w_gt:
        print("提示: SR 和 GT 尺寸不匹配，开始自动校正...")
        is_rotated = (abs(h_sr - w_gt) <= ROTATION_TOLERANCE and
                      abs(w_sr - h_gt) <= ROTATION_TOLERANCE)
        if is_rotated:
            print(f"  - 检测到旋转 (SR: {h_sr}x{w_sr}, GT: {h_gt}x{w_gt})，正在校正...")
            sr_img = np.rot90(sr_img)
            sr_img = np.rot90(sr_img)
            sr_img = np.rot90(sr_img)
            h_sr, w_sr, _ = sr_img.shape
        if h_sr != h_gt or w_sr != w_gt:
            print(f"  - 裁剪到共同最小尺寸...")
            target_h = min(h_sr, h_gt)
            target_w = min(w_sr, w_gt)
            sr_img = sr_img[:target_h, :target_w, :]
            gt_img = gt_img[:target_h, :target_w, :]
            print(f"  - 校正后尺寸: {target_h}x{target_w}")
    assert sr_img.shape == gt_img.shape

    if crop_border > 0:
        gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, ...]
        sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    psnr = calculate_psnr(gt_img, sr_img, data_range=255)

    is_multichannel = len(gt_img.shape) == 3
    ssim = calculate_ssim(gt_img, sr_img, data_range=255, multichannel=is_multichannel,
                          channel_axis=-1 if is_multichannel else None)

    return {'psnr': psnr, 'ssim': ssim}
