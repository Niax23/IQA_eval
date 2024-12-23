import cv2
import torch
import argparse
from psnr_ssim.calc import calculate_psnr, calculate_ssim,calculate_psnr_pt, calculate_ssim_pt
import os
import numpy as np
import json
from tqdm import tqdm
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def compute_metrics(img_dir, gt_dir, device='cpu'):
    results = {"ssim": {}, "psnr": {}}
    
    # 获取图片和GT目录中的文件列表
    img_files = sorted(os.listdir(img_dir))
    gt_files = sorted(os.listdir(gt_dir))

    psnr_scores = []
    ssim_scores = []
    
    # 遍历文件并计算指标
    for img_name, gt_name in tqdm(zip(img_files,gt_files)):
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir,gt_name)

        # 读取图片
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        psnr = calculate_psnr(img, gt, crop_border=4, input_order='HWC', test_y_channel=True)
        ssim = calculate_ssim(img, gt, crop_border=4, input_order='HWC', test_y_channel=True)
        
        # 记录路径和分数
        img_gt_pair = f"{img_path} | {gt_path}"
        results["ssim"][img_gt_pair] = ssim
        results["psnr"][img_gt_pair] = psnr
        
        # 保存每个分数用于计算均值
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    # 计算均值
    results["ssim"]["mean"] = np.mean(ssim_scores)
    results["psnr"]["mean"] = np.mean(psnr_scores)
    
    return results
def save_results(results, output_dir, output_filename):
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {output_path}")
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Compute PSNR and SSIM for a set of images against ground truth.")
    parser.add_argument('--img_dir', required=True, help='Directory containing images to be evaluated')
    parser.add_argument('--gt_dir', required=True, help='Directory containing ground truth images')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output JSON results')
    
    
    args = parser.parse_args()

    # 计算指标
    results = compute_metrics(args.img_dir, args.gt_dir)
    
    # 保存结果到文件
    filename = "psnr_ssim.json"
    save_results(results, args.output_dir, filename)

if __name__ == "__main__":
    main()