import os
import shutil
import random
import glob
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser('Calculate SSIM and PSNR')
parser.add_argument('--fold_test', dest='fold_test', help='input directory for test results', type=str,
                    default='Path to/test_latest/images')
parser.add_argument('--rand_seed', help='random seed', type=int, default=42)
parser.add_argument('--dataset_name', type=str, default='TCGA_LGG_GBM')


args = parser.parse_args()
random.seed(args.rand_seed)

def calculate_ssim_psnr(args):
    fake_img_path, folder_path = args
    
    base_name = os.path.basename(fake_img_path)
    real_img_name = base_name.replace('fake_B', 'real_B')
    real_img_path = os.path.join(folder_path, real_img_name)

    if os.path.exists(real_img_path):
        # 读取图像
        fake_image = io.imread(fake_img_path)
        real_image = io.imread(real_img_path)

        # 计算SSIM和PSNR
        ssim_value = ssim(fake_image, real_image, multichannel=True, channel_axis=2)
        psnr_value = psnr(fake_image, real_image)
        print(real_img_name,psnr_value)
         # 如果PSNR为无穷大，返回None
        if np.isinf(psnr_value):  #特别暗的情况
            return None, None
        else:
            return ssim_value, psnr_value
    else:
        return None, None

def Calculate_SSIM_and_PSNR(args):
    folder_path = args.fold_test

    # 使用glob找到所有fake_B和real_B图像
    fake_images = glob.glob(os.path.join(folder_path, '*_fake_B.png'))

    # 多进程计算
    with Pool() as pool:
        results = pool.map(calculate_ssim_psnr, [(fake_img_path, folder_path) for fake_img_path in fake_images])

    # 提取结果
    ssim_scores = [result[0] for result in results if result[0] is not None]
    psnr_scores = [result[1] for result in results if result[1] is not None]

    # 计算SSIM和PSNR的均值
    if ssim_scores and psnr_scores:
        mean_ssim = np.mean(ssim_scores)
        print(f"Average SSIM: {mean_ssim}")
        mean_psnr = np.mean(psnr_scores)
        print(f"Average PSNR: {mean_psnr}")
        #保存指标结果
        log_file_name = f'test_results_metrics'
        os.makedirs(log_file_name, exist_ok=True)
        with open(os.path.join(log_file_name,f'{args.dataset_name}.txt'), 'a') as f:
            f.write(f"dataset_input: {args.dataset_name}, ckpt_path:{args.fold_test} \n")
            f.write("Overall: PSNR {:4f} SSIM {:4f}  \n".format(mean_psnr, mean_ssim))

    else:
        print("No matching fake_B and real_B images found.")


if __name__ == '__main__':
    Calculate_SSIM_and_PSNR(args)

    # SSIM（结构相似性指数）：SSIM是一种衡量两幅图像视觉相似度的指标，范围在0到1之间。SSIM值为1意味着两幅图像在结构、亮度和对比度上完全相同。
    # 因此，SSIM越接近1，表明图像质量损失越少，或者说生成的图像与原始图像越相似。
    #
    # PSNR（峰值信噪比）：PSNR是一种评估图像重构质量的指标，通常用于衡量压缩或恢复后的图像与原始图像之间的差异。
    # 它是以分贝（dB）为单位的比例指标，数值越高表示误差越小。高PSNR通常意味着较低的失真，但它并不总是与人类视觉感知的质量评价一致。
