from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy import linalg
import os
import shutil
import random
import glob
import numpy as np
from skimage import io
import torch
import argparse

parser = argparse.ArgumentParser('Calculate FID')
parser.add_argument('--fold_test', dest='fold_test', help='input directory for test results', type=str,
                    default='/Path to/test_latest/images')
parser.add_argument('--rand_seed', help='random seed', type=int, default=42)
parser.add_argument('--batch_size', help='batch size', type=int, default=32)

args = parser.parse_args()
random.seed(args.rand_seed)


def calculate_fid(image1, image2, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 加载Inception模型
    inception_model = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1', transform_input=False).to(device)
    inception_model.eval()

    # 图像预处理函数
    def preprocess_images(images):
        images = [TF.resize(TF.to_pil_image(image), (299, 299)) for image in images]
        images = torch.stack([TF.to_tensor(image) for image in images])
        images = images.to(device)
        return images

    # 计算图像的特征
    def get_features(images):
        with torch.no_grad():
            pred = inception_model(images)
            pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        return pred

    # 分批处理图像
    batched_images1 = [image1[i:i + batch_size] for i in range(0, len(image1), batch_size)]
    batched_images2 = [image2[i:i + batch_size] for i in range(0, len(image2), batch_size)]

    act1 = np.concatenate([get_features(preprocess_images(batch)) for batch in batched_images1])
    act2 = np.concatenate([get_features(preprocess_images(batch)) for batch in batched_images2])

    # 计算均值和协方差
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # 计算FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid.real


def Calculate_FID(args):
    folder_path = args.fold_test

    # 使用glob找到所有fake_B和real_B图像
    fake_images = glob.glob(os.path.join(folder_path, '*_fake_B.png'))
    real_images = glob.glob(os.path.join(folder_path, '*_real_B.png'))

    all_fake_images = []
    all_real_images = []

    for fake_img_path in fake_images:
        base_name = os.path.basename(fake_img_path)
        real_img_name = base_name.replace('fake_B', 'real_B')
        real_img_path = os.path.join(folder_path, real_img_name)

        if real_img_path in real_images:
            all_fake_images.append(io.imread(fake_img_path))
            all_real_images.append(io.imread(real_img_path))

    # 计算FID
    fid_value = calculate_fid(all_fake_images, all_real_images, batch_size=args.batch_size)
    print('FID: ', fid_value / len(all_fake_images))


if __name__ == '__main__':
    Calculate_FID(args)

    '''
    FID（Fréchet Inception Distance）是一种衡量生成模型（特别是在图像领域）性能的指标，
    用于比较生成图像与真实图像之间的统计特性。FID的值通常在0到无穷大之间，
    数值越小表示生成图像与真实图像之间的差异越小，生成的图像质量越高。

    1、低于10-20的FID通常被认为是非常好的，意味着生成图像的质量很高。
    2、20-50的FID是比较合理的，表示生成图像的质量还不错。
    3、高于50的FID可能表明生成模型需要进一步改进。

    需要注意的是，FID只是衡量生成图像质量的一种方式，它主要关注图像在统计上的相似性，
    并不能完全反映人类的视觉感知。此外，不同的数据集和模型架构也会对FID值产生影响，
    因此在比较不同模型时需要考虑这些因素。
    '''