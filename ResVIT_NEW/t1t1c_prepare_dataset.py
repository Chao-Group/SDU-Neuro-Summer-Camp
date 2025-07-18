import os
import numpy as np
import nibabel as nib
from PIL import Image
import concurrent.futures
import argparse
import random


# random.seed(args.rand_seed)
# np.random.seed(args.rand_seed)

def normalize_image(image):
    """归一化图像数据到 0-255 范围。"""
    # 转换为浮点数以避免数据溢出
    image = image.astype(np.float32)

    # 检查 NaN 值并替换为零
    image = np.nan_to_num(image)

    # 获取最大值和最小值
    min_val = np.min(image)
    max_val = np.max(image)

    # 避免除以零的情况
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros(image.shape)

    # 转换为 0-255 范围的 uint8 类型
    image = (image * 255).astype(np.uint8)

    return image

# def process_patient(patient_name, split_dir, train_data_dir, test_data_dir, save_dir_t2_flair, save_dir_flair_t2, save_dir_t1_t2_flair, save_dir_t1_flair_t2, save_dir_t2_flair_t1, save_dir_t1_t1ce, num_slices):
def process_patient(patient_name, split_dir, train_data_dir, test_data_dir, val_data_dir, save_dir_t1_t1ce): #, num_slices):
    # # brats17  
    # if split_dir == "test":
    #     patient_dir = os.path.join(test_data_dir, patient_name)
    # elif split_dir == 'train':
    #     lgg_names = os.listdir(os.path.join(train_data_dir, 'LGG'))
    #     hgg_names = os.listdir(os.path.join(train_data_dir, 'HGG'))
        
    #     if patient_name in lgg_names:
    #         patient_dir = os.path.join(train_data_dir, 'LGG', patient_name)
    #     elif patient_name in hgg_names:
    #         patient_dir = os.path.join(train_data_dir, 'HGG', patient_name)

    # elif split_dir == 'val':
    #     patient_dir = os.path.join(val_data_dir, patient_name)
    
    

    #lgggbm
    if split_dir == "test":
        patient_dir = os.path.join(test_data_dir, patient_name)
    elif split_dir == 'train':
        patient_dir = os.path.join(train_data_dir, patient_name)
    elif split_dir == 'val':
        patient_dir = os.path.join(val_data_dir, patient_name)
    
  
    
   
    for patient_id in os.listdir(patient_dir):
        # import pdb;pdb.set_trace()
        if patient_id.endswith('_t1.nii.gz'):
            t1_path = os.path.join(patient_dir, patient_id)
        elif patient_id.endswith('_t1Gd.nii.gz'):
            t1c_path = os.path.join(patient_dir, patient_id)
    
    # t1_path = os.path.join(patient_dir, patient_name + "_t1.nii.gz")
    # t1c_path = os.path.join(patient_dir, patient_name + "_t1Gd.nii.gz")
    # t1_path = os.path.join(patient_dir, patient_name + "_t1.nii")
    # t1c_path = os.path.join(patient_dir, patient_name + "_t1ce.nii")
    # t1n_path = os.path.join(patient_dir, patient_name + "-t1n.nii.gz")
    # t2f_path = os.path.join(patient_dir, patient_name + "-t2f.nii.gz")
    # t2w_path = os.path.join(patient_dir, patient_name + "-t2w.nii.gz")

    # H, W, C
    t1c_data = nib.load(t1c_path).get_fdata()
    t1_data = nib.load(t1_path).get_fdata()
    # t2f_data = nib.load(t2f_path).get_fdata()
    # t2w_data = nib.load(t2w_path).get_fdata()

    total_slices = t1c_data.shape[2]

    # 计算开始和结束的索引以提取中间的100个切片
    # mid_point = total_slices // 2
    # start = max(mid_point - num_slices // 2, 0)
    # end = min(mid_point + num_slices // 2, total_slices)
    # selected_slices = [i for i in range(start, end)]
    # t1c_slices = t1c_data[:, :, selected_slices]
    # t1_slices = t1_data[:, :, selected_slices]
    # t2f_slices = t2f_data[:, :, selected_slices]
    # t2w_slices = t2w_data[:, :, selected_slices]

    # one to one tasks
    for idx in range(total_slices):
        # 将 T2 和 FLAIR 切片转换为 PIL 图像
        # t2_img_pil = Image.fromarray(normalize_image(t2w_slices[:, :, idx]))
        # flair_img_pil = Image.fromarray(normalize_image(t2f_slices[:, :, idx]))

        # 将 T1 和 T1C 切片转换为 PIL 图像
        t1_img_pil = Image.fromarray(normalize_image(t1_data[:, :, idx]))
        t1ce_img_pil = Image.fromarray(normalize_image(t1c_data[:, :, idx]))

        # # 拼接图像 T2 -> FLAIR
        # combined_img = Image.new('L', (t2_img_pil.width + flair_img_pil.width, t2_img_pil.height))
        # combined_img.paste(t2_img_pil, (0, 0))
        # combined_img.paste(flair_img_pil, (t2_img_pil.width, 0))

        # # 拼接图像 FLAIR -> T2
        # combined_img2 = Image.new('L', (t2_img_pil.width + flair_img_pil.width, t2_img_pil.height))
        # combined_img2.paste(flair_img_pil, (0, 0))
        # combined_img2.paste(t2_img_pil, (flair_img_pil.width, 0))

        # 拼接图像 T1 -> T1CE   # L代表灰度图像模式
        combined_img3 = Image.new('L', (t1_img_pil.width + t1ce_img_pil.width, t1_img_pil.height))
        combined_img3.paste(t1_img_pil, (0, 0))
        combined_img3.paste(t1ce_img_pil, (t1_img_pil.width, 0))

        # 保存图像
        # combined_img.save(os.path.join(save_dir_t2_flair, split_dir, f"{patient_name}_{idx}.png"))
        # combined_img2.save(os.path.join(save_dir_flair_t2, split_dir, f"{patient_name}_{idx}.png"))
        combined_img3.save(os.path.join(save_dir_t1_t1ce, split_dir, f"{patient_name}_{idx}.png"))

    # # many to one tasks
    # for idx in range(num_slices):
    #     # 将 T1, FLAIR, T2 切片转换为 PIL 图像并调整到相同的数据范围
    #     t2_img_pil = Image.fromarray(normalize_image(t2w_slices[:, :, idx]))
    #     flair_img_pil = Image.fromarray(normalize_image(t2f_slices[:, :, idx]))
    #     t1n_img_pil = Image.fromarray(normalize_image(t1n_slices[:, :, idx]))

    #     # 拼接图像 T1 -> FLAIR -> T2 RGB三通道
    #     combined_img_left = Image.merge("RGB", (t1n_img_pil, flair_img_pil, t2_img_pil))
    #     combined_img_right = Image.merge("RGB", (t2_img_pil, t2_img_pil, t2_img_pil))
    #     combined_img = Image.new('RGB',
    #                              (combined_img_left.width + combined_img_right.width, combined_img_left.height))
    #     combined_img.paste(combined_img_left, (0, 0))
    #     combined_img.paste(combined_img_right, (combined_img_left.width, 0))
    #     # 拼接图像 T2 -> FLAIR -> T1 RGB三通道
    #     combined_img_left2 = Image.merge("RGB", (t2_img_pil, flair_img_pil, t1n_img_pil))
    #     combined_img_right2 = Image.merge("RGB", (t1n_img_pil, t1n_img_pil, t1n_img_pil))
    #     combined_img2 = Image.new('RGB',
    #                               (combined_img_left2.width + combined_img_right2.width, combined_img_left2.height))
    #     combined_img2.paste(combined_img_left2, (0, 0))
    #     combined_img2.paste(combined_img_right2, (combined_img_left2.width, 0))
    #     # 拼接图像 T1 -> T2 -> FLAIR RGB三通道
    #     combined_img_left3 = Image.merge("RGB", (t1n_img_pil, t2_img_pil, flair_img_pil))
    #     combined_img_right3 = Image.merge("RGB", (flair_img_pil, flair_img_pil, flair_img_pil))
    #     combined_img3 = Image.new('RGB',
    #                               (combined_img_left3.width + combined_img_right3.width, combined_img_left3.height))
    #     combined_img3.paste(combined_img_left3, (0, 0))
    #     combined_img3.paste(combined_img_right3, (combined_img_left3.width, 0))

    #     # 保存图像
    #     combined_img.save(os.path.join(save_dir_t1_flair_t2, split_dir, f"{patient_name}_{idx}.png"))
    #     combined_img2.save(os.path.join(save_dir_t2_flair_t1, split_dir, f"{patient_name}_{idx}.png"))
    #     combined_img3.save(os.path.join(save_dir_t1_t2_flair, split_dir, f"{patient_name}_{idx}.png"))

def prepare_lgggbm(args):
    data_root = args.data_root
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    test_data_dir = args.test_data_dir
    # train_data_dir = os.path.join(data_root, "")
    # val_data_dir = os.path.join(data_root, "../../data/BRATS2017/Brats17ValidationData")
    # test_data_dir = os.path.join(data_root, "../../data/BRATS2017/Brats17TestingData")

    # one to one tasks
    # save_dir_t2_flair = os.path.join(data_root, "T2_FLAIR")
    # save_dir_flair_t2 = os.path.join(data_root, "FLAIR_T2")
    save_dir_t1_t1ce = os.path.join(data_root, "T1_T1CE")

    # # many to one tasks
    # save_dir_t1_t2_flair = os.path.join(data_root, "T1_T2_FLAIR")
    # save_dir_t1_flair_t2 = os.path.join(data_root, "T1_FLAIR_T2")
    # save_dir_t2_flair_t1 = os.path.join(data_root, "T2_FLAIR_T1")

    # os.makedirs(save_dir_t2_flair, exist_ok=True)
    # os.makedirs(save_dir_flair_t2, exist_ok=True)
    os.makedirs(save_dir_t1_t1ce, exist_ok=True)
    # os.makedirs(save_dir_t1_t2_flair, exist_ok=True)
    # os.makedirs(save_dir_t1_flair_t2, exist_ok=True)
    # os.makedirs(save_dir_t2_flair_t1, exist_ok=True)


    train_split_dir = ["train", "val", "test"]
    for split_dir in train_split_dir:
        # os.makedirs(os.path.join(save_dir_t2_flair, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_flair_t2, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t1_t2_flair, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t1_flair_t2, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t2_flair_t1, split_dir), exist_ok=True)
        os.makedirs(os.path.join(save_dir_t1_t1ce, split_dir), exist_ok=True)

    # valid_ratio = 0.2
    # num_slices = 100

    # # brats17  有lgg和hgg
    # train_patients_name = []
    # for j in ['LGG', 'HGG']:
    #     train_patients_name+=os.listdir(os.path.join(train_data_dir, j))

    # train_patients_name = os.listdir(train_data_dir)
    # import pdb;pdb.set_trace()

    #防止和patient id并列的那一层里面有其他的文件
    # import pdb;pdb.set_trace()
    train_patients_name = []
    valid_patients_name = []
    test_patients_name = []
    for train_pre in os.listdir(train_data_dir):
        if os.path.isdir(os.path.join(train_data_dir, train_pre)):
            train_patients_name.append(train_pre)
    for val_pre in os.listdir(val_data_dir):
        if os.path.isdir(os.path.join(val_data_dir, val_pre)):
            valid_patients_name.append(val_pre)
    for test_pre in os.listdir(test_data_dir):
        if os.path.isdir(os.path.join(test_data_dir, test_pre)):
            test_patients_name.append(test_pre)

    # valid_patients_name = np.random.choice(train_patients_name, int(len(train_patients_name) * valid_ratio))
    # train_patients_name = list(set(train_patients_name) - set(valid_patients_name))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for split_dir in train_split_dir:
            if split_dir == "train":
                patients_name = train_patients_name

            elif split_dir == "val":
                patients_name = valid_patients_name
            elif split_dir == "test":
                patients_name = test_patients_name

            # debug
            # process_patient(patients_name[0], split_dir, train_data_dir, test_data_dir,
            #                                    save_dir_t2_flair, save_dir_flair_t2, save_dir_t1_t2_flair,
            #                                    save_dir_t1_flair_t2, save_dir_t2_flair_t1, num_slices)

            for patient_name in patients_name:
                futures.append(executor.submit(process_patient, patient_name, split_dir, train_data_dir, test_data_dir, val_data_dir, save_dir_t1_t1ce))

        for future in concurrent.futures.as_completed(futures):
            future.result()

def prepare_BRATS2017(args):
    data_root = args.data_root
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    test_data_dir = args.test_data_dir
    # train_data_dir = os.path.join(data_root, "")
    # val_data_dir = os.path.join(data_root, "../../data/BRATS2017/Brats17ValidationData")
    # test_data_dir = os.path.join(data_root, "../../data/BRATS2017/Brats17TestingData")

    # one to one tasks
    # save_dir_t2_flair = os.path.join(data_root, "T2_FLAIR")
    # save_dir_flair_t2 = os.path.join(data_root, "FLAIR_T2")
    save_dir_t1_t1ce = os.path.join(data_root, "T1_T1CE")

    # # many to one tasks
    # save_dir_t1_t2_flair = os.path.join(data_root, "T1_T2_FLAIR")
    # save_dir_t1_flair_t2 = os.path.join(data_root, "T1_FLAIR_T2")
    # save_dir_t2_flair_t1 = os.path.join(data_root, "T2_FLAIR_T1")

    # os.makedirs(save_dir_t2_flair, exist_ok=True)
    # os.makedirs(save_dir_flair_t2, exist_ok=True)
    os.makedirs(save_dir_t1_t1ce, exist_ok=True)
    # os.makedirs(save_dir_t1_t2_flair, exist_ok=True)
    # os.makedirs(save_dir_t1_flair_t2, exist_ok=True)
    # os.makedirs(save_dir_t2_flair_t1, exist_ok=True)


    train_split_dir = ["train", "val", "test"]
    for split_dir in train_split_dir:
        # os.makedirs(os.path.join(save_dir_t2_flair, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_flair_t2, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t1_t2_flair, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t1_flair_t2, split_dir), exist_ok=True)
        # os.makedirs(os.path.join(save_dir_t2_flair_t1, split_dir), exist_ok=True)
        os.makedirs(os.path.join(save_dir_t1_t1ce, split_dir), exist_ok=True)

    # valid_ratio = 0.2
    # num_slices = 100

    # brats17  有lgg和hgg
    train_patients_name = []
    for j in ['LGG', 'HGG']:
        train_patients_name+=os.listdir(os.path.join(train_data_dir, j))

    # train_patients_name = os.listdir(train_data_dir)
    # import pdb;pdb.set_trace()
    train_patients_name = os.listdir(train_data_dir) 
    valid_patients_name = os.listdir(val_data_dir)
    test_patients_name = os.listdir(test_data_dir)

    # valid_patients_name = np.random.choice(train_patients_name, int(len(train_patients_name) * valid_ratio))
    # train_patients_name = list(set(train_patients_name) - set(valid_patients_name))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for split_dir in train_split_dir:
            if split_dir == "train":
                patients_name = train_patients_name

            elif split_dir == "val":
                patients_name = valid_patients_name
            elif split_dir == "test":
                patients_name = test_patients_name

            # debug
            # process_patient(patients_name[0], split_dir, train_data_dir, test_data_dir,
            #                                    save_dir_t2_flair, save_dir_flair_t2, save_dir_t1_t2_flair,
            #                                    save_dir_t1_flair_t2, save_dir_t2_flair_t1, num_slices)

            for patient_name in patients_name:
                futures.append(executor.submit(process_patient, patient_name, split_dir, train_data_dir, test_data_dir, val_data_dir, save_dir_t1_t1ce))

        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare_Dataset')
    parser.add_argument('--data_root', dest='data_root', help='input directory for data root', type=str,
                        default='../../dataset/BRATS2017', required=False)
    parser.add_argument('--train_data_dir', dest='train_data_dir', help='input directory for data root', type=str,
                        default='../../dataset/BRATS2017/Brats17TrainingData', required=False)
    parser.add_argument('--val_data_dir', dest='val_data_dir', help='input directory for data root', type=str,
                        default='../../dataset/BRATS2017/Brats17ValidationData', required=False)
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='input directory for data root', type=str,
                        default='../../dataset/BRATS2017/Brats17TestingData', required=False)
    # parser.add_argument('--rand_seed', help='random seed', type=int, default=42)

    args = parser.parse_args()

    # prepare_BRATS2017(args)
    prepare_lgggbm(args)



"""
python t1t1c_prepare_dataset.py --data_root ../../data/TCGA_LGG_GBM_Brats2017 --train_data_dir ../../data/TCGA_LGG_GBM_Brats2017/train --val_data_dir ../../data/TCGA_LGG_GBM_Brats2017/val  --test_data_dir ../../data/TCGA_LGG_GBM_Brats2017/test

python t1t1c_prepare_dataset.py --data_root ../../dataset/TCGA_LGG_GBM --train_data_dir ../../dataset/TCGA_LGG_GBM/train --val_data_dir ../../dataset/TCGA_LGG_GBM/val  --test_data_dir ../../dataset/TCGA_LGG_GBM/test

python t1t1c_prepare_dataset.py --data_root ../../dataset/LGG_T --train_data_dir ../../dataset/LGG_T/train --val_data_dir ../../dataset/LGG_T/val  --test_data_dir ../../dataset/LGG_T/test
"""
    