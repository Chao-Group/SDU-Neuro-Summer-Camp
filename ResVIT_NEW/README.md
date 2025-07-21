# Image Generation（Translation）ResViT的BraTS_TCGA_2017_lgg_gbm


## 环境搭建

0、python 3.8

1、安装pytorch，使用pytorch==1.12.0 cuda=10.2 这个配置，安装命令：
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

2、通过以下命令安装相应的python库
```
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ml_collections -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3、直接cd到文件夹中
```bash
cd ResViT_NEW
```

4、下载预训练权重
* [Pre-trained ViT models](https://console.cloud.google.com/storage/vit_models/):
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
```
**这个预训练权重的路径需要在ResViT/models/transformer_configs.py中的第29行修改，替换成自己的路径**

## 数据集准备
0、准备BRATS_2017数据集
原论文是支持one to one和many to one的，我们这里简化，只用one to one task

主要含义如下：

T1_T1C 对应 从T1和T1C单个模态生成T1C的one to one task

...
1、训练



# for each dataset
# 0、[统一T1T1C|CTCTC结构] 用resvit把 nii或nii.gz处理成  png, 统一的结构  datasetname/T1_T1CE/train|val|test 
# （1、去处理各自网络的dataloader->一次性写好）    
# 2、for each model:  train test val 都可以看train.sh (里面会包含环境, train的方法，test的方法，metric的方法，结果保存)
# ————————————复现stage↑  下游验证stage↓
# 3、（写在resvit里面）对上面复现生成的结果 png->nii
# 4、 分割 可视化结果  metrics
# 5、 分类 metrics


# 环境
conda activate dimba
pip install nibabel h5py dominate ml_collections


# 下载预训练权重
# * [Pre-trained ViT models](https://console.cloud.google.com/storage/vit_models/):
# ```bash
# wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
# ```
# **这个预训练权重的路径需要在ResViT/models/transformer_configs.py中的第29行修改，替换成自己的路径**

# train
# （1）预训练pretrain, 修改dataroot和checkpoints_dir    name     || 
# ckpt:   {checkpoints_dir}/{name}
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../../dataset/TCGA_LGG_GBM/T1_T1CE_Brats2017/ --name pre_train --gpu_ids 0 --model resvit_one \
      --which_model_netG res_cnn  --which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch \
      --pool_size 0 --output_nc 1 --input_nc 1 --loadSize 256 --fineSize 256  --niter 50 --niter_decay 50 \
      --save_epoch_freq 5 --checkpoints_dir ckpt/TCGA_LGG_GBM --display_id 0 --lr 0.0002

CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../../data/TCGA_LGG_GBM_Brats2017/T1_T1CE/ --name pre_train --gpu_ids 0 --model resvit_one \
      --which_model_netG res_cnn  --which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch \
      --pool_size 0 --output_nc 1 --input_nc 1 --loadSize 256 --fineSize 256  --niter 50 --niter_decay 50 \
      --save_epoch_freq 5 --checkpoints_dir ckpt/TCGA_LGG_GBM_Brats2017 --display_id 0 --lr 0.0002

# finetune:  
# (2)修改dataroot和checkpoints_dir    name    ; pre_trained_path的参数替换成上一步生成的latest_net_G.pth的路径
# ckpt: {checkpoints_dir}/{name}
# 这一步训练后，会在checkpoints_dir路径下保存好最终的模型latest_net_G.pth和latest_net_D.pth
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../../data/TCGA_LGG_GBM_Brats2017/T1_T1CE/ --name finetune --gpu_ids 0 --model resvit_one \
      --which_model_netG resvit --which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 \
      --output_nc 1 --input_nc 1 --loadSize 256 --fineSize 256 --niter 25 --niter_decay 25 --save_epoch_freq 5 \
      --checkpoints_dir ckpt/TCGA_LGG_GBM_Brats2017 --display_id 0 --pre_trained_transformer 1 --pre_trained_resnet 1 \
      --pre_trained_path ckpt/TCGA_LGG_GBM_Brats2017/pre_train/latest_net_G.pth --lr 0.001
    #   --pre_trained_path ckpt/TCGA_LGG_GBM/TCGA_LGG_GBM_T1_T1CE_pre_trained/latest_net_G.pth --lr 0.001


# test   修改dataroot和checkpoints_dir    name     
# --results_dir 表示结果保存路径: {results_dir}/{name}/test_latest/images  : ckpt/TCGA_LGG_GBM/test/finetune/test_latest/images/TCGA-19-2624_0_fake_B.png
#   ckpt: {checkpoints_dir}/{name} 


CUDA_VISIBLE_DEVICES=0 python3 test.py --dataroot ../../data/TCGA_LGG_GBM_Brats2017/T1_T1CE/ --name finetune --gpu_ids 0 --model resvit_one --which_model_netG resvit \
        --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 10000 --serial_batches \
        --fineSize 256 --loadSize 256 --results_dir ckpt/TCGA_LGG_GBM_Brats2017/test/100_net/ \
       --checkpoints_dir ckpt/TCGA_LGG_GBM_Brats2017 --which_epoch latest

# metrics
CUDA_VISIBLE_DEVICES=0 python test/test_SSIM.py --fold_test ckpt/TCGA_LGG_GBM_Brats2017/test/100_net/finetune/test_latest/images/ --dataset_name TCGA_LGG_GBM_Brats2017

CUDA_VISIBLE_DEVICES=4 python test/test_SSIM.py --fold_test ckpt/TCGA_LGG_GBM/test/finetune/test_latest/images/ --dataset_name TCGA_LGG_GBM



2、测试


