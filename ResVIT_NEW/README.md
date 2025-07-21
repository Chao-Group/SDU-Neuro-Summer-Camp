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
查看train.sh

2、测试
查看test.sh

