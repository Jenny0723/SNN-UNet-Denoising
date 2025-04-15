# 📄 SNN-UNet 图像去噪项目使用指南

## 📌 项目简介
本项目实现了基于**脉冲神经网络 (SNN)** 的 **U-Net** 模型，用于图像去噪任务。该模型通过**知识蒸馏 (KD)** 方法从传统**人工神经网络 (ANN)** 中学习，实现高效低能耗的图像处理。

---

## 🛠️ 环境配置

### 📑 系统要求
- Python 3.8+
- CUDA 11.0+（用于 GPU 加速）
- 足够的 GPU 内存（建议 24GB）

### 📦 安装步骤

#### 创建 conda 环境
```bash
conda create -n snn-unet python=3.8
conda activate snn-unet
```

#### 安装 PyTorch 和 CUDA
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

#### 安装SpikingJelly
```bash
pip install spikingjelly
```

#### 安装其他依赖
```bash
pip install opencv-python scikit-image tqdm thop tensorboard
```

## 📂 数据准备与处理

### 📑 原始数据目录结构
```
data/
└── BSDS200/
    └── original_png/
        └── *.png
└── CBSD68/
    ├── original_png/
    │   └── *.png
    ├── noisy15/
    │   └── *.png
    ├── noisy25/
    │   └── *.png
    ├── noisy35/
    │   └── *.png
    ├── noisy45/
    │   └── *.png
    └── noisy50/
        └── *.png
```

### 🔄 数据处理流程

#### 1. 处理训练数据集
训练数据集使用BSDS200图像，通过以下脚本处理：

```bash
# 处理训练数据集
python create_train_dataset.py
```

此脚本会：
- 从`data/BSDS200/original_png`读取原始图像
- 对每张图像进行随机翻转
- 从每张图像中随机裁剪多个192×256大小的patch
- 为每个patch添加随机高斯噪声
- 将处理后的数据保存为HDF5格式：
  - `./data/benli/CBSD_dataset/CBSD_patch_diff_train.hdf5`（噪声图像）
  - `./data/benli/CBSD_dataset/CBSD_patch_diff_label.hdf5`（标签图像）

#### 2. 处理测试数据集
测试数据集使用CBSD68图像，通过以下脚本处理：

```bash
# 处理测试数据集
python create_test_dataset.py
```

此脚本会：
- 从`data/CBSD68/original_png`读取原始图像
- 将图像调整为480×320大小
- 从`data/CBSD68/noisy{15,25,35,45,50}`读取不同噪声级别的图像
- 将每张噪声图像分割为四个192×256大小的patch
- 将处理后的数据保存为HDF5格式：
  - `./data/benli/CBSD_dataset/CBSD_original_pictures.hdf5`（原始图像）
  - `./data/benli/CBSD_dataset/CBSD_patch_test_img_sigma_{15,25,35,45,50}.hdf5`（不同噪声级别的测试patch）

### 📝 注意事项
- 确保原始图像目录存在且包含PNG格式图像
- 处理脚本会自动创建所需的输出目录
- 如需重新生成数据集，可以先删除`./data/benli/CBSD_dataset`目录

## 📂 模型训练

### 📂 训练ANN模型（如果没有预训练模型）

```bash
python ann_train.py --train -n CBSD -b 8 -e 401 -lr 1e-4 -op adam
```

### 📂 使用知识蒸馏训练SNN模型

#### 使用SAKD方法进行知识蒸馏
```bash
python train_KD_main.py --train -n CBSD -b 8 -T 4 -e 401 -lr 1e-4 -op adam -a path/to/ann_model.pth --kd SAKD
```

#### 使用BKD方法进行知识蒸馏
```bash
python train_KD_main.py --train -n CBSD -b 8 -T 4 -e 401 -lr 1e-4 -op adam -a path/to/ann_model.pth --kd BKD
```

注意：
- 进行知识蒸馏时，需要提供ANN预训练模型路径
- ANN预训练模型路径为`/data1/graduation/model/benli/CBSD/CBSD_1e_4.pth`

#### 不使用知识蒸馏直接训练SNN
```bash
python train_KD_main.py --train -n CBSD -b 8 -T 4 -e 401 -lr 1e-4 -op adam
```

### 📂 参数说明
-n: 数据集名称
-b: 批次大小
-T: SNN时间步数
-e: 训练轮数
-lr: 学习率
-op: 优化器
-a: ANN预训练模型路径
--kd: 知识蒸馏方法(SAKD或BKD)


### 📂 评估SNN模型
```bash
python train_KD_main.py --evaluate -s path/to/snn_model.pth
```

注意：
- 评估SNN模型时，需要提供SNN模型路径
- SNN模型路径为`/data1/graduation/model/benli/CBSD/CBSD_snn_xxx.pth`

## 🖥️ 使用 Screen 进行长时间训练

### 📌 基本 Screen 命令

#### 重新连接到 Screen 会话
```bash
# 通过名称重新连接
screen -r snn
```

#### 在 Screen 会话中运行训练
```bash
# 训练ANN模型
python code/ann/Spiking-UNet-master/ann_train.py --train -n CBSD -b 8 -e 401 -lr 1e-4 -op adam

# 直接训练SNN
python code/ann/Spiking-UNet-master/train_KD_main.py --train -b 8 -T 4 -e 401 -lr 1e-4 -op adam

# 使用SAKD方法进行知识蒸馏脚本
python code/ann/Spiking-UNet-master/train_KD_main.py --train -b 8 -T 4 -e 401 -lr 1e-4 -op adam -a model/benli/CBSD/CBSD_1e_4.pth --kd SAKD
```

## 📂 模型权重与评估结果

### 📑 模型权重文件
所有训练好的模型权重文件存放在以下目录：
```
/data1/graduation/model/benli/CBSD/
```

该目录包含：
- ANN模型权重：`CBSD_1e_4.pth`
- SNN模型权重：`CBSD_snn_xxx.pth`（其中xxx表示不同的训练配置）

### 📊 评估结果
评估结果存放在以下目录：

#### ANN评估结果
```
/data1/graduation/new_results/ann/ann/CBSD/64/
```
该目录包含ANN模型在不同噪声级别下的去噪结果、性能指标和能耗。

#### SNN的评估结果
```
/data1/graduation/snn_KD_results/
```
该目录包含SNN模型的评估结果，包括：
- 去噪后的图像
- PSNR和SSIM指标
- 能耗统计(待完成)

### 💾 使用预训练模型
要使用预训练模型进行评估，请指定相应的模型路径：
```bash
# 评估ANN模型
python code/ann/Spiking-UNet-master/ann_train.py -v 1e_4
```

```bash
# 评估SNN模型
python code/ann/Spiking-UNet-master/train_KD_main.py --evaluate -s model/benli/CBSD/CBSD_snn_xxx.pth
```

注意：请将`CBSD_snn_xxx.pth`替换为实际的SNN模型文件名。
