# 标准 SRCNN 实验

基于 DIV2K 数据集实现的图像超分辨率实验，包含 **SRCNN** 和 **EDSR** 双模型实现。

## 项目结构

```
srcnn/
├── 动手实现图像超分辨率实验手册.pdf          ← 实验手册（对照学习）
│
├── src/                                      ← 源代码
│   ├── model.py                              ← SRCNN 模型定义
│   ├── model_edsr.py                         ← EDSR 模型定义（增强版）
│   ├── dataset.py                            ← SRCNN 数据加载（DIV2K）
│   ├── dataset_edsr.py                       ← EDSR 数据加载
│   ├── train.py                              ← SRCNN 训练脚本
│   ├── train_edsr.py                         ← EDSR 训练脚本
│   ├── infer.py                              ← SRCNN 推理脚本
│   ├── infer_edsr.py                         ← EDSR 推理脚本
│   ├── test.py                               ← SRCNN 测试（Set5 / Set14）
│   ├── test_edsr.py                          ← EDSR 测试脚本
│   └── utils.py                              ← 公共工具函数
│
├── data/                                     ← 数据集（运行前准备）
│   ├── DIV2K/
│   │   ├── DIV2K_train_HR/                   ← 800 张训练图 (.jpeg)
│   │   └── DIV2K_valid_HR/                   ← 100 张验证图 (.jpeg)
│   ├── benchmark/
│   │   ├── Set5/HR/                          ← 5 张标准测试图 (.png)
│   │   └── Set14/HR/                         ← 14 张标准测试图 (.png)
│   └── raw/
│       └── SelfExSR/                         ← SelfExSR MATLAB 参考实现
│
└── outputs/                                  ← 输出（运行后生成）
    ├── checkpoints/                          ← SRCNN 模型权重
    │   ├── last.pth                          ← 最新权重
    │   └── best.pth                          ← 最佳权重
    ├── checkpoints_edsr/                     ← EDSR 模型权重
    │   ├── last.pth
    │   └── best.pth
    ├── results/                              ← SRCNN 超分结果图
    │   ├── Set5/
    │   └── Set14/
    └── results_edsr/                         ← EDSR 超分结果图
        ├── Set5/
        └── Set14/
```

## 快速开始

```bash
# 1. 准备数据（将 DIV2K 和 benchmark 放入 data/ 目录）
#    DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
#    Set5/Set14: 标准超分辨率 benchmark

# 2. 训练 SRCNN
python src/train.py

# 3. 测试
python src/test.py

# 4. 推理
python src/infer.py
```

## 模型对比

| 模型 | 论文 | 特点 |
|------|------|------|
| SRCNN | Dong et al. 2015 | 三层卷积，开创性工作 |
| EDSR | Lim et al. 2017 | 残差块 + 无 BN，精度更高 |

## 命令行参数

所有路径均通过 `argparse` 传入，无硬编码：

```bash
# 训练（可覆盖默认路径）
python src/train.py --train-dir data/DIV2K/DIV2K_train_HR \
                    --save-dir outputs/checkpoints

# 测试
python src/test.py --checkpoint outputs/checkpoints/best.pth \
                   --set5-dir data/benchmark/Set5/HR \
                   --set14-dir data/benchmark/Set14/HR \
                   --output-dir outputs/results

# 推理
python src/infer.py --input your_image.jpg \
                    --checkpoint outputs/checkpoints/best.pth \
                    --output outputs/results/infer.png
```
