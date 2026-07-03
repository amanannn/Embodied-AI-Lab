# 标准 SRCNN 实验

基于 DIV2K 数据集实现的图像超分辨率实验，包含 SRCNN 和 EDSR 两个模型。

## 文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | SRCNN 模型定义 |
| `model_edsr.py` | EDSR 模型定义（增强版） |
| `dataset.py` | 数据集加载（DIV2K） |
| `dataset_edsr.py` | EDSR 数据集加载 |
| `train.py` | SRCNN 训练脚本 |
| `train_edsr.py` | EDSR 训练脚本 |
| `infer.py` | SRCNN 推理脚本 |
| `infer_edsr.py` | EDSR 推理脚本 |
| `test.py` | SRCNN 测试脚本 |
| `test_edsr.py` | EDSR 测试脚本 |
| `utils.py` | 公共工具函数 |

## 用法

```bash
# 下载数据
bash ../code/download_cifar10.sh

# 训练 SRCNN
python train.py

# 推理
python infer.py
```
