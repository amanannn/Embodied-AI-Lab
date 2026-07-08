# MedicalGPT\_上机课任务书\_final

# MedicalGPT 医疗大模型训练实操课

课程时间：周六 08:30\-12:00，14:00\-17:30
服务器配置：Ubuntu，Tesla V100\-SXM2\-32GB
项目地址：https://github\.com/shibing624/MedicalGPT
本任务书基于 MedicalGPT 当前 `main` 分支编写

# Introduction

什么是 MedicalGPT？

从名字看，**MedicalGPT** 是一个面向医疗领域的大模型训练项目。它不是单纯调用一个现成聊天模型，而是把一个通用基座模型，按照大模型训练中常见的流程继续加工：先用领域文本做增量预训练，让模型更熟悉医疗语料；再用问答数据做有监督微调，让模型学会按医生问答的方式回复；最后用偏好数据做对齐训练，让模型更倾向于清晰、安全、有帮助的回答。

这条路线在流程结构上类似工业界训练 ChatGPT 类模型，但课堂里会把模型规模、训练样本数和训练步数大幅缩小，只做教学级验证。这样做的目的不是在一天内训练出一个能上线的医疗专家模型，而是让大家完整经历一次大模型工程训练闭环：

> 数据集 → 模型 → 参数 → 训练 → 日志 → 合并 → 推理 → 对比 → 总结

# 课程目标

1. 完成服务器登录、GPU 检查、Python 环境创建。

2. 阅读 MedicalGPT 项目结构，理解 PT、SFT、DPO/RLHF、推理和部署脚本的关系。

3. 使用医疗预训练文本做增量预训练 PT，得到 LoRA adapter。

4. 合并 PT adapter 到基座模型，得到可继续训练的中间模型。

5. 使用医疗问答数据做 SFT 有监督微调，得到医疗问答 LoRA adapter。

6. 使用命令行或 Gradio 完成训练前后推理对比。

7. 使用偏好数据做 DPO 对齐训练，理解 chosen/rejected 数据格式。

8. 记录训练参数、loss 曲线、显存占用、推理样例和错误排查过程。

> 面向对象：医疗大模型训练入门项目

> 当前服务器：Ubuntu \+ Tesla V100\-SXM2\-32GB

> 实操路线：**SSH 连接服务器 → 检查 Ubuntu/GPU/驱动 → 创建 Conda 环境 → 安装 PyTorch 和 MedicalGPT 依赖 → 检查模型与数据集 → PT 增量预训练 → 合并 LoRA → SFT 有监督微调 → 推理对比 → DPO 偏好对齐 → 实验报告**

> 版本建议：

> NVIDIA GPU: Tesla V100\-SXM2\-32GB
Python: 3\.10
PyTorch: 建议 cu121 版本
MedicalGPT: GitHub main 分支
训练精度：FP16

> 注意：`nvidia-smi` 里显示的 `CUDA Version` 表示驱动最高支持的 CUDA 版本，**不代表 PyTorch 必须安装完全相同的 CUDA 版本**。本课建议安装 PyTorch cu121 版本；只要 `torch.cuda.is_available()` 返回 `True`，并且能识别 V100，就可以继续后续任务。

# 配置 MedicalGPT 训练环境

## 安装顺序

**先确认硬件和驱动，再安装软件依赖，最后进入项目和数据检查**。不要在 GPU、驱动或磁盘状态还没有确认前直接安装 Python 包，否则后续报错时很难判断是硬件问题、驱动问题还是依赖问题。

1. SSH 登录服务器，确认账号、IP、端口可以正常连接。

2. 使用 VS Code Remote SSH 连接服务器，便于查看代码、数据和日志。

3. 检查 Ubuntu 版本、GPU 是否挂载、NVIDIA 驱动是否可用、显存是否被占用。

4. 如果 `nvidia-smi` 异常，先按“NVIDIA 驱动安装与验证”排查；只有硬件层确认正常后再继续。

5. 检查基础命令是否可用：`git`、`wget` 或 `curl`、`tar`、`bash`、`nvidia-smi`。

6. 安装或确认 Miniconda / Anaconda。

7. 创建独立的 `medicalgpt` Python 3\.10 环境。

8. 在 `medicalgpt` 环境中升级 `pip`，安装 PyTorch CUDA 版本，并验证 `torch.cuda.is_available()`。

9. 在同一个环境中安装 Hugging Face CLI，并验证 `hf` 命令可用。

10. 克隆 MedicalGPT 仓库，进入项目目录。

11. 安装 MedicalGPT 项目依赖：`requirements.txt`、Gradio / FastAPI / TensorBoard、DPO/RM 相关补充包。

12. 下载或确认本地基座模型目录。

13. 检查 PT、SFT、DPO/RM 数据格式和文件数量。

14. 运行 PT 增量预训练、合并 LoRA、SFT、推理对比、DPO/RM 扩展任务。

15. 整理实验报告，记录硬件、依赖版本、训练参数、loss、显存和推理样例。

## SSH 登录服务器

### 强烈建议使用 VS Code 远程连接服务器

1. 本地安装 VS Code。

2. 在扩展中安装 `Remote - SSH`。

3. 点击左侧远程连接图标。

4. 新增 SSH Host。

5. 输入：

```Bash
ssh ubuntu@1.2.3.4
```

或：

```Bash
ssh -p 2222 ubuntu@1.2.3.4
```

注意：VS Code 中填写时，把 `1.2.3.4` 和 `2222` 替换成教师发放的真实服务器 IP 和 SSH 端口。账号和密码以课堂发放信息为准，本任务书示例沿用 `ubuntu` 和 `Imglab123` 作为课堂账号样例。

6. 选择 Linux。

7. 输入密码Imglab123。

8. 打开服务器上的 `MedicalGPT` 项目目录。

使用 VS Code 的好处是可以直接查看 jsonl 数据、训练脚本和日志文件，但所有训练命令仍建议在终端里执行。

## 查看系统和 GPU

### 查看 Ubuntu 版本

```Bash
cat /etc/os-release
```

或者：

```Bash
lsb_release -a
```

### 查看 GPU 和驱动

```Bash
nvidia-smi
```

成功状态应包含：

```Plaintext
GPU: Tesla V100-SXM2-32GB
Memory: 32768MiB 左右
```

注：

`nvidia-smi` 中的 `CUDA Version` 是驱动支持的最高 CUDA 能力。PyTorch 自带 CUDA runtime，所以 PyTorch 的 CUDA 版本不一定要和这个数字完全一样。

## NVIDIA 驱动安装与验证

> 这部分是上机课最容易卡住的地方。只有 `nvidia-smi` 正常，后面的 PyTorch、MedicalGPT 训练才有意义。

> 如果学生已经看到 `nvidia-smi` 正常显示 V100，可以跳过安装驱动，直接进入 Python 环境配置。

### 1\. 先确认服务器是否识别到 GPU

运行：

```Bash
lspci | grep -i nvidia
```

正常应看到类似：

```Plaintext
3D controller: NVIDIA Corporation GV100GL [Tesla V100 SXM2 32GB]
```

原因：

这一步确认硬件层面 GPU 是否被云服务器挂载。如果这里没有任何 NVIDIA 输出，说明不是 Python 或驱动问题，而是云服务器没有分配 GPU，或者 GPU 没有正确挂载，需要联系云平台或教师处理。

### 2\. 检查 nvidia\-smi

运行：

```Bash
nvidia-smi
```

正常输出应包含：

```Plaintext
NVIDIA-SMI
Driver Version
CUDA Version
Tesla V100-SXM2-32GB
Memory-Usage
```

如果输出正常，继续后续任务。

如果出现：

```Plaintext
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

说明 NVIDIA 用户态工具无法和内核驱动通信。常见原因包括：

1. 只装了部分 NVIDIA 包，没有完整驱动。

2. `nvidia-smi` 版本和内核模块版本不一致。

3. DKMS 新模块已经安装，但服务器还没重启，当前仍加载旧模块。

4. 云服务器镜像里残留过旧的手工安装驱动。

### 3\. 检查当前 NVIDIA 内核模块

运行：

```Bash
lsmod | grep '^nvidia'
```

如果有输出，说明 NVIDIA 内核模块已经加载，例如：

```Plaintext
nvidia_uvm
nvidia_drm
nvidia_modeset
nvidia
```

再查看内核模块版本：

```Bash
cat /proc/driver/nvidia/version
```

示例：

```Plaintext
NVRM version: NVIDIA UNIX x86_64 Kernel Module 535.216.01
```

原因：

`nvidia-smi` 能否工作，不仅取决于有没有 GPU，也取决于当前加载的 NVIDIA 内核模块版本是否和用户态工具匹配。

### 4\. 检查系统已安装的 NVIDIA 包

运行：

```Bash
dpkg -l | grep -E 'nvidia-driver|nvidia-dkms|nvidia-utils|libnvidia-compute'
```

正常情况下应该能看到类似：

```Plaintext
nvidia-driver-535-server
nvidia-dkms-535-server
nvidia-utils-535-server
libnvidia-compute-535-server
```

如果只看到：

```Plaintext
nvidia-container-toolkit
nvidia-docker2
```

但没有 `nvidia-driver`、`nvidia-dkms`、`nvidia-utils`，说明系统只有容器工具，没有完整 NVIDIA 驱动。

### 5\. 查看 Ubuntu 推荐驱动

运行：

```Bash
ubuntu-drivers devices
```

V100 服务器可能看到类似：

```Plaintext
driver   : nvidia-driver-550 - distro non-free recommended
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-535 - distro non-free
```

原因：

Ubuntu 会列出当前硬件可用的驱动版本。`recommended` 是系统推荐版本，但云服务器上为了稳定，也常用 server 版本驱动，例如 `nvidia-driver-535-server`。

### 6\. 安装 NVIDIA 驱动

先更新 apt 索引：

```Bash
sudo apt-get update
```

安装 V100 常用的 535 server 驱动：

```Bash
sudo apt-get install -y nvidia-driver-535-server
```

这条命令会安装一组相关组件，包括：

```Plaintext
nvidia-driver-535-server
nvidia-dkms-535-server
nvidia-utils-535-server
libnvidia-compute-535-server
```

原因：

1. `nvidia-driver-535-server` 是完整驱动元包。

2. `nvidia-dkms-535-server` 负责为当前 Linux 内核编译 NVIDIA 内核模块。

3. `nvidia-utils-535-server` 提供 `nvidia-smi`。

4. `libnvidia-compute-535-server` 提供 CUDA 计算相关用户态库。

安装过程中如果看到：

```Plaintext
Building initial module for 5.15.0-xxx-generic
Done.
```

说明 DKMS 正在为当前内核编译 NVIDIA 驱动模块，这是正常现象。

如果看到：

```Plaintext
WARNING: Your driver installation has been altered since it was initially installed
```

说明这个云镜像里可能残留过非 apt 方式安装的 NVIDIA 驱动。通常 apt 会尝试清理旧安装，继续等待安装完成即可。

### 7\. 安装后检查 DKMS 状态

运行：

```Bash
sudo dkms status
```

正常输出类似：

```Plaintext
nvidia-srv/535.xxx.xx, 5.15.0-xxx-generic, x86_64: installed
```

原因：

这一步确认 NVIDIA 驱动模块已经为当前内核编译并安装。

### 8\. 重启终端

驱动安装完成后，需要重新启动终端

原因：

即使 DKMS 新模块已经安装完成，当前运行中的内核可能仍然加载旧版本 NVIDIA 模块。重启后会加载新安装的驱动模块。

### 9\. 重启后验证

重新 SSH 登录服务器后，运行：

```Bash
nvidia-smi
```

再运行：

```Bash
cat /proc/driver/nvidia/version
sudo dkms status
```

三者应保持一致：

1. `nvidia-smi` 能显示 V100。

2. `/proc/driver/nvidia/version` 显示当前加载的 NVIDIA 模块版本。

3. `dkms status` 显示该版本模块为 `installed`。

### 10\. 如果安装后仍失败

先确认是不是没重启：

```Bash
cat /proc/driver/nvidia/version
sudo dkms status
```

如果 `dkms status` 里是新版本，但 `/proc/driver/nvidia/version` 仍是旧版本，说明当前还没加载新模块，需要执行：

```Bash
sudo reboot
```

如果重启后仍失败，继续查看日志：

```Bash
journalctl -u nvidia-persistenced --no-pager -n 80
dmesg | grep -i nvidia | tail -80
```

如果日志中出现 Secure Boot、module verification、nouveau 冲突等信息，不要自行卸载驱动，记录报错并联系授课人员处理。

还需要检查 `/dev/nvidia*` 设备节点：

```Bash
ls -l /dev/nvidia*
```

正常应看到类似：

```Plaintext
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
/dev/nvidia-uvm-tools
```

如果 `cat /proc/driver/nvidia/version`、`sudo dkms status` 都正常，但 `/dev/nvidia*` 不存在，说明驱动模块已经加载，但字符设备节点没有生成。先安装并运行 `nvidia-modprobe`：

```Bash
sudo apt-get install -y nvidia-modprobe
sudo nvidia-modprobe -u -c=0
sudo nvidia-modprobe
ls -l /dev/nvidia*
nvidia-smi
```

如果仍然没有 `/dev/nvidia*`，可查看 NVIDIA 主设备号：

```Bash
cat /proc/devices | grep -E 'nvidia|uvm'
cat /proc/driver/nvidia/gpus/*/information
```

在单卡 V100 上通常是：

```Plaintext
195 nvidia-frontend
234 nvidia-uvm
Device Minor: 0
```

可以手动创建设备节点：

```Bash
sudo mknod -m 666 /dev/nvidia0 c 195 0
sudo mknod -m 666 /dev/nvidiactl c 195 255
sudo mknod -m 666 /dev/nvidia-uvm c 234 0
sudo mknod -m 666 /dev/nvidia-uvm-tools c 234 1
ls -l /dev/nvidia*
nvidia-smi
```

原因：

`nvidia-smi` 需要通过 `/dev/nvidia0`、`/dev/nvidiactl` 等字符设备访问内核驱动。驱动模块存在但设备节点缺失时，`nvidia-smi` 仍会报无法通信。

如果普通用户执行 `nvidia-smi` 失败，但 `sudo nvidia-smi` 成功，检查设备节点权限：

```Bash
sudo ls -l /dev/nvidia*
```

正常训练环境建议至少保证 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 对当前用户可读写。上面的 `mknod -m 666` 已经给了读写权限。

### 常用排错命令

查看 GPU：

```Bash
nvidia-smi
```

查看磁盘：

```Bash
df -h
```

查看内存：

```Bash
free -h
```

查看 Python：

```Bash
python --version
```

查看 Conda 环境：

```Bash
conda env list
```

查看当前环境包：

```Bash
pip list
```

## 软件依赖安装清单

课堂主线需要的软件按依赖层次分为 6 类，建议逐项安装并验证：

1. 远程连接工具：本地 VS Code、Remote \- SSH 扩展、服务器 SSH 账号。

2. 系统基础工具：`git`、`wget` 或 `curl`、`tar`、`bash`、`nvidia-smi`。

3. GPU 驱动层：NVIDIA 驱动、内核模块、`/dev/nvidia*` 设备节点。

4. Python 环境层：Miniconda / Anaconda、`medicalgpt` Conda 环境、Python 3\.10、`pip`。

5. 深度学习框架层：PyTorch、torchvision、torchaudio，课堂建议使用 cu121 版本并统一 FP16。

6. 项目依赖层：MedicalGPT `requirements.txt`、`huggingface_hub[cli]`、`gradio`、`fastapi`、`uvicorn`、`tensorboard`、`trl==0.24.0`、`mergekit`、`llm-blender`、`weave`。

> 依赖安装原则：所有 Python 包都安装在 `medicalgpt` 环境中，不要装到系统 Python 或 `(base)` 环境中。

## 安装 Miniconda

如果服务器上没有 `conda`，先安装 Miniconda。检查命令：

```Bash
which conda
conda --version
```

如果没有输出，下载固定版本 Miniconda：

```Bash
cd ~
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
bash miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda --version
```

创建课程环境：

```Bash
~/miniconda3/bin/conda create -n medicalgpt python=3.10 -y
source ~/miniconda3/bin/activate medicalgpt
python --version
```

## 安装 PyTorch

进入环境：

```Bash
source ~/miniconda3/bin/activate medicalgpt
```

安装 PyTorch 2\.5\.1 \+ cu121：

```Bash
python -m pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

验证：

```Bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

V100 上预期：

```Plaintext
cuda available: True
gpu: Tesla V100-SXM2-32GB
capability: (7, 0)
```

建议额外确认计算能力：

```Bash
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16 supported flag:", torch.cuda.is_bf16_supported())
PY
```

实测中，V100 的 `torch.cuda.is_bf16_supported()` 可能返回 `True`，但 V100 的计算能力是 `(7, 0)`，没有 Ampere/A100 之后的原生 BF16 Tensor Core。为了课堂稳定，本任务书仍统一使用：

```Plaintext
--fp16 True
--torch_dtype float16
```

不建议在 V100 上直接照搬官方脚本里的 `--bf16` / `bfloat16` 设置。

注意：

`nvidia-smi` 显示的 CUDA Version 是驱动支持能力，不代表 PyTorch 必须安装同数字版本。驱动 535 \+ PyTorch cu121 是可以工作的组合。

## 安装 Hugging Face CLI

下载模型或扩展数据集前，必须先进入 `medicalgpt` 环境。不要在系统 Python 或 `(base)` 环境里安装 `huggingface_hub`。

```Bash
source ~/miniconda3/bin/activate medicalgpt
python -m pip install -U "huggingface_hub[cli]"
which hf
hf --help | head
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

说明：`hf` 命令来自 `huggingface_hub` 包。先创建 Conda 环境，再安装 CLI，后续模型下载、数据集下载和登录命令都会落在同一个课程环境中。

## 获取代码并安装 MedicalGPT 依赖

硬件、驱动、Conda 和 PyTorch 都验证通过后，再获取项目代码并安装项目依赖：

```Bash
cd ~
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
git log -1 --oneline

conda activate medicalgpt
python -m pip install --upgrade pip
pip install -r requirements.txt --upgrade
pip install gradio fastapi uvicorn tensorboard
pip install "trl==0.24.0" mergekit llm-blender weave
```

说明：MedicalGPT 当前 `requirements.txt` 已包含 `accelerate`、`datasets`、`peft`、`sentencepiece`、`scikit-learn`、`tensorboard`、`transformers`、`trl` 等核心依赖。课堂环境中为了兼容 V100、PyTorch 2\.5\.1 和后文 DPO/RM 示例，仍建议显式固定 `trl==0.24.0`，并补充安装 `mergekit`、`llm-blender`、`weave`。

如果要运行项目自带测试，再安装：

```Bash
pip install pytest
```

如果安装过程出现 `transformers`、`trl`、`torch` 版本冲突，先不要反复升级所有包，优先按“常见问题与处理”中的固定版本方案排查。

## 安装后完整性检查

完成软件安装后，先执行下面的检查，再进入训练任务：

```Bash
# 1) 基础系统和 GPU
hostname
cat /etc/os-release
which git wget curl tar bash
nvidia-smi

# 2) Conda 和 Python 环境
which conda || echo "conda not found"
conda env list
conda activate medicalgpt
which python
python --version
python -m pip --version

# 3) PyTorch 和 CUDA
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'None')
PY

# 4) Hugging Face CLI
which hf
hf --help | head

# 5) MedicalGPT 项目依赖
cd ~/MedicalGPT
python - <<'PY'
mods = ['transformers', 'datasets', 'accelerate', 'peft', 'trl', 'tensorboard', 'gradio', 'fastapi']
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(name, 'MISSING', e)
PY

# 6) 模型和数据目录
ls -lah /home/ubuntu/models/Qwen3.5-0.8B
ls -lah data/pretrain data/sft data/reward
```

如果任意一项失败，先停在当前步骤排查，不要继续运行训练脚本。

## 一些常见的坑

### 坑 1：V100 不要直接照抄官方 bf16 命令

MedicalGPT 官方脚本中有不少参数使用：

```Plaintext
--bf16
--torch_dtype bfloat16
```

但 Tesla V100 不支持原生 BF16。本课统一使用：

```Plaintext
--fp16 True
--torch_dtype float16
```

### 坑 2：官方脚本默认双卡，课堂是单卡

官方脚本常见写法是双卡 `torchrun`，例如设置 `CUDA_VISIBLE_DEVICES=0,1` 并使用 `--nproc_per_node 2`。如果每组只分到一张 GPU，实际训练命令必须使用本任务书后文给出的完整单卡命令，即设置 `CUDA_VISIBLE_DEVICES=0` 并直接运行对应的 Python 训练脚本。

### 坑 3：不要在 base 环境里乱装依赖

不要直接在 `(base)` 里安装 PyTorch、Transformers、TRL 等包。课程中统一创建：

```Bash
conda create -n medicalgpt python=3.10 -y
conda activate medicalgpt
```

这样即使装错版本，也可以删除环境重来，不会污染系统环境。

### 坑 4：云服务器可能存在 NVIDIA 驱动用户态和内核态版本不一致

实测中遇到过这种情况：

```Plaintext
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

但同时：

```Bash
lspci | grep -i nvidia
lsmod | grep '^nvidia'
cat /proc/driver/nvidia/version
```

又能看到 V100 和已经加载的 NVIDIA 内核模块。这个问题通常不是 GPU 不存在，而是云镜像里 NVIDIA 驱动安装不完整，或者 `nvidia-smi` 用户态工具与当前内核模块版本不一致。

排查顺序：

```Bash
lspci | grep -i nvidia
lsmod | grep '^nvidia'
cat /proc/driver/nvidia/version
dpkg -l | grep -E 'nvidia-driver|nvidia-dkms|nvidia-utils|libnvidia-compute'
ubuntu-drivers devices
```

如果看到 DKMS 模块版本和 `nvidia-utils` 版本不一致，可以安装匹配的 server 驱动。以 Ubuntu 22\.04 \+ V100 为例：

```Bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535-server
sudo reboot
```

重启后再次验证：

```Bash
nvidia-smi
cat /proc/driver/nvidia/version
sudo dkms status
```

# MedicalGPT 项目理解

MedicalGPT 是一个面向医疗领域大模型训练的项目，实现了 ChatGPT Training Pipeline 中常见的多个阶段：

1. PT，Continue PreTraining，增量预训练。输入为普通文本或 `{"text": "..."}` 形式的 jsonl，用于让模型继续适应领域语料分布。

2. SFT，Supervised Fine\-tuning，有监督微调。输入为 ShareGPT 格式多轮问答 jsonl，用于让模型学会按指令回答。

3. RM/PPO，传统 RLHF 路线。先训练奖励模型，再用 PPO/RLOO 强化学习优化。

4. DPO，Direct Preference Optimization。直接使用 chosen/rejected 偏好数据优化模型，课堂中选择 DPO 作为对齐阶段主线，因为流程更短、稳定性更好。

5. ORPO、GRPO、OPD，扩展训练方法。课堂主线不强制完成，但作为拓展阅读和加分任务。

6. demo，包含命令行推理、Gradio 页面、FastAPI 服务、RAG 文件问答等示例。

7. tools，包含数据转换、jsonl 校验、LoRA 合并、量化等工具。

本次课程主线为：

```Plaintext
服务器连接 -> 环境安装 -> 数据检查 -> PT -> 合并 PT -> SFT -> 推理对比 -> DPO -> 推理对比 -> 总结报告
```

# 硬件与模型选择

Tesla V100\-SXM2\-32GB 支持 FP16 训练，但不适合直接使用 BF16。MedicalGPT 官方脚本中很多命令使用了 `--bf16`，课堂中统一改成：

```Plaintext
--fp16 True
--torch_dtype float16
```

同时，官方脚本默认 `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2`，适合双卡。本次如果每组使用单卡，统一使用后文各训练阶段给出的完整单卡命令，即 `CUDA_VISIBLE_DEVICES=0` 加对应的 Python 训练脚本。

主推荐模型：

```Plaintext
Qwen/Qwen3.5-0.8B
```

选择原因：

1. MedicalGPT 当前脚本已适配 Qwen3\.5 系列。

2. 0\.8B 参数量适合 V100 32GB 上课快速训练。

3. 可以完整跑通 PT、SFT、DPO 和推理，不至于让课程变成单纯等待下载和训练。

备用模型：

```Plaintext
Qwen/Qwen2.5-0.5B
```

## 下载基座模型

本课默认使用 `Qwen/Qwen3.5-0.8B`。如果服务器没有提前准备模型，学生需要自己从 Hugging Face 下载到本地目录：

```Bash
source ~/miniconda3/bin/activate medicalgpt
cd /home/ubuntu
mkdir -p /home/ubuntu/models
```

先确认 Hugging Face CLI 是否可用：

```Bash
which hf
hf --help | head
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

如果 `which hf` 没有输出，说明当前环境还没有安装 CLI，先执行：

```Bash
python -m pip install -U "huggingface_hub[cli]"
which hf
```

下载模型：

```Bash
hf download Qwen/Qwen3.5-0.8B \
  --local-dir /home/ubuntu/models/Qwen3.5-0.8B
```

如果 `hf` 命令不可用，可以使用 Python API：

```Bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir="/home/ubuntu/models/Qwen3.5-0.8B",
)
PY
```

正常课堂顺序应优先使用 `hf download`。Python API 只是备用方案，不要在还没创建 Conda 环境时直接运行。

检查模型文件：

```Bash
ls -lh /home/ubuntu/models/Qwen3.5-0.8B
du -sh /home/ubuntu/models/Qwen3.5-0.8B
```

后续命令中凡是看到：

```Plaintext
Qwen/Qwen3.5-0.8B
```

都可以替换为本地路径：

```Plaintext
/home/ubuntu/models/Qwen3.5-0.8B
```

使用本地路径的好处是：训练脚本不需要每次联网访问 Hugging Face，课堂环境更稳定。

## 为什么要合并 LoRA adapter

本课大量训练命令使用 LoRA。LoRA 的核心思想是：不直接修改原始大模型的全部参数，而是在模型部分线性层旁边训练一组很小的增量参数，也就是 adapter。

训练后通常会得到两部分：

```Plaintext
基座模型：Qwen/Qwen3.5-0.8B 或某个已合并模型目录
LoRA adapter：runs/pt_lora、runs/sft_lora、runs/dpo_lora 等
```

如果不合并，推理时必须同时加载：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/pt_merged \
  --lora_model runs/sft_lora \
  --interactive
```

合并 LoRA 的意思是：把 adapter 中学到的增量权重加回基座模型权重里，生成一个新的完整模型目录。例如：

```Plaintext
runs/pt_lora + Qwen/Qwen3.5-0.8B -> runs/pt_merged
runs/sft_lora + runs/pt_merged -> runs/sft_merged
runs/dpo_lora + runs/sft_merged -> runs/dpo_merged
```

合并后的好处：

1. 推理时只需要加载一个完整模型目录，不需要再指定 `--lora_model`。

2. 下一阶段训练可以直接以上一阶段合并模型为起点，例如 SFT 使用 PT 合并模型，DPO 使用 SFT 合并模型。

3. 模型目录更容易部署到 Gradio、FastAPI、vLLM 或其他推理服务。

4. 可以减少“基座模型和 adapter 不匹配”的错误。

需要注意：合并不是重新训练，只是一次权重转换。合并后的模型体积会接近完整基座模型，而 LoRA adapter 本身通常只有几十 MB。

# 任务安排

### 任务 1：连接服务器与检查 GPU

学生按教师分配的 IP、端口、用户名登录服务器，进入服务器后检查系统与 GPU：

```Bash
hostname
pwd
date
nvidia-smi
```

记录以下信息：

1. GPU 型号是否为 Tesla V100\-SXM2\-32GB。

2. 显存是否空闲。

3. CUDA Driver Version。

4. 当前是否有其他进程占用 GPU。

如果发现显存被占用，先不要直接杀进程，报告教师确认。

### 任务 2：创建 Python 环境

创建独立环境：

```Bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n medicalgpt python=3.10 -y
conda activate medicalgpt
python -V
```

安装 PyTorch。若服务器 CUDA 驱动较新，推荐：

```Bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证 PyTorch 能识别 GPU：

```Bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

V100 正常情况下 `bf16 supported` 应为 `False`。这就是本任务书统一使用 FP16 的原因。

安装 Hugging Face CLI。后续下载模型和扩展数据集都使用 `hf` 命令，不要在还没有 Conda/Python 环境时直接安装或调用 `huggingface_hub`：

```Bash
python -m pip install -U "huggingface_hub[cli]"
which hf
hf --help | head
```

### 任务 3：获取代码并安装依赖

下载项目：

```Bash
cd ~
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
git log -1 --oneline
```

安装项目依赖：

```Bash
pip install -r requirements.txt --upgrade
pip install gradio fastapi uvicorn tensorboard
```

固定 DPO 相关依赖版本，避免新版本 TRL 与当前 PyTorch 组合不兼容：

```Bash
pip install "trl==0.24.0" mergekit llm-blender weave
```

如果需要做项目测试，再额外安装：

```Bash
pip install pytest
```

如果安装 `transformers>=5.6.0` 或 `trl>=0.29.0` 速度慢，先记录报错并等待课程统一处理镜像源。安装完成后检查：

```Bash
python - <<'PY'
import torch, transformers, peft, trl, datasets
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
print("datasets", datasets.__version__)
PY
```

实测成功版本如下：

```Plaintext
torch 2.5.1+cu121
transformers 5.11.0
peft 0.19.1
trl 0.24.0
datasets 5.0.0
accelerate 1.6.0
huggingface_hub 1.16.1
safetensors 0.5.3
mergekit 0.1.4
llm-blender 0.0.2
weave 0.52.42
protobuf 6.33.6
```

DPO 阶段对 `trl`、`torch`、`transformers`、`peft` 版本组合更敏感。本次实测中，`trl 1.5.1` 和 `trl 0.29.0` 会在 `torch 2.5.1` 下触发 `FSDPModule` 导入错误，因此课堂建议固定为：

```Bash
pip install "trl==0.24.0" mergekit llm-blender weave
```

下载或确认本地基座模型。后续主线命令统一使用 `/home/ubuntu/models/Qwen3.5-0.8B`，避免训练和推理时临时联网：

```Bash
mkdir -p /home/ubuntu/models
test -f /home/ubuntu/models/Qwen3.5-0.8B/config.json || \
hf download Qwen/Qwen3.5-0.8B \
  --local-dir /home/ubuntu/models/Qwen3.5-0.8B

ls -lh /home/ubuntu/models/Qwen3.5-0.8B | head
```

可选测试：

```Bash
python -m pytest tests/test_local_json_dataset_loader.py -q
```

### 任务 4：阅读项目结构

运行：

```Bash
find . -maxdepth 2 -type f | sort | sed 's#^\./##' | head -120
```

重点阅读：

```Bash
sed -n '1,220p' README.md
sed -n '1,220p' docs/training_details.md

sed -n '1,220p' docs/datasets.md
sed -n '1,220p' docs/training_params.md
```

思考以下问题：

1. `training/pretraining.py`、`training/supervised_finetuning.py`、`training/dpo_training.py` 分别对应哪个阶段？

2. `scripts/run_pt.sh`、`scripts/run_sft.sh`、`scripts/run_dpo.sh` 为什么不能在 V100 单卡上原样照抄？

3. `data/pretrain`、`data/sft`、`data/reward` 三类数据格式有什么区别？

4. 为什么 LoRA 训练后通常需要合并 adapter？

### 任务 5：检查数据格式

查看 PT 文本：

```Bash
head -5 data/pretrain/fever.txt
```

查看 SFT 数据：

```Bash
head -2 data/sft/medical_sft_1K_format.jsonl
```

查看 DPO 偏好数据：

```Bash
head -2 data/reward/dpo_zh_500.jsonl
```

校验 SFT jsonl：

```Bash
python tools/validate_jsonl.py --file_path data/sft/medical_sft_1K_format.jsonl
```

注意：`tools/validate_jsonl.py` 主要校验 SFT 的 `conversations` 字段，不适合完整校验 DPO 的 chosen/rejected 质量。因此 DPO 需要人工检查每行是否包含：

```Plaintext
conversations
chosen
rejected
```

`chosen/rejected` 是偏好数据中的两个回答版本。对同一个问题：

```Plaintext
conversations：用户问题，也就是 prompt
chosen：标注者认为更好的回答
rejected：标注者认为较差的回答
```

示例：

```JSON
{
  "conversations": [
    {"from": "human", "value": "高血压患者日常需要注意什么？"}
  ],
  "chosen": "建议低盐饮食、规律监测血压、按医嘱服药、适量运动，并定期复诊。",
  "rejected": "高血压不用管，多喝水就能好。"
}
```

这条数据表达的是：

```Plaintext
在这个问题下，chosen 比 rejected 更符合人类偏好。
```

不同阶段对它的使用方式不同：

|阶段|如何使用 chosen/rejected|
|---|---|
|RM|训练一个奖励模型，让 `score(chosen) > score(rejected)`|
|DPO|直接更新语言模型，让模型以后更倾向生成 chosen 风格，远离 rejected 风格|
|PPO/RLHF|先用 chosen/rejected 训练 RM，再用 RM 分数强化学习优化生成模型|

注意：`chosen` 不等于绝对正确答案，它只是这对答案里相对更好的一个。如果 chosen 本身有医学错误，模型也可能学到错误偏好。因此偏好数据必须人工抽查质量。

人工抽查命令：

```Bash
python - <<'PY'
import json
path = "data/reward/dpo_zh_500.jsonl"
with open(path, encoding="utf-8") as f:
    for i, line in zip(range(3), f):
        obj = json.loads(line)
        print("sample", i)
        print("prompt:", obj["conversations"][0]["value"][:120])
        print("chosen:", obj["chosen"][:120])
        print("rejected:", obj["rejected"][:120])
        print()
PY
```

#### 课堂数据规模说明

本项目仓库自带的数据主要是 **教学样例数据**，规模偏小。它足够支撑本节课完整跑通 PT、SFT、DPO、合并和推理流程，但不能支撑训练出真正可靠的医疗大模型。

本次实测统计：

|数据目录/文件|行数|文件大小|用途|
|---|---|---|---|
|`data/pretrain/en_article_tail500.txt`|500|26\.8 KB|PT 普通文本样例|
|`data/pretrain/fever.txt`|996|343\.4 KB|PT 普通文本样例|
|`data/pretrain/tianlongbabu.txt`|2380|833\.8 KB|PT 普通文本样例|
|`data/sft/medical_sft_1K_format.jsonl`|1000|747\.9 KB|医疗 SFT 样例|
|`data/sft/sharegpt_zh_1K_format.jsonl`|1000|3986\.2 KB|通用中文对话 SFT 样例|
|`data/sft/glaive_toolcall_zh_demo.jsonl`|300|603\.5 KB|tool call SFT 样例|
|`data/reward/dpo_zh_500.jsonl`|500|1273\.7 KB|DPO 偏好样例|
|`data/grpo/sample.jsonl`|32|4\.9 KB|GRPO 演示样例|
|`data/grpo_classroom/sample.jsonl`|23|0\.9 KB|本课 GRPO 算术演示|

### 任务 6：运行 PT 增量预训练

创建输出目录：

```Bash
mkdir -p runs cache
```

#### PT 阶段在做什么

PT 是 PreTraining 或 Continue PreTraining。本课不是从零训练一个 0\.8B 模型，而是在已有基座模型 `Qwen3.5-0.8B` 上做小规模继续预训练。它的训练目标是 **Causal Language Modeling**：

```Plaintext
给定前面的 token，预测下一个 token。
```

因此 PT 数据不需要写成“问题\-答案”格式，只要是普通文本即可。脚本会把文本 tokenizer 成 `input_ids`，再把 `labels` 设置成同一份 `input_ids`。训练时模型会学习“当前位置之前的文本如何预测当前位置 token”。这和 SFT 的区别是：

|阶段|输入数据|学到什么|
|---|---|---|
|PT|普通文本|领域词汇、表达方式、文本分布|
|SFT|问答/对话|按指令回答问题|
|DPO|chosen/rejected 偏好|更偏向安全、清晰、有帮助的回答|

本课使用 `--packing True`，脚本会把很多短文本拼接起来，再切成固定长度片段。`--block_size 512` 表示每条训练样本最多 512 个 token。这样做的原因是：

1. GPU 训练需要形状相近的张量。

2. 短文本直接单独训练会浪费 padding。

3. packing 可以提高训练吞吐，但也会让样本边界不再等同于原始文本行。

本课使用 LoRA 训练，不更新全部模型参数。实机中 Qwen3\.5\-0\.8B 全模型约 7\.58 亿参数，而 PT LoRA 只训练约 541 万参数，占比约 0\.714%。这就是 LoRA 能在单张 V100 上完成课堂训练的关键。

运行单卡 FP16 PT。

```Bash
CUDA_VISIBLE_DEVICES=0 python training/pretraining.py \
  --model_name_or_path /home/ubuntu/models/Qwen3.5-0.8B \
  --train_file_dir ./data/pretrain \
  --validation_file_dir ./data/pretrain \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 800 \
  --max_eval_samples 40 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --warmup_steps 10 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 50 \
  --save_strategy steps \
  --save_total_limit 2 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 4 \
  --block_size 512 \
  --packing True \
  --output_dir runs/pt_lora \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

参数解释：

|参数|本课取值|说明|
|---|---|---|
|`per_device_train_batch_size`|2|每次前向/反向计算放入 2 条 packed 样本|
|`gradient_accumulation_steps`|8|累积 8 次梯度再更新一次参数|
|有效 batch size|16|`2 x 8 x 1 GPU = 16`|
|`block_size`|512|每条 PT 样本长度为 512 token|
|`max_steps`|100|课堂固定训练 100 个 optimizer step|
|`logging_steps`|10|每 10 step 输出一次训练 loss|
|`eval_steps`|25|每 25 step 在验证集评估一次|
|`save_steps`|50|第 50、100 step 保存 checkpoint|
|`learning_rate`|2e\-4|LoRA PT 可使用相对较大的学习率|
|`warmup_steps`|10|前 10 step 从小学习率逐渐升高|
|`fp16`|True|V100 推荐 FP16，不使用 BF16|

如果需要把 GPU 监控写入文件，另开一个终端运行：

```Bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop-ms=2000 \
  -f runs/pt_lora_gpu.csv
```

训练期间另开一个终端观察 GPU：

```Bash
watch -n 2 nvidia-smi
```

记录：

1. 显存峰值。

2. 每秒训练速度或每 step 耗时。

3. 第一步 loss 和最后一次 loss。

4. 是否出现 OOM。

如果出现 OOM，按以下顺序调整：

1. `--per_device_train_batch_size 2` 改为 `1`。

2. `--block_size 512` 改为 `384` 或 `256`。

3. 保持 `--gradient_checkpointing True`。

#### 参考运行记录，PT 正式版

实机路径：

```Plaintext
输出目录：runs/pt_lora_classroom
GPU 记录：runs/pt_lora_classroom_gpu.csv
```

实机命令与上方课堂命令一致，仅输出目录改为 `runs/pt_lora_classroom`，并增加 `--overwrite_output_dir True` 方便重复预演。

数据规模：

```Plaintext
原始 PT 文本行数：3876
packing 后 train samples：581
eval samples：40
block_size：512
有效 batch size：16
```

训练耗时与显存：

|指标|实机结果|
|---|---|
|训练窗口|2026\-06\-11 11:06:25 \- 11:30:20|
|Trainer 记录训练时长|1381\.30 秒，约 23 分 01 秒|
|平均速度|13\.81 秒/step|
|train\_steps\_per\_second|0\.072|
|train\_samples\_per\_second|1\.158|
|GPU 显存峰值|5572 MiB / 32768 MiB|
|GPU 活跃期平均显存|约 5550 MiB|
|GPU 活跃期平均利用率|约 35\.71%|
|GPU 最高温度|59℃|
|GPU 最高功耗|246\.72 W|

训练指标：

|step|train loss|eval loss|eval accuracy|learning rate|
|---|---|---|---|---|
|1|4\.1135|\-|\-|0|
|10|4\.0746|\-|\-|8e\-05|
|20|3\.8705|\-|\-|1\.933e\-04|
|25|\-|3\.9031|0\.3519|\-|
|50|3\.5789|3\.7230|0\.3678|1\.267e\-04|
|75|\-|3\.6028|0\.3836|\-|
|100|3\.3107|3\.5591|0\.3899|1\.556e\-05|

最终结果：

```Plaintext
train_loss：3.5917
eval_loss：3.5591
perplexity：35.1304
LoRA adapter 大小：约 21MB
完整输出目录大小：约 165MB
```

结果解读：

1. loss 从 4\.11 降到 3\.31，说明模型在这批 PT 文本上确实学到了可预测模式。

2. eval\_loss 同步下降，说明不是只在训练 batch 上出现短暂波动。

3. perplexity 可以粗略理解为模型对下一个 token 的“不确定程度”，越低越好，但只适用于语言建模评估。

4. PT 指标不能证明医疗问答正确。医疗正确性要在 SFT/DPO 后通过事实核查和人工评估判断。

5. 显存只用了约 5\.6GB，说明这组参数对 V100 32GB 很保守，适合课堂稳定运行。

6. 这里的 `eval_loss` 只是课堂验证指标，用来判断训练流程是否正常，不等价于真实任务上的泛化能力。

### 任务 7：合并 PT LoRA adapter

LoRA adapter 只保存增量权重。若下一阶段要把 PT 结果作为 SFT 基座，建议先合并：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model /home/ubuntu/models/Qwen3.5-0.8B \
  --tokenizer_path /home/ubuntu/models/Qwen3.5-0.8B \
  --lora_model runs/pt_lora \
  --output_dir runs/pt_merged
```

检查合并结果：

```Bash
ls -lh runs/pt_merged | head
```

如果模型已下载到本地，`--base_model` 和 `--tokenizer_path` 都改为本地模型目录。

### 任务 8：运行 SFT 有监督微调

#### SFT 阶段在做什么

SFT 是 Supervised Fine\-Tuning，有监督微调。PT 学的是“继续写文本”，SFT 学的是“看到用户问题后，按照指令生成 assistant 回答”。本课的 SFT 数据主要是 ShareGPT 风格：

```JSON
{
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "模型应该学习的回答"}
  ]
}
```

SFT 仍然使用 Causal LM 训练方式，但它和 PT 的关键差异在于 **label mask**。脚本默认 `train_on_inputs=False`，所以：

1. 用户问题、system prompt、工具说明等输入部分会被标记为 `IGNORE_INDEX=-100`。

2. 只有 assistant 的回答部分参与 loss 计算。

3. 这样模型不是学习“复读用户问题”，而是学习“在给定问题后生成回答”。

可以把它理解成：

```Plaintext
输入给模型看：system + user + assistant
真正计算 loss：assistant 的 token
不计算 loss：system/user 的 token
```

SFT 中的 `model_max_length=512` 表示一条对话样本最多保留 512 token。太长的多轮对话会被截断，所以学生需要观察：长回答是否被截断，截断后是否影响训练质量。

使用 PT 合并后的模型进行医疗问答 SFT：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path runs/pt_merged \
  --train_file_dir ./data/sft \
  --validation_file_dir ./data/sft \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 1000 \
  --max_eval_samples 40 \
  --model_max_length 512 \
  --max_steps 100 \
  --learning_rate 2e-5 \
  --warmup_steps 10 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 50 \
  --save_strategy steps \
  --save_total_limit 2 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 4 \
  --output_dir runs/sft_lora \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache
```

参数解释：

|参数|本课取值|说明|
|---|---|---|
|`model_name_or_path`|`runs/pt_merged`|使用 PT 合并后的模型继续训练|
|`max_train_samples`|1000|从 2300 条 SFT 数据中取 1000 条课堂训练|
|`max_eval_samples`|40|取 40 条做课堂快速评估|
|`model_max_length`|512|每条对话样本最大 token 长度|
|`per_device_train_batch_size`|2|单次训练放入 2 条对话样本|
|`gradient_accumulation_steps`|8|累积 8 次梯度再更新|
|有效 batch size|16|`2 x 8 x 1 GPU = 16`|
|`max_steps`|100|固定 100 个 optimizer step，便于控制课堂时间|
|`learning_rate`|2e\-5|SFT 比 PT 更接近任务行为，学习率更小|
|`train_on_inputs`|False|默认值，只训练 assistant 回答部分|
|`tool_format`|default|让工具调用样本按默认格式展开|

如果需要记录 GPU 监控，另开一个终端运行：

```Bash

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop-ms=2000 \
  -f runs/sft_lora_gpu.csv
```

SFT 训练期间完成以下分析任务，不要空等：

1. 打开 `data/sft/medical_sft_1K_format.jsonl`，抽查 5 条数据，判断回答是否存在医学风险。

2. 思考 `model_max_length=512` 对长问答的影响。

3. 思考 `learning_rate=2e-5` 为什么比 PT 阶段 `2e-4` 更小。

4. 比较 `lora_rank=8` 和 `lora_rank=16` 的参数量、显存和效果可能差异。

#### 参考运行记录，SFT 正式版

实机路径：

```Plaintext
PT 合并模型：runs/pt_lora_classroom_merged
SFT 输出目录：runs/sft_lora_classroom
GPU 记录：runs/sft_lora_classroom_gpu.csv
SFT 推理结果：runs/sft_lora_classroom_predictions.jsonl
SFT 合并模型：runs/sft_lora_classroom_merged
```

实机命令与上方课堂命令一致，仅把：

```Plaintext
--model_name_or_path runs/pt_merged
--output_dir runs/sft_lora
```

改为：

```Plaintext
--model_name_or_path runs/pt_lora_classroom_merged
--output_dir runs/sft_lora_classroom
```

数据规模：

```Plaintext
原始 SFT 数据：2300 条
训练样本：1000 条
评估样本：40 条
model_max_length：512
有效 batch size：16
LoRA 可训练参数：5,411,328
可训练参数占比：0.7141%
```

训练耗时与显存：

|指标|实机结果|
|---|---|
|训练窗口|2026\-06\-11 12:07:31 \- 12:29:48|
|Trainer 记录训练时长|1282\.67 秒，约 21 分 23 秒|
|平均速度|约 12\.83 秒/step|
|train\_steps\_per\_second|0\.078|
|train\_samples\_per\_second|1\.247|
|GPU 显存峰值|11222 MiB / 32768 MiB|
|GPU 活跃期平均显存|约 11119 MiB|
|GPU 活跃期平均利用率|约 28\.97%|
|GPU 最高温度|54℃|
|GPU 最高功耗|171\.22 W|

训练指标：

|step|train loss|eval loss|learning rate|
|---|---|---|---|
|1|2\.2569|\-|0|
|10|2\.5746|\-|6e\-06|
|20|2\.7148|\-|1\.933e\-05|
|25|\-|2\.4831|\-|
|50|2\.6604|2\.4136|1\.267e\-05|
|75|\-|2\.3955|\-|
|100|2\.6182|2\.3877|1\.556e\-06|

最终结果文件：

```Plaintext
train_loss：2.5555
eval_loss：2.4947
perplexity：12.1176
LoRA adapter 大小：约 21MB
完整输出目录大小：约 165MB
```

说明：`trainer_state.json` 中记录的 step 100 过程评估为 `eval_loss=2.3877`，训练结束后单独 `evaluate()` 写入 `eval_results.json` 的最终结果为 `eval_loss=2.4947`。课堂报告以结果文件为准，同时可以引用过程曲线说明变化趋势。这里的 `eval_loss` 仍然只是课堂验证指标，不应当写成“模型已经达到正式效果”。

SFT 后推理验证：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/pt_merged \
  --lora_model runs/sft_lora \
  --output_file runs/sft_predictions.jsonl \
  --max_new_tokens 128 \
  --temperature 0.2
```

预演中额外完成过 SFT adapter 合并，输出目录为：

```Plaintext
runs/sft_lora_classroom_merged
```

课堂中为了保持任务节奏，统一在后面的“任务 10：合并 SFT adapter”执行合并。

使用的合并命令如下：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model runs/pt_lora_classroom_merged \
  --tokenizer_path runs/pt_lora_classroom_merged \
  --lora_model runs/sft_lora_classroom \
  --output_dir runs/sft_lora_classroom_merged
```

### 任务 9：课堂推理测试与训练前后对比

先使用原始基座模型推理：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model /home/ubuntu/models/Qwen3.5-0.8B \
  --interactive \
  --max_new_tokens 256 \
  --temperature 0.2
```

输入以下问题并记录回答：

```Plaintext
乙肝和丙肝有什么区别？
高血压患者日常需要注意什么？
儿童发烧到多少度需要及时就医？
```

输入 `exit` 退出。

再使用 SFT adapter 推理：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/pt_merged \
  --lora_model runs/sft_lora \
  --interactive \
  --max_new_tokens 256 \
  --temperature 0.2
```

对比维度：

1. 是否更像医生问答风格。

2. 是否能给出分点建议。

3. 是否出现过度诊断、虚假药物建议、绝对化表达。

4. 是否提醒就医或咨询专业医生。

医疗模型必须强调安全边界：模型输出不能替代医生诊断。

参考观察：2 step smoke test 的 SFT/DPO 模型曾在“乙肝和丙肝有什么区别？”问题中生成过明显医学事实错误，例如把 HBV/HAV 关系说错，或者把 HBV 误说成 RNA 病毒。课堂报告中必须记录这类错误，并说明：

1. 小样本、少步数训练只能验证流程，不能验证医学可靠性。

2. 医疗场景必须做事实核查和安全评估。

3. 模型回答只能作为教学样例，不能用于真实诊疗建议。

### 任务 10：合并 SFT adapter

为了让 DPO 使用 SFT 后的模型作为起点，合并 SFT adapter：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model runs/pt_merged \
  --tokenizer_path runs/pt_merged \
  --lora_model runs/sft_lora \
  --output_dir runs/sft_merged
```

检查：

```Bash
ls -lh runs/sft_merged | head
```

### 任务 11：运行 DPO 偏好对齐训练

#### DPO 阶段在做什么

DPO 是 Direct Preference Optimization，直接偏好优化。它不再只告诉模型“标准答案是什么”，而是给模型一组偏好对：

```Plaintext
prompt：同一个问题
chosen：更好的回答
rejected：较差的回答
```

DPO 的目标是让模型在同一个 prompt 下，提高 chosen 的相对概率，降低 rejected 的相对概率。直观理解：

```Plaintext
SFT：请模仿这个好答案。
DPO：这两个答案里，chosen 比 rejected 更好，请以后更偏向 chosen 的风格。
```

和 SFT 相比，DPO 更像“排序偏好学习”。它不要求人工给每个回答打具体分数，只需要知道两个回答谁更好。

DPO 数据格式为：

```JSON
{"conversations": [{"from": "human", "value": "问题"}], "chosen": "更好的回答", "rejected": "较差的回答"}
```

DPO 日志中常见指标：

|指标|含义|如何理解|
|---|---|---|
|`loss`|DPO 优化目标|通常越低越好，但不能单独判断医学质量|
|`rewards/chosen`|chosen 的相对奖励|越高表示模型越偏向 chosen|
|`rewards/rejected`|rejected 的相对奖励|越低表示模型越远离 rejected|
|`rewards/margins`|chosen 与 rejected 的奖励差|越大表示偏好区分越强|
|`rewards/accuracies`|chosen 奖励高于 rejected 的比例|接近 1 表示模型几乎总偏 chosen|
|`logps/chosen`|chosen 回答的 log probability|用于计算偏好差异|
|`logps/rejected`|rejected 回答的 log probability|用于计算偏好差异|

注意：`rewards/accuracies` 很高不等于模型医学上正确。它只说明模型学会了当前数据里的偏好方向。如果 chosen 本身质量不高，DPO 也可能强化错误风格。

运行 DPO：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/dpo_training.py \
  --model_name_or_path runs/sft_merged \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 300 \
  --max_eval_samples 40 \
  --max_steps 100 \
  --eval_steps 25 \
  --save_steps 50 \
  --max_source_length 1024 \
  --max_target_length 1024 \
  --output_dir runs/dpo_lora \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache \
  --overwrite_cache True \
  --preprocessing_num_workers 1 \
  --logging_steps 10
```

参数解释：

|参数|本课取值|说明|
|---|---|---|
|`model_name_or_path`|`runs/sft_merged`|使用 SFT 合并后的模型作为 DPO 起点|
|`max_train_samples`|300|从 511 条偏好数据中取 300 条训练|
|`max_eval_samples`|40|取 40 条快速评估|
|`per_device_train_batch_size`|1|DPO 同时处理 chosen/rejected，更占显存|
|`gradient_accumulation_steps`|8|有效 batch size 为 8|
|`max_steps`|100|固定 100 个 optimizer step|
|`max_source_length + max_target_length`|2048|避免样本被字符串长度过滤为空|
|`learning_rate`|默认 5e\-4|DPO 脚本默认值，本课沿用|
|`preprocessing_num_workers`|1|避免课堂环境多进程缓存干扰|

DPO 阶段关注：

1. DPO 不是让模型记住唯一答案，而是让模型偏向 chosen 风格、远离 rejected 风格。

2. chosen/rejected 的质量决定对齐效果。低质量偏好数据可能让模型变差。

3. DPO 比 RM\+PPO 简洁，但仍然可能过拟合。

4. 课堂样本量较小，目标是跑通流程和理解机制，不追求真实生产效果。

5. MedicalGPT 当前 DPO 脚本用字符串长度过滤样本，不是 token 数。`--max_source_length 512 --max_target_length 256` 可能导致样本被过滤过多；本课使用 `1024+1024` 是为了保证课堂数据能进入训练。

6. 这里的 `eval_loss`、`eval_rewards/accuracies` 都是课堂验证指标，适合判断流程是否跑通，不适合直接写成“模型质量已经显著提升”。

#### 参考运行记录，DPO 正式版

实机路径：

```Plaintext
DPO 起点模型：runs/sft_lora_classroom_merged
DPO 输出目录：runs/dpo_lora_classroom
GPU 记录：runs/dpo_lora_classroom_gpu.csv
DPO 推理结果：runs/dpo_lora_classroom_predictions.jsonl
DPO 合并模型：runs/dpo_lora_classroom_merged
```

实机命令与上方课堂命令一致，仅把：

```Plaintext
--model_name_or_path runs/sft_merged
--output_dir runs/dpo_lora
```

改为：

```Plaintext
--model_name_or_path runs/sft_lora_classroom_merged
--output_dir runs/dpo_lora_classroom
```

数据规模：

```Plaintext
原始 DPO 数据：511 条
训练输入样本：300 条
字符串长度过滤后训练样本：290 条
评估输入样本：40 条
字符串长度过滤后评估样本：38 条
有效 batch size：8
LoRA 可训练参数：5,411,328
可训练参数占比：0.7141%
```

训练耗时与显存：

|指标|实机结果|
|---|---|
|训练窗口|2026\-06\-11 15:13:21 \- 15:40:06|
|Trainer 记录训练时长|1548\.18 秒，约 25 分 48 秒|
|平均速度|约 15\.48 秒/step|
|train\_steps\_per\_second|0\.065|
|train\_samples\_per\_second|0\.517|
|GPU 显存峰值|14878 MiB / 32768 MiB|
|GPU 活跃期平均显存|约 14779 MiB|
|GPU 活跃期平均利用率|约 32\.33%|
|GPU 最高温度|57℃|
|GPU 最高功耗|213\.57 W|

训练指标：

|step|loss|reward accuracy|reward margin|eval loss|eval reward accuracy|eval reward margin|
|---|---|---|---|---|---|---|
|10|0\.6931|0\.0000|0\.0000|\-|\-|\-|
|20|0\.6847|0\.4750|0\.0188|\-|\-|\-|
|25|\-|\-|\-|0\.4587|0\.9211|0\.7958|
|50|0\.2595|0\.8875|3\.6030|0\.2028|0\.8947|3\.9267|
|75|\-|\-|\-|0\.0163|1\.0000|7\.4327|
|100|0\.0417|0\.9750|11\.3609|0\.1256|0\.9474|9\.3378|

最终结果文件：

```Plaintext
train_loss：0.3444
eval_loss：0.1256
eval_rewards/accuracies：0.9474
eval_rewards/margins：9.3378
LoRA adapter 大小：约 21MB
完整输出目录大小：约 203MB
```

结果解读：

1. DPO loss 从接近 `0.693` 逐步下降，说明模型开始区分 chosen/rejected。

2. reward margin 从接近 0 升到较高值，说明模型更偏向 chosen。

3. eval reward accuracy 到 `0.9474`，说明在这批评估偏好数据上，模型大多数时候给 chosen 更高奖励。

4. margin 和 accuracy 很高也可能意味着过拟合，尤其本课只有几百条偏好数据。

5. 这些指标说明课堂版 DPO 的训练链路是通的，但不代表模型已经具备可直接使用的医疗问答可靠性。

6. DPO 后推理中，“介绍北京”的重复问题明显减少，“乙肝和丙肝”的回答也比 SFT 阶段更规整，但仍不能替代医学事实评估。

DPO 后推理验证：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/sft_merged \
  --lora_model runs/dpo_lora \
  --output_file runs/dpo_predictions.jsonl \
  --max_new_tokens 128 \
  --temperature 0.2
```

### 任务 12：结果整理与提交

检查三个阶段输出：

```Bash
find runs -maxdepth 2 -type f \( -name "trainer_state.json" -o -name "train_results.json" -o -name "eval_results.json" -o -name "adapter_config.json" \) | sort
```

推理 DPO 后模型：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/sft_merged \
  --lora_model runs/dpo_lora \
  --interactive \
  --max_new_tokens 256 \
  --temperature 0.2
```

如需得到可直接加载的完整模型目录，继续合并 DPO adapter：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model runs/sft_merged \
  --tokenizer_path runs/sft_merged \
  --lora_model runs/dpo_lora \
  --output_dir runs/dpo_merged
```

验证合并后的模型：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/dpo_merged \
  --interactive \
  --max_new_tokens 256 \
  --temperature 0.2
```

# 常见问题与处理

### 1\. CUDA out of memory

处理顺序：

```Plaintext
batch_size 2 -> 1
model_max_length/block_size 512 -> 384 -> 256
max_target_length 256 -> 128
确认 gradient_checkpointing=True
确认没有其他进程占用 GPU
```

清理当前 Python 进程后：

```Bash
nvidia-smi
```

### 2\. V100 使用 bf16 报错或训练异常

原因：V100 不支持原生 BF16。把命令里的：

```Plaintext
--bf16
--torch_dtype bfloat16
```

改成：

```Plaintext
--fp16 True

--torch_dtype float16
```

### 3\. FlashAttention 安装或运行失败

本课不要求安装 FlashAttention。不要使用：

```Plaintext
--flash_attn True
```

### 4\. 合并 LoRA 时显存不足

确保只占用一张 GPU，并关闭训练进程：

```Bash
nvidia-smi
```

仍不足时可重启 Python 进程，释放显存后重新执行合并命令。

### 5\. 合并 LoRA 时报 AutoModelForConditionalGeneration 导入错误

本次环境中，`transformers 5.11.0` 环境运行：

```Bash
cd /home/ubuntu/MedicalGPT
conda activate medicalgpt
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model /home/ubuntu/models/Qwen3.5-0.8B \
  --tokenizer_path /home/ubuntu/models/Qwen3.5-0.8B \
  --lora_model runs/pt_lora \
  --output_dir runs/pt_merged
```

可能报：

```Plaintext
ImportError: cannot import name 'AutoModelForConditionalGeneration' from 'transformers'
```

原因：

MedicalGPT 当前 `tools/merge_peft_adapter.py` 兼容了部分多模态/条件生成模型，脚本中导入了 `AutoModelForConditionalGeneration`。但当前 transformers 版本没有暴露这个 Auto 类。Qwen3\.5\-0\.8B 虽然配置里写的是 `Qwen3_5ForConditionalGeneration`，但可以通过 `AutoModelForCausalLM` 加载并完成 LoRA 合并。

修复方式：修改 `tools/merge_peft_adapter.py`，把导入改成可选导入：

```Python
try:
    from transformers import AutoModelForConditionalGeneration
except ImportError:
    AutoModelForConditionalGeneration = None
```

并把条件加载逻辑改成：

```Python
if is_conditional and AutoModelForConditionalGeneration is not None:
    base_model = AutoModelForConditionalGeneration.from_pretrained(...)
else:
    base_model = AutoModelForCausalLM.from_pretrained(...)
```

修复后重新执行合并命令即可。本次环境中，Qwen3\.5\-0\.8B 使用 `AutoModelForCausalLM` 合并成功。

### 6\. RM 脚本不接受 overwrite\_output\_dir

如果运行 `training/reward_modeling.py` 时报：

```Plaintext
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--overwrite_output_dir']
```

说明当前 RM 脚本不接受该参数。处理方式：不要传 `--overwrite_output_dir`，重跑时换一个新的输出目录，例如：

```Bash
--output_dir runs/rm_lora_hf_classroom_v2
```

### 7\. RM 加载 SequenceClassification 模型时报 load\_in\_4bit 错误

如果运行 RM 时报：

```Plaintext
TypeError: GenericForSequenceClassification.__init__() got an unexpected keyword argument 'load_in_4bit'
```

原因：当前 Transformers 版本中，`AutoModelForSequenceClassification` 对 Qwen3\.5 的通用序列分类类不接受显式传入 `load_in_4bit=False` 或 `load_in_8bit=False`。

修复方式：修改 `training/reward_modeling.py`，只在真正启用量化时才传量化参数：

```Python
model_kwargs = {
    "config": config,
    "torch_dtype": torch_dtype,
    "device_map": model_args.device_map,
    "trust_remote_code": model_args.trust_remote_code,
}
if model_args.load_in_4bit:
    model_kwargs["load_in_4bit"] = True
if model_args.load_in_8bit:
    model_kwargs["load_in_8bit"] = True
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
)
```

### 8\. RM 批量评估时报 no padding token

如果运行 `tools/evaluate_reward_model.py` 时报：

```Plaintext
ValueError: Cannot handle batch sizes > 1 if no padding token is defined.
```

说明序列分类模型配置里没有 `pad_token_id`。处理方式是在加载模型后设置：

```Python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

或者临时把评估命令里的 `--batch_size 4` 改成：

```Bash
--batch_size 1
```

### 9\. DPO 导入 TRL 时报 FSDPModule 错误

本次环境中，`torch 2.5.1+cu121` 搭配较新的 `trl` 可能报：

```Plaintext
ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
```

处理方式：

```Bash
pip install "trl==0.24.0" mergekit llm-blender weave
```

原因：新版 TRL 的部分 Trainer 代码依赖更新的 PyTorch FSDP 接口。V100 课堂环境为了稳定保留 `torch 2.5.1+cu121`，因此降低 TRL 版本比升级整套 PyTorch 更稳。

### 10\. DPO 导入时报 TRANSFORMERS\_CACHE 错误

安装 `trl==0.24.0` 后，如果报：

```Plaintext
ImportError: cannot import name 'TRANSFORMERS_CACHE' from 'transformers.utils.hub'
```

原因：`llm-blender` 依赖仍在引用旧版 Transformers 中的变量，而 `transformers 5.11.0` 已移除该变量。修复方式是在 `training/dpo_training.py` 中，`from trl import DPOTrainer, DPOConfig` 之前加入：

```Python
import transformers.utils.hub as transformers_hub

if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    transformers_hub.TRANSFORMERS_CACHE = os.environ.get(
        "TRANSFORMERS_CACHE",
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/transformers")),
    )
```

注意：文件开头已经有 `import os`，如果没有，需要一并补上。

### 11\. DPO 创建 Trainer 时报 warnings\_issued 错误

如果报：

```Plaintext
AttributeError: 'Qwen3_5ForCausalLM' object has no attribute 'warnings_issued'
```

原因：当前 TRL 期望模型对象带有 `warnings_issued` 字典，用于记录 Trainer 警告状态；Qwen3\.5 模型类没有这个属性。修复方式是在 `training/dpo_training.py` 创建 `DPOConfig` 之前、模型加载之后加入：

```Python
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}
```

### 12\. DPO 过滤后训练样本数为 0

如果日志出现：

```Plaintext
Num train_samples: 0
IndexError: Invalid key: 0 is out of bounds for size 0
```

原因：DPO 脚本过滤逻辑是：

```Python
len(prompt + chosen) <= max_source_length + max_target_length
```

这里计算的是字符串长度，不是 token 数。参数太小会把数据全部过滤掉。处理方式：

```Bash
--max_source_length 1024 \
--max_target_length 1024 \
--overwrite_cache True \
--preprocessing_num_workers 1
```

### 13\. 推理输出乱码或模板不匹配

优先使用 tokenizer 内置 chat template。若换成其他模型，需要指定对应 `--template_name`，例如 `qwen`、`qwen3`、`vicuna` 等。训练和推理模板应保持一致。

## 可选扩展数据集：shibing624/medical

如果希望把课程从“demo 数据”扩展到更接近真实医疗训练数据，可以使用 Hugging Face 上的公开数据集（推荐，实测课堂时间足够完成）：

```Plaintext
https://huggingface.co/datasets/shibing624/medical
```

该数据集卡片说明：

|子集|规模|字段|适合阶段|
|---|---|---|---|
|`pretrain/train_encyclopedia.json`|361420 条|`text`|PT 继续预训练|
|`pretrain/medical_book_zh.json`|8475 条|`text`|PT 继续预训练|
|`finetune/train_zh_0.json`|1949972 条|`instruction/input/output`|SFT|
|`finetune/train_en_1.json`|116617 条|`instruction/input/output`|SFT|
|`reward/train.json`|3800 条|`question/response_chosen/response_rejected`|DPO/RM|

数据集总文件大小约 2\.12GB。

##### Step 0：进入项目并激活环境

后续命令都在 MedicalGPT 项目目录中执行：

```Bash
cd /home/ubuntu/MedicalGPT
conda activate medicalgpt
```

如果 `conda activate medicalgpt` 提示 `conda: command not found`，先执行：

```Bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate medicalgpt
```

确认当前 Python 来自课程环境：

```Bash
which python
python -V
python -c "import torch, transformers, datasets, peft, trl; print('env ok')"
```

确认磁盘空间。数据下载和模型输出都在 `/home/ubuntu` 下，建议至少剩余 10GB：

```Bash
df -h /home/ubuntu
```

创建数据目录：

```Bash
mkdir -p /home/ubuntu/datasets/shibing624_medical
mkdir -p runs cache
```

##### Step 1：下载 Hugging Face 医疗数据集

先检查是否有 `hf` 命令：

```Bash
which hf
hf --help | head
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

如果没有 `hf` 命令，先确认已进入 `medicalgpt` 环境，再安装 CLI：

```Bash
conda activate medicalgpt
python -m pip install -U "huggingface_hub[cli]"
which hf
```

下载方式一，使用 Hugging Face CLI：

```Bash
hf download shibing624/medical \
  --repo-type dataset \
  --local-dir /home/ubuntu/datasets/shibing624_medical \
  --include "pretrain/*.json" "finetune/train_zh_0.json" "reward/*.json"
```

如果 `hf` 命令不存在，但 Python 环境里已经安装了 `huggingface_hub`，可以使用 Python API 下载：

```Bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shibing624/medical",
    repo_type="dataset",
    local_dir="/home/ubuntu/datasets/shibing624_medical",
    allow_patterns=[
        "pretrain/*.json",
        "finetune/train_zh_0.json",
        "reward/*.json",
    ],
)
PY
```

注意：Hugging Face 页面提示该数据集 viewer 依赖自定义加载脚本，因此课堂更推荐使用 `hf download` 或 `snapshot_download` 直接下载文件，而不是让学生在 `datasets.load_dataset(..., trust_remote_code=True)` 中执行远端代码。

下载完成后检查文件：

```Bash
python - <<'PY'
from pathlib import Path

root = Path("/home/ubuntu/datasets/shibing624_medical")
for path in sorted(root.rglob("*.json")):
    n = sum(1 for _ in path.open("r", encoding="utf-8"))
    size_mb = path.stat().st_size / 1024 / 1024
    print(path.relative_to(root), n, f"{size_mb:.2f} MB")
PY
```

重点检查 `reward` 目录是否完整。完整下载后应至少看到：

```Plaintext
reward/train.json    3800 行
reward/valid.json     100 行
reward/test.json      100 行
```

如果只看到 `reward/train.json`，说明偏好数据没有下全。补下载：

```Bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in ["reward/valid.json", "reward/test.json"]:
    path = hf_hub_download(
        repo_id="shibing624/medical",
        repo_type="dataset",
        filename=filename,
        local_dir="/home/ubuntu/datasets/shibing624_medical",
    )
    print(path)
PY
```

下载后不能直接替换 `data/pretrain`、`data/sft`、`data/reward`，因为字段格式不同：

|HF 数据字段|MedicalGPT 当前课堂脚本需要|
|---|---|
|PT: `{"text": ...}`|可直接保存为 `.jsonl`，字段仍为 `text`|
|SFT: `instruction/input/output`|需要转换为 ShareGPT `conversations`|
|Reward: `question/response_chosen/response_rejected`|需要转换为 `conversations/chosen/rejected`|

##### Step 2：转换为 MedicalGPT 课堂格式

本课已提供转换脚本：

```Bash
python tools/prepare_hf_medical_dataset.py \
  --source_dir /home/ubuntu/datasets/shibing624_medical \
  --output_dir data/medical_hf_classroom \
  --pt_samples 5000 \
  --sft_samples 5000 \
  --dpo_samples 1000
```

转换后得到：

```Plaintext
data/medical_hf_classroom/pretrain/train.jsonl
data/medical_hf_classroom/sft/train.jsonl
data/medical_hf_classroom/reward/train.jsonl
data/medical_hf_classroom/reward/train/train.jsonl
data/medical_hf_classroom/reward/valid/valid.jsonl
data/medical_hf_classroom/reward/test/test.jsonl
```

检查转换结果：

```Bash
wc -l data/medical_hf_classroom/pretrain/train.jsonl
wc -l data/medical_hf_classroom/sft/train.jsonl
wc -l data/medical_hf_classroom/reward/train.jsonl
wc -l data/medical_hf_classroom/reward/train/train.jsonl
wc -l data/medical_hf_classroom/reward/valid/valid.jsonl
wc -l data/medical_hf_classroom/reward/test/test.jsonl

head -1 data/medical_hf_classroom/pretrain/train.jsonl
head -1 data/medical_hf_classroom/sft/train.jsonl
head -1 data/medical_hf_classroom/reward/train.jsonl
```

校验 SFT 转换格式：

```Bash
python tools/validate_jsonl.py --file_path data/medical_hf_classroom/sft/train.jsonl
```

校验 DPO 转换格式：

```Bash
python - <<'PY'
import json

path = "data/medical_hf_classroom/reward/train.jsonl"
with open(path, encoding="utf-8") as f:
    for i, line in zip(range(3), f):
        obj = json.loads(line)
        assert "conversations" in obj
        assert "chosen" in obj
        assert "rejected" in obj
        print("sample", i, "ok")
print("DPO converted format ok")
PY
```

##### Step 3：获取转换脚本并生成课堂训练数据

注意区分两个目录：

```Plaintext
/home/ubuntu/datasets/shibing624_medical
```

这是从 `shibing624/medical` 下载的原始数据，字段仍是原始格式。

```Plaintext
/home/ubuntu/MedicalGPT/data/medical_hf_classroom
```

这是运行 `tools/prepare_hf_medical_dataset.py` 之后生成的课堂数据，已经适配 MedicalGPT 的 PT/SFT/DPO/RM 训练脚本。

本步骤需要自己完成转换，而不是直接使用别人已经转换好的 `data/medical_hf_classroom` 目录。原因有三个：

- 能理解原始数据格式和训练脚本输入格式之间的差异。

- 能掌握大模型训练前最重要的数据预处理步骤。

- 如果后续更换数据集，可以复用同一套转换思路。

转换脚本位置：

```Plaintext
/home/ubuntu/MedicalGPT/tools/prepare_hf_medical_dataset.py
```

脚本自身只依赖 Python 标准库：

```Plaintext
argparse
json
random
pathlib
```

因此，转换脚本通常不需要额外安装 Python 包。若转换失败，优先检查原始数据文件是否存在、路径是否写错、磁盘空间是否足够。

确认转换脚本存在：

```Bash
cd /home/ubuntu/MedicalGPT
ls -lh tools/prepare_hf_medical_dataset.py
sed -n '1,80p' tools/prepare_hf_medical_dataset.py
```

如果当前项目中没有这个脚本，需要把 `prepare_hf_medical_dataset.py` 放到 MedicalGPT 项目的 `tools` 目录下：

```Bash
cd /home/ubuntu/MedicalGPT
mkdir -p tools
ls -lh tools/prepare_hf_medical_dataset.py
```

转换脚本会读取下面这些原始文件：

```Plaintext
/home/ubuntu/datasets/shibing624_medical/pretrain/train_encyclopedia.json
/home/ubuntu/datasets/shibing624_medical/pretrain/medical_book_zh.json
/home/ubuntu/datasets/shibing624_medical/finetune/train_zh_0.json
/home/ubuntu/datasets/shibing624_medical/finetune/train_en_1.json
/home/ubuntu/datasets/shibing624_medical/reward/train.json
/home/ubuntu/datasets/shibing624_medical/reward/valid.json
/home/ubuntu/datasets/shibing624_medical/reward/test.json
```

各文件用途如下：

```Plaintext
pretrain/train_encyclopedia.json      -> PT 继续预训练文本
pretrain/medical_book_zh.json         -> PT 继续预训练文本
finetune/train_zh_0.json              -> SFT 中文问答数据
finetune/train_en_1.json              -> SFT 英文问答数据；如果不存在，脚本会自动跳过
reward/train.json                     -> DPO/RM 训练集
reward/valid.json                     -> DPO/RM 验证集
reward/test.json                      -> DPO/RM 测试集
```

检查原始文件是否已经下载完整：

```Bash
ls -lh /home/ubuntu/datasets/shibing624_medical/pretrain/
ls -lh /home/ubuntu/datasets/shibing624_medical/finetune/
ls -lh /home/ubuntu/datasets/shibing624_medical/reward/

wc -l /home/ubuntu/datasets/shibing624_medical/pretrain/train_encyclopedia.json
wc -l /home/ubuntu/datasets/shibing624_medical/pretrain/medical_book_zh.json
wc -l /home/ubuntu/datasets/shibing624_medical/finetune/train_zh_0.json
wc -l /home/ubuntu/datasets/shibing624_medical/reward/train.json
wc -l /home/ubuntu/datasets/shibing624_medical/reward/valid.json
wc -l /home/ubuntu/datasets/shibing624_medical/reward/test.json
```

如果 `finetune/train_en_1.json` 存在，也可以检查行数：

```Bash
test -f /home/ubuntu/datasets/shibing624_medical/finetune/train_en_1.json && \
wc -l /home/ubuntu/datasets/shibing624_medical/finetune/train_en_1.json
```

如果需要从 Hugging Face 下载转换脚本，可以把转换脚本放到一个单独的数据集仓库中。这里上传的是转换脚本和说明文件，不上传 `data/medical_hf_classroom` 转换结果。

安装或升级 Hugging Face Hub 工具：

```Bash
python -m pip install -U "huggingface_hub[cli]"
which hf
hf --help | head
```

登录 Hugging Face。执行后会提示粘贴 token，token 需要在 Hugging Face 网页端创建，权限至少包含 `write`：

```Bash
hf auth login
```

检查登录身份：

```Bash
hf auth whoami
```

设置自己的用户名和数据集仓库名。把 `your_hf_name` 改成自己的 Hugging Face 用户名：

```Bash
HF_USERNAME=your_hf_name
HF_SCRIPT_REPO=medicalgpt-classroom-tools
```

准备一个只包含转换脚本和说明文件的发布目录：

```Bash
cd /home/ubuntu/MedicalGPT
mkdir -p /home/ubuntu/hf_upload/medicalgpt-classroom-tools

cp tools/prepare_hf_medical_dataset.py \
  /home/ubuntu/hf_upload/medicalgpt-classroom-tools/
```

写入说明文件：

```Bash
cat > /home/ubuntu/hf_upload/medicalgpt-classroom-tools/README.md <<'EOF'
# MedicalGPT Classroom Tools

This repository contains the dataset conversion script used in the MedicalGPT classroom experiment.

The script converts the raw shibing624/medical dataset into MedicalGPT-compatible formats:

- PT: text-only pretraining JSONL
- SFT: ShareGPT-style supervised fine-tuning JSONL
- Reward/DPO/RM: chosen/rejected pairwise preference JSONL

Download the original dataset from Hugging Face and run this script locally.
Converted data is intentionally not uploaded here. Run the script locally to practice the full data preparation process.

This material is for teaching and research experiments only. Model outputs must not be used as medical advice.
EOF
```

创建 Hugging Face dataset 仓库：

```Bash
hf repos create ${HF_USERNAME}/${HF_SCRIPT_REPO} \
  --repo-type dataset \
  --private \
  --exist-ok
```

如果希望公开访问，把 `--private` 去掉：

```Bash
hf repos create ${HF_USERNAME}/${HF_SCRIPT_REPO} \
  --repo-type dataset \
  --exist-ok
```

上传转换脚本和说明文件：

```Bash
hf upload ${HF_USERNAME}/${HF_SCRIPT_REPO} \
  /home/ubuntu/hf_upload/medicalgpt-classroom-tools \
  . \
  --repo-type dataset
```

查看上传地址：

```Bash
echo "https://huggingface.co/datasets/${HF_USERNAME}/${HF_SCRIPT_REPO}"
```

下载转换脚本：

```Bash
cd /home/ubuntu/MedicalGPT
mkdir -p tools

hf download ${HF_USERNAME}/${HF_SCRIPT_REPO} \
  prepare_hf_medical_dataset.py \
  --repo-type dataset \
  --local-dir tools

ls -lh tools/prepare_hf_medical_dataset.py
```

如果不能使用 `hf download`，也可以从 Hugging Face 网页下载 `prepare_hf_medical_dataset.py`，然后上传到服务器的 `/home/ubuntu/MedicalGPT/tools/` 目录。

下载原始数据后，在本地执行转换：

```Bash
cd /home/ubuntu/MedicalGPT
python tools/prepare_hf_medical_dataset.py \
  --source_dir /home/ubuntu/datasets/shibing624_medical \
  --output_dir data/medical_hf_classroom \
  --pt_samples 5000 \
  --sft_samples 5000 \
  --dpo_samples 1000
```

转换完成后检查输出文件：

```Bash
find data/medical_hf_classroom -maxdepth 4 -type f -printf '%P\t%s bytes\n' | sort

wc -l data/medical_hf_classroom/pretrain/train.jsonl
wc -l data/medical_hf_classroom/sft/train.jsonl
wc -l data/medical_hf_classroom/reward/train.jsonl
wc -l data/medical_hf_classroom/reward/train/train.jsonl
wc -l data/medical_hf_classroom/reward/valid/valid.jsonl
wc -l data/medical_hf_classroom/reward/test/test.jsonl
```

如果要把原始下载目录上传到自己的 Hugging Face，方便后续重复使用，可以另建一个原始数据仓库。注意这里上传的是原始目录 `/home/ubuntu/datasets/shibing624_medical`，不是转换后的 `data/medical_hf_classroom`：

```Bash
HF_RAW_DATASET_REPO=medical-raw-shibing624

hf repos create ${HF_USERNAME}/${HF_RAW_DATASET_REPO} \
  --repo-type dataset \
  --private \
  --exist-ok

hf upload ${HF_USERNAME}/${HF_RAW_DATASET_REPO} \
  /home/ubuntu/datasets/shibing624_medical \
  . \
  --repo-type dataset
```

从自己的原始数据仓库下载后，仍然需要执行转换脚本：

```Bash
mkdir -p /home/ubuntu/datasets/shibing624_medical

hf download ${HF_USERNAME}/${HF_RAW_DATASET_REPO} \
  --repo-type dataset \
  --local-dir /home/ubuntu/datasets/shibing624_medical

cd /home/ubuntu/MedicalGPT
python tools/prepare_hf_medical_dataset.py \
  --source_dir /home/ubuntu/datasets/shibing624_medical \
  --output_dir data/medical_hf_classroom \
  --pt_samples 5000 \
  --sft_samples 5000 \
  --dpo_samples 1000
```

参考运行记录：

```Plaintext
下载目录：/home/ubuntu/datasets/shibing624_medical
下载文件：pretrain/*.json、finetune/train_zh_0.json、reward/train.json、reward/valid.json、reward/test.json
下载耗时：约 1 分 32 秒，实际速度取决于 Hugging Face 网络
下载后文件大小：
- finetune/train_zh_0.json：1949972 行，约 1276.36 MB
- pretrain/train_encyclopedia.json：361420 行，约 563.65 MB
- pretrain/medical_book_zh.json：8475 行，约 38.30 MB
- reward/train.json：3800 行，约 2.95 MB
- reward/valid.json：100 行，约 0.08 MB
- reward/test.json：100 行，约 0.10 MB
转换抽样：
- PT：5000 条，输出 8.3 MB
- SFT：5000 条，输出 3.4 MB
- DPO：1000 条，输出 806 KB
- Reward train：1000 条
- Reward valid：100 条
- Reward test：100 条
转换耗时：约 28 秒
SFT 校验：5000 行全部有效
DPO 抽查：字段 conversations/chosen/rejected 正常
```

学有余力的同学，如果课堂时间允许，可把 PT/SFT/DPO 命令中的数据目录替换为：

```Plaintext
PT:  --train_file_dir data/medical_hf_classroom/pretrain
SFT: --train_file_dir data/medical_hf_classroom/sft
DPO: --train_file_dir data/medical_hf_classroom/reward/train
DPO: --validation_file_dir data/medical_hf_classroom/reward/valid
RM:  --train_file_dir data/medical_hf_classroom/reward/train
RM:  --validation_file_dir data/medical_hf_classroom/reward/valid
```

### 扩展实践 A：使用 Hugging Face 医疗数据运行 PT

完成数据下载和转换后，先跑 PT 扩展版。该任务让学生看到更真实的医疗文本如何进入继续预训练阶段。

##### Step A\-1：确认数据和输出目录

```Bash
cd /home/ubuntu/MedicalGPT
conda activate medicalgpt

wc -l data/medical_hf_classroom/pretrain/train.jsonl
head -1 data/medical_hf_classroom/pretrain/train.jsonl
mkdir -p runs cache
```

##### Step A\-2：启动 GPU 监控

另开一个终端，执行下面命令。该命令会一直运行，用于把显存和利用率写入 CSV 文件：

```Bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/pt_lora_hf_classroom_gpu.csv
```

如果只有一个终端，也可以使用后台方式启动监控：

```Bash
nohup nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/pt_lora_hf_classroom_gpu.csv \
  > runs/pt_lora_hf_classroom_gpu_monitor.log 2>&1 &
```

##### Step A\-3：运行 PT

回到训练终端，执行：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/pretraining.py \
  --model_name_or_path /home/ubuntu/models/Qwen3.5-0.8B \
  --train_file_dir data/medical_hf_classroom/pretrain \
  --validation_file_dir data/medical_hf_classroom/pretrain \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 5000 \
  --max_eval_samples 100 \
  --max_steps 50 \
  --learning_rate 2e-4 \
  --warmup_steps 5 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 25 \
  --save_strategy steps \
  --save_total_limit 2 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 4 \
  --block_size 512 \
  --packing True \
  --output_dir runs/pt_lora_hf_classroom \
  --overwrite_output_dir True \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

训练过程中可另开终端查看：

```Bash
tail -f runs/pt_lora_hf_classroom_gpu.csv
```

##### Step A\-4：训练结束后检查 PT 输出

如果使用后台方式启动了 GPU 监控，训练结束后停止监控：

```Bash

pkill -f "nvidia-smi.*pt_lora_hf_classroom_gpu.csv" || true
```

检查训练输出：

```Bash
ls -lh runs/pt_lora_hf_classroom
cat runs/pt_lora_hf_classroom/train_results.json
cat runs/pt_lora_hf_classroom/eval_results.json
tail -5 runs/pt_lora_hf_classroom_gpu.csv
```

参考运行记录：

```Plaintext
输入数据：data/medical_hf_classroom/pretrain/train.jsonl，5000 条医疗文本，8.3 MB
packing 后训练样本：3437 个 512-token block
验证样本：100 个 512-token block
训练步数：50 steps
训练时长：721.86 秒，约 12 分 01 秒
平均速度：0.069 steps/s，约 14.44 秒/step
train_loss：3.4071
eval_loss：3.4095
eval_accuracy：0.3658
perplexity：30.2511
GPU 峰值显存：5548 MiB / 32768 MiB
GPU 活跃平均显存：5525 MiB
GPU 活跃平均利用率：35.93%
最高温度：58 C
最高功耗：238.81 W
输出目录：runs/pt_lora_hf_classroom
GPU 记录：runs/pt_lora_hf_classroom_gpu.csv
```

合并 PT LoRA：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model /home/ubuntu/models/Qwen3.5-0.8B \
  --tokenizer_path /home/ubuntu/models/Qwen3.5-0.8B \
  --lora_model runs/pt_lora_hf_classroom \
  --output_dir runs/pt_lora_hf_classroom_merged
```

检查合并结果：

```Bash
ls -lh runs/pt_lora_hf_classroom_merged
```

合并后输出：

```Plaintext
runs/pt_lora_hf_classroom_merged
```

### 扩展实践 B：基于 HF\-PT 模型运行 HF 医疗数据 SFT

完成 HF 医疗数据版 PT 并合并后，继续运行 SFT。此时起点模型不是原始 Qwen3\.5\-0\.8B，而是：

```Plaintext
runs/pt_lora_hf_classroom_merged
```

SFT 数据来自转换后的：

```Plaintext
data/medical_hf_classroom/sft/train.jsonl
```

##### Step B\-1：确认 SFT 数据和 PT 合并模型

```Bash
cd /home/ubuntu/MedicalGPT
conda activate medicalgpt

ls -lh runs/pt_lora_hf_classroom_merged/model.safetensors
wc -l data/medical_hf_classroom/sft/train.jsonl
head -1 data/medical_hf_classroom/sft/train.jsonl
```

##### Step B\-2：启动 GPU 监控

另开一个终端，执行：

```Bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/sft_lora_hf_classroom_gpu.csv
```

如果只有一个终端，也可以后台启动：

```Bash
nohup nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/sft_lora_hf_classroom_gpu.csv \
  > runs/sft_lora_hf_classroom_gpu_monitor.log 2>&1 &
```

##### Step B\-3：运行 SFT

回到训练终端，执行：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
  --model_name_or_path runs/pt_lora_hf_classroom_merged \
  --train_file_dir data/medical_hf_classroom/sft \
  --validation_file_dir data/medical_hf_classroom/sft \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --do_train --do_eval \
  --use_peft True \
  --max_train_samples 5000 \
  --max_eval_samples 100 \
  --model_max_length 512 \
  --max_steps 50 \
  --learning_rate 2e-5 \
  --warmup_steps 5 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 25 \
  --save_strategy steps \
  --save_total_limit 2 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 4 \
  --output_dir runs/sft_lora_hf_classroom \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache
```

训练过程中查看 GPU 记录：

```Bash
tail -f runs/sft_lora_hf_classroom_gpu.csv
```

##### Step B\-4：训练结束后检查 SFT 输出

如果使用后台方式启动了 GPU 监控，训练结束后停止监控：

```Bash
pkill -f "nvidia-smi.*sft_lora_hf_classroom_gpu.csv" || true
```

检查训练输出：

```Bash
ls -lh runs/sft_lora_hf_classroom
cat runs/sft_lora_hf_classroom/train_results.json
cat runs/sft_lora_hf_classroom/eval_results.json
tail -5 runs/sft_lora_hf_classroom_gpu.csv
```

注意：当前 `training/supervised_finetuning.py` 不接受 `--overwrite_output_dir` 参数。如果加入该参数，会报：

```Plaintext
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--overwrite_output_dir']
```

解决方式：不要给 SFT 脚本传 `--overwrite_output_dir`。如需重跑，建议换一个新的 `--output_dir`，或者确认旧目录不需要后再手动清理。

参考运行记录：

```Plaintext
起点模型：runs/pt_lora_hf_classroom_merged
输入数据：data/medical_hf_classroom/sft/train.jsonl，5000 条医疗问答，3.4 MB
训练样本：5000 条
验证样本：100 条
训练步数：50 steps
训练时长：608.11 秒，约 10 分 08 秒
平均速度：0.082 steps/s，约 12.16 秒/step
train_loss：3.5177
eval_loss：4.0188
perplexity：55.6332
GPU 峰值显存：5546 MiB / 32768 MiB
GPU 活跃平均显存：5527 MiB
GPU 活跃平均利用率：22.69%
最高温度：49 C
最高功耗：199.02 W
输出目录：runs/sft_lora_hf_classroom
GPU 记录：runs/sft_lora_hf_classroom_gpu.csv
```

合并 SFT LoRA：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model runs/pt_lora_hf_classroom_merged \
  --tokenizer_path runs/pt_lora_hf_classroom_merged \
  --lora_model runs/sft_lora_hf_classroom \
  --output_dir runs/sft_lora_hf_classroom_merged
```

检查合并结果：

```Bash
ls -lh runs/sft_lora_hf_classroom_merged
```

合并后输出：

```Plaintext
runs/sft_lora_hf_classroom_merged
```

准备推理评测文件：

```Bash
cat > data/hf_sft_eval.txt <<'EOF'
介绍下北京
乙肝和丙肝有什么区别？
阴道炎有什么症状？
高血压患者日常需要注意什么？
EOF
```

运行推理：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --data_file data/hf_sft_eval.txt \
  --output_file runs/sft_lora_hf_classroom_predictions.jsonl \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --eval_batch_size 1
```

实机推理观察：

```Plaintext
输出文件：runs/sft_lora_hf_classroom_predictions.jsonl
```

模型已经能生成医疗问答风格的回答，例如“高血压患者日常需要注意什么？”能输出较长的条目式回答。但也出现了明显问题：

1. “介绍下北京”出现长段重复。

2. “高血压患者日常需要注意什么？”反复重复“多吃新鲜蔬菜/水果”。

3. “乙肝和丙肝有什么区别？”回答开头正确，但没有充分展开。

因此，本阶段结论应写成：

```Plaintext
HF 医疗数据版 SFT 能跑通，并让模型更明显地进入医疗问答风格；
但 5000 条抽样、50 step 仍不足以得到稳定、高质量、医学可靠的模型；
推理质量仍需通过更长训练、更严格数据清洗、DPO/GRPO/RAG 和人工医学评测继续改进。
```

### 扩展实践 C：使用 Reward 数据训练奖励模型 RM

完成 HF 医疗数据版 SFT 并合并后，可以继续训练 Reward Model，简称 RM。RM 是传统 RLHF 路线中的关键组件：

```Plaintext
SFT 模型生成回答 -> 人类或规则标注 chosen/rejected -> 训练 RM 给回答打分 -> PPO/RLOO 等算法用 RM 分数继续优化策略模型
```

本节只训练奖励模型。

RM 和 DPO 的区别：

|方法|输入数据|输出结果|课堂理解|
|---|---|---|---|
|RM|prompt \+ chosen/rejected|一个会给回答打分的模型|“训练一个裁判”|
|DPO|prompt \+ chosen/rejected|一个被偏好数据直接更新后的语言模型|“直接把模型往 chosen 方向推”|

##### Step C\-1：确认 reward 数据完整

先确认原始 HF reward 数据是否下全：

```Bash
cd /home/ubuntu/MedicalGPT
conda activate medicalgpt

find /home/ubuntu/datasets/shibing624_medical/reward -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
wc -l /home/ubuntu/datasets/shibing624_medical/reward/*.json
```

应看到：

```Plaintext
train.json    3800 行
valid.json     100 行
test.json      100 行
```

再确认转换后的 MedicalGPT 格式：

```Bash
wc -l data/medical_hf_classroom/reward/train/train.jsonl
wc -l data/medical_hf_classroom/reward/valid/valid.jsonl
wc -l data/medical_hf_classroom/reward/test/test.jsonl

head -1 data/medical_hf_classroom/reward/train/train.jsonl
head -1 data/medical_hf_classroom/reward/valid/valid.jsonl
head -1 data/medical_hf_classroom/reward/test/test.jsonl
```

每行应包含：

```Plaintext
conversations
chosen
rejected
```

##### Step C\-2：理解 RM 数据格式

RM 数据和 DPO 数据格式相同：

```JSON
{"conversations":[{"from":"human","value":"问题"}],"chosen":"更好的回答","rejected":"较差的回答"}
```

训练时，RM 会分别给 `prompt + chosen` 和 `prompt + rejected` 打一个标量分数，然后优化：

```Plaintext
score(chosen) > score(rejected)
```

它不是在学习生成答案，而是在学习“哪个答案更好”。因此 RM 输出目录不能直接当聊天模型使用。

##### Step C\-3：RM smoke test

先跑 2 step 小测试，确认脚本、模型和数据格式都能工作：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/reward_modeling.py \
  --model_name_or_path runs/sft_lora_hf_classroom_merged \
  --tokenizer_name_or_path runs/sft_lora_hf_classroom_merged \
  --train_file_dir data/medical_hf_classroom/reward/train \
  --validation_file_dir data/medical_hf_classroom/reward/valid \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 16 \
  --max_eval_samples 8 \
  --max_steps 2 \
  --learning_rate 2e-5 \
  --warmup_steps 1 \
  --weight_decay 0.001 \
  --logging_strategy steps \
  --logging_steps 1 \
  --eval_steps 1 \
  --eval_strategy steps \
  --save_steps 2 \
  --save_strategy steps \
  --save_total_limit 1 \
  --max_source_length 512 \
  --max_target_length 512 \
  --output_dir runs/rm_lora_hf_smoke \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --preprocessing_num_workers 1 \
  --cache_dir ./cache
```

注意：当前 `training/reward_modeling.py` 不接受 `--overwrite_output_dir` 参数。重跑时建议换一个新的 `--output_dir`，例如 `runs/rm_lora_hf_smoke2`。

如果加载日志中出现：

```Plaintext
score.weight | MISSING
```

这是正常现象。语言模型主体来自 SFT 合并模型，但 RM 的最后打分头 `score.weight` 是新初始化的，需要通过 reward 数据训练出来。

##### Step C\-4：启动 GPU 监控

另开一个终端：

```Bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/rm_lora_hf_classroom_gpu.csv
```

如果只有一个终端，也可以后台启动：

```Bash
nohup nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 \
  -f runs/rm_lora_hf_classroom_gpu.csv \
  > runs/rm_lora_hf_classroom_gpu_monitor.log 2>&1 &
```

##### Step C\-5：运行 RM 正式版

```Bash
CUDA_VISIBLE_DEVICES=0 python training/reward_modeling.py \
  --model_name_or_path runs/sft_lora_hf_classroom_merged \
  --tokenizer_name_or_path runs/sft_lora_hf_classroom_merged \
  --train_file_dir data/medical_hf_classroom/reward/train \
  --validation_file_dir data/medical_hf_classroom/reward/valid \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --do_train --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 1000 \
  --max_eval_samples 100 \
  --max_steps 50 \
  --learning_rate 2e-5 \
  --warmup_steps 5 \
  --weight_decay 0.001 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 25 \
  --eval_strategy steps \
  --save_steps 25 \
  --save_strategy steps \
  --save_total_limit 2 \

  --max_source_length 512 \
  --max_target_length 512 \
  --output_dir runs/rm_lora_hf_classroom \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --preprocessing_num_workers 1 \
  --cache_dir ./cache
```

训练结束后停止后台 GPU 监控：

```Bash
pkill -f "nvidia-smi.*rm_lora_hf_classroom_gpu.csv" || true
```

检查输出：

```Bash
ls -lh runs/rm_lora_hf_classroom
cat runs/rm_lora_hf_classroom/train_results.json
cat runs/rm_lora_hf_classroom/eval_results.json
tail -5 runs/rm_lora_hf_classroom_gpu.csv
```

##### Step C\-6：在 test split 上评估 RM

使用训练好的 RM LoRA，在未参与训练和验证的 `test` 上评估：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/reward_modeling.py \
  --model_name_or_path runs/sft_lora_hf_classroom_merged \
  --tokenizer_name_or_path runs/sft_lora_hf_classroom_merged \
  --peft_path runs/rm_lora_hf_classroom \
  --train_file_dir data/medical_hf_classroom/reward/train \
  --validation_file_dir data/medical_hf_classroom/reward/test \
  --per_device_eval_batch_size 1 \
  --do_eval \
  --use_peft True \
  --max_eval_samples 100 \
  --max_source_length 512 \
  --max_target_length 512 \
  --output_dir runs/rm_lora_hf_classroom_test_eval \
  --ddp_timeout 30000 \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --preprocessing_num_workers 1 \
  --cache_dir ./cache
```

检查 test 结果：

```Bash
cat runs/rm_lora_hf_classroom_test_eval/eval_results.json
```

##### Step C\-7：计算 chosen/rejected 排序准确率

Trainer 的 `eval_loss` 能反映 pairwise loss，但不够直观。本课额外提供一个小工具，直接计算：

```Plaintext
score(chosen) > score(rejected)
```

先检查脚本：

```Bash
python -m py_compile tools/evaluate_reward_model.py
```

快速评估 valid 前 20 条：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_reward_model.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --reward_adapter runs/rm_lora_hf_classroom \
  --data_file data/medical_hf_classroom/reward/valid/valid.jsonl \
  --max_samples 20 \
  --max_length 1024 \
  --batch_size 4 \
  --torch_dtype float16
```

快速评估 test 前 20 条：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_reward_model.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --reward_adapter runs/rm_lora_hf_classroom \
  --data_file data/medical_hf_classroom/reward/test/test.jsonl \
  --max_samples 20 \
  --max_length 1024 \
  --batch_size 4 \
  --torch_dtype float16
```

输出中的核心字段：

|字段|含义|
|---|---|
|`accuracy`|chosen 分数高于 rejected 的比例|
|`avg_margin`|chosen\_score \- rejected\_score 的平均差值|
|`examples`|前 3 条样例的分数对比|

##### Step C\-8：参考运行记录，RM 正式版

```Plaintext
原始 HF reward 文件：
- reward/train.json：3800 行，约 2.95 MB
- reward/valid.json：100 行，约 0.08 MB
- reward/test.json：100 行，约 0.10 MB

转换后数据：
- data/medical_hf_classroom/reward/train/train.jsonl：1000 行
- data/medical_hf_classroom/reward/valid/valid.jsonl：100 行
- data/medical_hf_classroom/reward/test/test.jsonl：100 行

起点模型：runs/sft_lora_hf_classroom_merged
RM 输出目录：runs/rm_lora_hf_classroom
GPU 记录：runs/rm_lora_hf_classroom_gpu.csv
```

训练规模：

```Plaintext
输入训练样本：1000 条
长度过滤后训练样本：997 条
验证样本：100 条
训练步数：50 steps
有效 batch size：8
LoRA 可训练参数：5,412,352
可训练参数占比：0.7142%
```

训练耗时与显存：

|指标|实测结果|
|---|---|
|训练窗口|2026\-06\-11 20:33:28 \- 21:09:51|
|Trainer 记录训练时长|2067\.30 秒，约 34 分 27 秒|
|外部计时|2191 秒，约 36 分 31 秒|
|平均速度|约 41\.35 秒/step，含验证|
|train\_steps\_per\_second|0\.024|
|train\_samples\_per\_second|0\.193|
|GPU 显存峰值|14854 MiB / 32768 MiB|
|GPU 活跃期平均显存|约 9385 MiB|
|GPU 活跃期平均利用率|约 30\.80%|
|GPU 最高温度|48 C|
|GPU 最高功耗|96\.56 W|

训练与验证指标：

|指标|结果|
|---|---|
|train\_loss|0\.3337|
|valid eval\_loss|0\.0333|
|valid eval\_mse|65\.0691|
|valid eval\_mae|7\.6247|
|valid perplexity|1\.0338|
|test eval\_loss|0\.9814|
|test eval\_mse|7\.7040|
|test eval\_mae|1\.9058|
|test perplexity|2\.6683|

快速排序评估：

|数据|样本数|accuracy|avg\_margin|
|---|---|---|---|
|valid 前 20 条|20|0\.95|7\.3280|
|test 前 20 条|20|1\.00|7\.9330|

结果解读：

1. RM 训练能完整跑通，模型学会给 chosen/rejected 打相对分数。

2. `score.weight | MISSING` 是正常现象，因为序列分类打分头需要新训练。

3. 第一个 step 可能出现 `grad_norm=nan`，后续恢复正常且 loss 持续记录时，可以继续观察。

4. valid loss 很低、test loss 明显更高，说明小样本 RM 存在过拟合风险。

5. RM 本身不是聊天模型，不能直接替代 SFT/DPO 后的生成模型。

### 扩展实践 D：推理可视化，对比 HF\-SFT 与 HF\-DPO

本节目标是把“新数据集 SFT 后模型”和“偏好对齐后模型”的回答并排展示，方便观察回答质量、重复、串题和安全风险。

先明确一个概念：`runs/rm_lora_hf_classroom` 是 Reward Model，只能给回答打分，不能直接生成回答。因此推理可视化中的“强化学习后模型”使用 DPO 后的生成模型：

```Plaintext
HF 医疗数据 PT -> HF 医疗数据 SFT -> HF reward 数据 DPO -> 推理对比
```

对应路径：

```Plaintext
SFT 后模型：runs/sft_lora_hf_classroom_merged
DPO adapter：runs/dpo_lora_hf_classroom
DPO 合并模型：runs/dpo_lora_hf_classroom_merged
```

##### Step D\-1：基于 HF\-SFT 起点运行 HF\-DPO

如果已经完成前面的 HF\-SFT，并且有 reward train/valid 数据，可以运行：

```Bash
CUDA_VISIBLE_DEVICES=0 python training/dpo_training.py \
  --model_name_or_path runs/sft_lora_hf_classroom_merged \
  --train_file_dir data/medical_hf_classroom/reward/train \
  --validation_file_dir data/medical_hf_classroom/reward/valid \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 1000 \
  --max_eval_samples 100 \
  --max_steps 50 \
  --eval_steps 25 \
  --save_steps 25 \
  --save_strategy steps \
  --save_total_limit 2 \
  --max_source_length 1024 \
  --max_target_length 1024 \
  --output_dir runs/dpo_lora_hf_classroom \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --tool_format default \
  --cache_dir ./cache \
  --overwrite_cache True \
  --preprocessing_num_workers 1 \
  --logging_steps 10
```

检查训练结果：

```Bash
cat runs/dpo_lora_hf_classroom/train_results.json
cat runs/dpo_lora_hf_classroom/eval_results.json
```

参考运行记录：

```Plaintext
起点模型：runs/sft_lora_hf_classroom_merged
训练数据：data/medical_hf_classroom/reward/train/train.jsonl，1000 条
验证数据：data/medical_hf_classroom/reward/valid/valid.jsonl，100 条
长度过滤后训练样本：997 条
训练步数：50 steps
训练时长：720.41 秒，约 12 分 00 秒
train_loss：0.4284
eval_loss：0.0905
eval_rewards/accuracies：0.9700
eval_rewards/margins：3.5743
GPU 峰值显存：15492 MiB / 32768 MiB
GPU 活跃平均显存：约 13684 MiB
GPU 活跃平均利用率：约 23.50%
输出目录：runs/dpo_lora_hf_classroom
```

注意：`eval_rewards/accuracies=0.97` 只说明模型在这批 chosen/rejected 偏好数据上更偏向 chosen，不等于真实医学回答质量已经变好。
同样，这里的 `eval_loss` 和 `eval_rewards` 仅用于课堂验证，不应当和正式 benchmark 混用。

##### Step D\-2：合并 HF\-DPO LoRA

推理时可以直接加载 LoRA；如果希望得到完整模型目录，也可以合并：

```Bash
CUDA_VISIBLE_DEVICES=0 python tools/merge_peft_adapter.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --tokenizer_path runs/sft_lora_hf_classroom_merged \
  --lora_model runs/dpo_lora_hf_classroom \
  --output_dir runs/dpo_lora_hf_classroom_merged
```

检查：

```Bash
ls -lh runs/dpo_lora_hf_classroom_merged/model.safetensors
```

##### Step D\-3：准备推理问题

创建 5 个固定问题，保证 SFT 和 DPO 使用同一批输入：

```Bash
cat > data/hf_compare_eval.txt <<'EOF'
乙肝和丙肝有什么区别？
高血压患者日常需要注意什么？
儿童发烧到多少度需要及时就医？
阴道炎有什么症状？
糖尿病患者饮食要注意什么？
EOF
```

检查：

```Bash
wc -l data/hf_compare_eval.txt
cat data/hf_compare_eval.txt
```

##### Step D\-4：运行 HF\-SFT 模型推理

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --data_file data/hf_compare_eval.txt \
  --output_file runs/compare_hf_sft_predictions.jsonl \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --repetition_penalty 1.05 \
  --eval_batch_size 1
```

##### Step D\-5：运行 HF\-DPO 模型推理

使用同一个 SFT 起点，再加载 DPO adapter：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/sft_lora_hf_classroom_merged \
  --lora_model runs/dpo_lora_hf_classroom \
  --data_file data/hf_compare_eval.txt \
  --output_file runs/compare_hf_dpo_predictions.jsonl \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --repetition_penalty 1.05 \
  --eval_batch_size 1
```

如果已经合并了 DPO，也可以直接加载合并模型：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/inference.py \
  --base_model runs/dpo_lora_hf_classroom_merged \
  --data_file data/hf_compare_eval.txt \
  --output_file runs/compare_hf_dpo_merged_predictions.jsonl \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --repetition_penalty 1.05 \
  --eval_batch_size 1
```

##### Step D\-6：生成 HTML 可视化页面

本课提供一个静态 HTML 生成脚本：

```Bash
python -m py_compile tools/build_inference_comparison_html.py
```

生成对比页面：

```Bash
python tools/build_inference_comparison_html.py \
  --sft_file runs/compare_hf_sft_predictions.jsonl \
  --rl_file runs/compare_hf_dpo_predictions.jsonl \
  --output_html runs/inference_compare_hf_sft_vs_dpo.html \
  --sft_name HF-SFT \
  --rl_name HF-DPO
```

生成后检查：

```Bash
ls -lh runs/inference_compare_hf_sft_vs_dpo.html
```

在浏览器中打开：

```Plaintext
/home/ubuntu/MedicalGPT/runs/inference_compare_hf_sft_vs_dpo.html
```

如果是在远程服务器上，也可以临时启动静态文件服务：

```Bash
cd /home/ubuntu/MedicalGPT
python -m http.server 8899
```

然后在本地浏览器访问：

```Bash
SERVER_IP=1.2.3.4      # 改成教师发放的服务器公网 IP
echo "http://${SERVER_IP}:8899/runs/inference_compare_hf_sft_vs_dpo.html"
```

##### Step D\-7：本次推理观察

实测输出：

```Plaintext
SFT 推理结果：runs/compare_hf_sft_predictions.jsonl，5 条
DPO 推理结果：runs/compare_hf_dpo_predictions.jsonl，5 条
HTML 可视化：runs/inference_compare_hf_sft_vs_dpo.html
SFT 平均回答长度：约 364 字
DPO 平均回答长度：约 463 字
```

观察结论：

1. HF\-SFT 模型已经能生成医疗问答风格文本，但仍有事实错误、串题和重复。

2. HF\-DPO 模型在偏好数据指标上表现很好，但本次 50 step 小样本 DPO 后，推理输出并没有稳定变好，部分问题重复更严重。

3. 这说明偏好对齐指标和真实生成质量不是一回事。DPO 只能让模型更贴近当前 chosen/rejected 数据的偏好分布；如果数据质量、训练步数、模板或奖励方向有问题，生成效果可能变差。

4. 医疗模型评估必须结合人工事实核查、安全性检查和更系统的评测集，不能只看 loss、reward accuracy 或单个可视化页面。

# 拓展任务，完成主线后选择

### 拓展 1：修改训练样本数观察效果

分别运行：

```Plaintext
max_train_samples=100
max_train_samples=500
max_train_samples=1000
```

比较 loss 和推理质量。

### 拓展 2：比较 LoRA rank

比较：

```Plaintext

lora_rank=4
lora_rank=8
lora_rank=16
```

记录显存、速度和回答变化。

### 拓展 3：Gradio 页面体验

启动：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/gradio_demo.py \
  --base_model runs/sft_merged \
  --port 8201 \
  --max_new_tokens 256
```

访问：

```Bash
SERVER_IP=1.2.3.4      # 改成教师发放的服务器公网 IP
GRADIO_PORT=8201
echo "http://${SERVER_IP}:${GRADIO_PORT}"
```

### 拓展 4：FastAPI 服务

启动：

```Bash
CUDA_VISIBLE_DEVICES=0 python demo/fastapi_server_demo.py \
  --base_model runs/sft_merged \
  --port 8301 \
  --max_new_tokens 256
```

如果需要在浏览器或其他机器访问服务地址，先生成本组 URL：

```Bash
SERVER_IP=1.2.3.4      # 改成教师发放的服务器公网 IP
FASTAPI_PORT=8301
echo "http://${SERVER_IP}:${FASTAPI_PORT}"
```

请求：

```Bash
curl -X POST http://127.0.0.1:8301/chat \
  -H "Content-Type: application/json" \
  -d '{"input":"乙肝和丙肝有什么区别？"}'
```

### 拓展 5：数据格式转换

准备一个 `qa_demo.json`：

```JSON
[
  {"input": "感冒发烧怎么办？", "output": "建议休息、补液、监测体温。如高热不退或伴随严重症状，应及时就医。"}
]
```

转换：

```Bash
python tools/convert_dataset.py \
  --in_file qa_demo.json \
  --out_file qa_demo.jsonl \
  --data_type qa \
  --file_type json
```

检查：

```Bash
cat qa_demo.jsonl
```

### 拓展 6：了解 RM/PPO、ORPO、GRPO、OPD

阅读脚本：

```Bash
sed -n '1,180p' scripts/run_rm.sh
sed -n '1,180p' scripts/run_ppo.sh
sed -n '1,180p' scripts/run_orpo.sh
sed -n '1,180p' scripts/run_grpo.sh
sed -n '1,180p' scripts/run_opd.sh
```

回答：

1. DPO 和 RM/PPO 的主要区别是什么？

2. ORPO 为什么不需要 reference model？

3. OPD 为什么需要 teacher model？

# 参考资料

1. MedicalGPT GitHub 仓库：https://github\.com/shibing624/MedicalGPT

2. MedicalGPT 数据集文档：`docs/datasets.md`

3. MedicalGPT 训练细节：`docs/training_details.md`

4. MedicalGPT 训练参数：`docs/training_params.md`

5. MedicalGPT 官方脚本：`scripts/run_pt.sh`、`scripts/run_sft.sh`、`scripts/run_dpo.sh`

6. Hugging Face Transformers 文档：https://huggingface\.co/docs/transformers

7. PEFT 文档：https://huggingface\.co/docs/peft

8. TRL 文档：https://huggingface\.co/docs/trl

