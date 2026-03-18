#!/bin/bash

# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装 CUDA 12.1 版本的 PyTorch (Linux 下通常非常顺利)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖
pip install -r requirements_linux.txt

# 4. 打印安装成功提示
echo "=========================================================="
echo "环境安装完成！请运行以下命令开始评估："
echo "python eval/infer_batch.py"
echo "=========================================================="