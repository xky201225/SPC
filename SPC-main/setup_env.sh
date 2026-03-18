#!/bin/bash

# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装/更新兼容的 PyTorch 及其生态 (让 pip 自动选择与新版 vllm 兼容的 torch 2.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖 (自动解析并升级到最新稳定版)
pip install -r requirements_linux.txt

# 4. 强制更新 transformers 和 vllm 以防止属性报错
pip install --upgrade transformers vllm

# 5. 打印安装成功提示
echo "=========================================================="
echo "环境安装及升级完成！请运行以下命令开始评估："
echo "python eval/infer_batch.py"
echo "=========================================================="