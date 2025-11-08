#!/bin/bash
# ----------------------------------------
# 一键部署多交易所套利模拟系统（Render / Linux / macOS）
# 完全避免 pandas metadata-generation-failed
# 使用 Python 3.11 + pandas 2.1.1
# ----------------------------------------

# 1️⃣ 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 2️⃣ 升级 pip / setuptools / wheel
python -m pip install --upgrade pip setuptools wheel

# 3️⃣ 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev gfortran

# 4️⃣ 安装 Python 包依赖（使用 wheel，避免源码编译）
pip install --upgrade --no-cache-dir -r requirements.txt

# 5️⃣ 创建日志和报表文件夹
mkdir -p logs reports

# 6️⃣ 启动套利模拟器
python simulator.py
