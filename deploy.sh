#!/bin/bash
# ----------------------------------------
# 一键部署套利模拟系统
# 适用于 Render / Linux / macOS
# ----------------------------------------

# 1️⃣ 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2️⃣ 升级 pip / setuptools / wheel
python -m pip install --upgrade pip setuptools wheel

# 3️⃣ 安装系统依赖（Render Ubuntu 系统）
sudo apt-get update
sudo apt-get install -y build-essential python3.11-dev libatlas-base-dev gfortran

# 4️⃣ 安装 Python 包依赖
pip install --no-cache-dir -r requirements.txt

# 5️⃣ 创建日志和报表文件夹
mkdir -p logs reports

# 6️⃣ 启动套利模拟器
python simulator.py
