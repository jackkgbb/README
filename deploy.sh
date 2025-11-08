#!/bin/bash
# ----------------------------------------
# Render 一键部署套利模拟器
# Python 3.13 + pandas 2.0.1
# ----------------------------------------

# 1️⃣ 创建并激活虚拟环境
python3.13 -m venv venv
source venv/bin/activate

# 2️⃣ 升级 pip / setuptools / wheel
python -m pip install --upgrade pip setuptools wheel

# 3️⃣ 安装依赖，强制使用 wheel 避免源码编译 pandas
pip install --no-cache-dir --only-binary=:all: pandas==2.0.1
pip install --no-cache-dir -r requirements.txt

# 4️⃣ 创建日志和报表文件夹
mkdir -p logs reports

# 5️⃣ 启动套利模拟器
python simulator.py
