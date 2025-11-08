#!/bin/bash
# ----------------------------------------
# Render 一键部署套利模拟器
# 使用 Python 3.11 + pandas 2.1.1
# ----------------------------------------

# 1️⃣ 创建并激活虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 2️⃣ 升级 pip / setuptools / wheel
python -m pip install --upgrade pip setuptools wheel

# 3️⃣ 安装依赖
pip install --no-cache-dir -r requirements.txt

# 4️⃣ 创建日志和报表文件夹
mkdir -p logs reports

# 5️⃣ 启动套利模拟器
python simulator.py
