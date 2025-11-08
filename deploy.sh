#!/bin/bash
# Render 部署脚本

# 升级 pip、setuptools、wheel
python -m pip install --upgrade pip setuptools wheel

# 安装依赖
pip install --no-cache-dir -r requirements.txt

# 创建日志和报表目录
mkdir -p logs reports
