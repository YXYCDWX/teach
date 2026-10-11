#!/usr/bin/env bash
# 3D 检测训练链: CenterPoint (Linux/4090 服务器版)
# 用法: bash run_chain3d.sh
# 前置: conda/pip 环境装好 torch(cuda) numpy pillow; 数据放 ./v1.0-mini/
# 特性: 断点续训 —— 中断后重跑本脚本, 已完成的模型自动跳过, 跑一半的自动接续
set -e
cd "$(dirname "$0")"
PY=${PY:-python}

run () {
    local name=$1; shift
    echo "=================================================="
    echo "训练 $name"
    echo "=================================================="
    "$PY" "$@" 2>&1 | tee -a "chain_${name}.log"
}

run centerpoint    train_3d.py

echo "=================================================="
echo "3D 链全部完成"
echo "=================================================="
