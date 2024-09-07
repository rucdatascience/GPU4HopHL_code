#!/bin/bash

# 指定 CUDA 11.8 的路径
CUDA_PATH=/usr/local/cuda-11.8

# 检查 CUDA 11.8 是否安装
if [ -d "$CUDA_PATH" ]; then
    echo "切换到 CUDA 11.8"

    # 更新 PATH
    export PATH=$CUDA_PATH/bin:$PATH

    # 更新 LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

    # 输出 CUDA 版本以确认切换
    nvcc --version
else
    echo "CUDA 11.8 未找到，请确保它已安装。"
fi
