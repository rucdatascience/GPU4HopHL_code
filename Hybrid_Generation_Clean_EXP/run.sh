#!/bin/bash
echo "测试"

rm -rf build
mkdir build
cd build
cmake3 ..
make
cd ..

# 设置实验参数
data_dir="/home/mdnd/data_exp_10" # 数据文件夹路径
output="/home/mdnd/Hybrid_Generation_Clean_EXP/result_Hybrid_4GPU_data_exp_10.csv" # 输出文件名
gmax="1000"
thread="1000"

# 遍历 new-data 目录中的每个子目录
for algo in $(seq 5 5); do # algo 取值为 1、2、3、4、5, CPU, CPU_opt, GPU, Hybrid_4GPU, Hybrid_CPU_GPU
    for iter in $(seq 1 1); do # 迭代几次
        for dataset_dir in "$data_dir"/*; do
            if [ -d "$dataset_dir" ]; then # 检查是否为目录

                # 获取数据集文件和查询文件
                dataset_file="$dataset_dir/*.e" # 数据集文件
                # query_file="$dataset_dir/*.query" # 查询文件
                
                # 检查是否存在数据集文件和查询文件
                if ls $dataset_file 1> /dev/null 2>&1; then

                    # 将数据集和查询文件路径具体化
                    dataset=$(ls $dataset_file)
                    # query_path=$(ls $query_file)

                    # 运行测试
                    for upper_k in $(seq 5 5); do  # upper_k 从 2 到 5
                        echo "./build/bin/Test $dataset_file $upper_k $algo $output $gmax $thread 1"
                        ./build/bin/Test "$dataset" "$upper_k" "$algo" "$output" "$gmax" "$thread" 1 # 运行 C++ 程序
                    done
                else
                    echo "Warning: 目录 $dataset_dir 中未找到合适的数据集文件或查询文件，跳过该目录"
                fi
            fi
        done
    done
done

echo "测试完成，结果已写入 $output 文件。"