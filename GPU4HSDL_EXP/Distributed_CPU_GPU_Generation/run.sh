# ./build/test /home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web.e 5 0 /home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web.query /home/pengchang/GPU4HSDL_EXP/CPU_HopHL/Res/git_web.5.HSDL
#!/bin/bash

# 设置参数
data_dir="/home/pengchang/GPU4HSDL_EXP/new-data"          # 数据文件夹路径
output="/home/pengchang/GPU4HSDL_EXP/results4.csv"         # 输出文件名

# 清空已有的结果文件并写入表头

# 遍历 new-data 目录中的每个子目录
for dataset_dir in "$data_dir"/*; do
    if [ -d "$dataset_dir" ]; then  # 检查是否为目录
        # 获取数据集文件和查询文件
        dataset_file="$dataset_dir/*.e"          # 数据集文件
        query_file="$dataset_dir/*.query"        # 查询文件
        
        # 检查是否存在数据集文件和查询文件
        if ls $dataset_file 1> /dev/null 2>&1 && ls $query_file 1> /dev/null 2>&1; then
            # 将数据集和查询文件路径具体化
            dataset=$(ls $dataset_file)
            query_path=$(ls $query_file)

            # 运行测试
            for algo in 0; do  # algo 取值为 0 和 1,Hybrid,GPU
                for k in $(seq 2 4); do  # upper_k 从 2 到 4
                    echo "./build/bin/test $dataset_file $k $algo $query_file 1"
                    ./build/bin/test "$dataset" "$k" "$algo" "$query_path" "$output" 1 # 运行 C++ 程序
                done
            done
        else
            echo "Warning: 目录 $dataset_dir 中未找到合适的数据集文件或查询文件，跳过该目录。"
        fi
    fi
done

echo "测试完成，结果已写入 $output 文件。"
