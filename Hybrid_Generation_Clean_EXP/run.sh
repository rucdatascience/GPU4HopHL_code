# ./build/test /home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web.e 5 0 /home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web.query /home/pengchang/GPU4HSDL_EXP/CPU_HopHL/Res/git_web.5.HSDL
#!/bin/bash

rm -rf build
mkdir build
cd build
cmake3 ..
make
cd ..

# ���ò���
data_dir="/home/mdnd/data_exp_10"          # �����ļ���·��
# output="/home/mdnd/Hybrid_Generation_Clean_EXP/result_Hybrid_4GPU_data_exp_0.csv"         # ����ļ���
output="/home/mdnd/Hybrid_Generation_Clean_EXP/result_Hybrid_4GPU_data_exp_10.csv"         # ����ļ���
# output="/home/mdnd/Hybrid_Generation_Clean_EXP/result_Hybrid_1CPU_data_exp_reddit.csv"         # ����ļ���
# output="/home/mdnd/Hybrid_Generation_Clean_EXP/result_Hybrid_4GPU_data_exp_1.csv"
gmax="1000"
thread="1000"
# ������еĽ���ļ���д���ͷ

# ���� new-data Ŀ¼�е�ÿ����Ŀ¼
# for iter in $(seq 1 1); do
    # for dataset_dir in "$data_dir"/*; do
    #     if [ -d "$dataset_dir" ]; then  # ����Ƿ�ΪĿ¼
    #         # ��ȡ���ݼ��ļ��Ͳ�ѯ�ļ�
    #         dataset_file="$dataset_dir/*.e"          # ���ݼ��ļ�
    #         # query_file="$dataset_dir/*.query"        # ��ѯ�ļ�
            

    #         # ����Ƿ�������ݼ��ļ��Ͳ�ѯ�ļ�
    #         if ls $dataset_file 1> /dev/null 2>&1; then
    #             # �����ݼ��Ͳ�ѯ�ļ�·�����廯
    #             dataset=$(ls $dataset_file)
    #             # query_path=$(ls $query_file)

    #             # ���в���
    #             for algo in $(seq 1 2); do  # algo ȡֵΪ 0 �� 1,Hybrid,GPU
    #                 for k in $(seq 2 5); do  # upper_k �� 2 �� 4
    #                     echo "./build/bin/Test $dataset_file $k $algo $output 1"
    #                     ./build/bin/Test "$dataset" "$k" "$algo" "$output" 1 # ���� C++ ����
    #                 done
    #             done
    #         else
    #             echo "Warning: Ŀ¼ $dataset_dir ��δ�ҵ����ʵ����ݼ��ļ����ѯ�ļ���������Ŀ¼��"
    #         fi
    #     fi
    # done
# done
for algo in $(seq 5 5); do  # algo ȡֵΪ1��2��3��4��5, CPU, CPU_opt, GPU, Hybrid_4GPU, Hybrid_CPU_GPU
    for iter in $(seq 1 1); do
        for dataset_dir in "$data_dir"/*; do
            if [ -d "$dataset_dir" ]; then  # ����Ƿ�ΪĿ¼
                # ��ȡ���ݼ��ļ��Ͳ�ѯ�ļ�
                dataset_file="$dataset_dir/*.e"          # ���ݼ��ļ�
                # query_file="$dataset_dir/*.query"        # ��ѯ�ļ�
                
                # ����Ƿ�������ݼ��ļ��Ͳ�ѯ�ļ�
                if ls $dataset_file 1> /dev/null 2>&1; then
                    # �����ݼ��Ͳ�ѯ�ļ�·�����廯
                    dataset=$(ls $dataset_file)
                    # query_path=$(ls $query_file)

                    # ���в���
                    for k in $(seq 5 5); do  # upper_k �� 2 �� 4
                        echo "./build/bin/Test $dataset_file $k $algo $output $gmax $thread 1"
                        ./build/bin/Test "$dataset" "$k" "$algo" "$output" "$gmax" "$thread" 1 # ���� C++ ����
                    done
                else
                    echo "Warning: Ŀ¼ $dataset_dir ��δ�ҵ����ʵ����ݼ��ļ����ѯ�ļ���������Ŀ¼"
                fi
            fi
        done
    done
done

echo "������ɣ������д�� $output �ļ���"