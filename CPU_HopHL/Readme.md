# CPU_HopHL
- CPU版本的HopHL算法，用于对比GPU版本的性能。
- 本项目使用clangd，cmake
- 安装boost库，用于使用fibonacci堆。https://www.boost.org/
- 在HBPLL/test.h文件中，可以修改宏定义DATASET_PATH ，来选择不同的数据集
- cmake编译
```shell
mkdir build
cd build
cmake ..
make
./test
```