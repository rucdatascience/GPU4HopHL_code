# GPU_query
- GPU 版本的 Query 算法，用于对比 CPU 版本的性能。
- 代码编写环境：cmake 3.8, g++ 4.8.5, cuda 12.4, boost 1.85.0
- cmake编译运行，也可以直接运行目录下的test.sh

170 sever require:
source /opt/rh/devtoolset-11/enable
source switch-cuda.sh 11.8

```shell
rm -rf build
mkdir build
cd build
cmake3 ..
make
./test
```