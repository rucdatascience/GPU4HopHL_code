# GPU HSDL
- GPU������hop label�Ĵ���
- �����д������cmake 3.8, g++ 4.8.5, cuda 12.4, boost 1.85.0
- cmake�������У�Ҳ����ֱ������Ŀ¼�µ�test.sh
```
rm -rf build
mkdir build
cd build
cmake ..
make
./bin/Test
```

thread_num = 1000
V = 2000000
G_max = 10000
hop_cst = 4

Graph: 2 * 100000000 int
L: 10G
hash: 1000 * 10000 * 5 int
d_hash: 1000 * 2000000 int
d_vector: 1000 * 2000000 int
L_pushback: 1000 * 2000000 int
T_pushback: 1000 * 2000000 int