#ifndef GRAPHPOOL_HPP
#define GRAPHPOOL_HPP

#include <thread>
#include <unistd.h>
#include "definition/mmpool_size.h"
#include <iostream>
#include <mutex>

template <typename T> class Graph_pool {
public:
    vector<vector<T> > graph_group;
    int next_graph = 0;
    std::mutex mtx;

    // ���캯��
    Graph_pool(int Group_Num);
    vector<T> get_next_graph();

};

// ���캯��
template <typename T> Graph_pool<T>::Graph_pool(int Group_Num) {
    graph_group.resize(Group_Num);
    next_graph = 0;
}

// ���ҿտ�
template <typename T> vector<T> Graph_pool<T>::get_next_graph() {
    //ʹ����������
    //��ȡ��
    mtx.lock(); // ��ȡ��
    vector<int> ret = graph_group[next_graph];
    next_graph ++;
    mtx.unlock(); // �ͷ���
    return ret;
}

#endif