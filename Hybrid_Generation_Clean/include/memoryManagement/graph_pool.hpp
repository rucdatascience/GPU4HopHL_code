#ifndef GRAPHPOOL_HPP
#define GRAPHPOOL_HPP

#include <thread>
#include <unistd.h>
#include "definition/mmpool_size.h"
#include <iostream>
#include <mutex>

using std::vector;

template <typename T> class Graph_pool {
public:
    vector<vector<T> > graph_group;
    int next_graph = 0;
    std::mutex mtx;

    // ���캯��
    Graph_pool();
    Graph_pool(int Group_Num);
    int get_next_graph();
    int size();

};

// ���캯��
template <typename T> Graph_pool<T>::Graph_pool() {
    next_graph = 0;
}

// ���캯��
template <typename T> Graph_pool<T>::Graph_pool(int Group_Num) {
    graph_group.resize(Group_Num);
    next_graph = 0;
}

// ���ҿտ�
template <typename T> int Graph_pool<T>::get_next_graph() {
    //ʹ����������
    //��ȡ��
    mtx.lock(); // ��ȡ��
    int ret = -1;
    if (next_graph >= graph_group.size()) {
        ret = -1;
    }else{
        ret = next_graph;
        next_graph ++;
    }
    mtx.unlock(); // �ͷ���
    return ret;
}

// ��ѯ graphpool �� size
template <typename T> int Graph_pool<T>::size() {
    //ʹ����������
    //��ȡ��
    mtx.lock(); // ��ȡ��
    int ret = graph_group.size() - next_graph;
    mtx.unlock(); // �ͷ���
    return ret;
}

#endif