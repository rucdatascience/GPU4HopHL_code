#ifndef HOP_CONSTRAINED_TWO_HOP_LABELS_V2_H
#define HOP_CONSTRAINED_TWO_HOP_LABELS_V2_H
#pragma once

#include "definition/hub_def.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

/* T队列中元素类型 */
struct T_item {
    int vertex;
    short distance;

    __device__ __host__ T_item (int vertex, short distance) : vertex(vertex), distance(distance) {}

    __device__ __host__ T_item() {}

    T_item (const T_item &other) {
        vertex = other.vertex;
        distance = other.distance;
    }

};

/* label type v4 */
// lebal compress
struct hop_constrained_two_hop_label_v4 {
    // hub_vertex(<<37), parent_vertex(<<13), dis(<<3), hop(<<0)
    long long label_info;

    __device__ __host__ hop_constrained_two_hop_label_v4 (long long hub_vertex, long long parent_vertex, long long distance, long long hop) {
        label_info = (hub_vertex << 37) | (parent_vertex << 13) | (distance << 3) | hop;
    }

    __device__ __host__ hop_constrained_two_hop_label_v4 () {}

    hop_constrained_two_hop_label_v4 (const hop_constrained_two_hop_label_v4 &other) {
        label_info = other.label_info;
    }

};

/* label type v3 */
struct hop_constrained_two_hop_label_v3 {
    int hub_vertex, hop;
    weight_type distance;

    __device__ __host__ hop_constrained_two_hop_label_v3 (int hub_vertex, int hop, weight_type distance)
    : hub_vertex(hub_vertex), hop(hop), distance(distance) {}

    __device__ __host__ hop_constrained_two_hop_label_v3() {}

    hop_constrained_two_hop_label_v3 (const hop_constrained_two_hop_label_v3 &other) {
        hub_vertex = other.hub_vertex;
        hop = other.hop;
        distance = other.distance;
    }

};

/* 标签类型 */
struct hop_constrained_two_hop_label_v2 {
    int hub_vertex, parent_vertex, hop;
    weight_type distance;

    __device__ __host__ hop_constrained_two_hop_label_v2 (int hv, int pv, int h, weight_type d)
        : hub_vertex(hv), parent_vertex(pv), hop(h), distance(d) {}

    __device__ __host__ hop_constrained_two_hop_label_v2() {}
    // copy
    __device__ __host__
    hop_constrained_two_hop_label_v2 (const hop_constrained_two_hop_label_v2 &other) {
        hub_vertex = other.hub_vertex;
        parent_vertex = other.parent_vertex;
        hop = other.hop;
        distance = other.distance;
    }
    // __device__ __host__ bool operator <(const hop_constrained_two_hop_label_v2 &y) const {
    //     //congregate the same hub_vertex into a continuous block, so we can use segment to find the label
    //     if(hub_vertex != y.hub_vertex){
    //         return hub_vertex < y.hub_vertex;
    //     }
    //     if (distance != y.distance) {
    //         return distance < y.distance; // < is the max-heap; > is the min heap
    //     } else {
    //         return hop < y.hop; // < is the max-heap; > is the min heap
    //     }
    // }

    // __device__ __host__ bool operator ==(const hop_constrained_two_hop_label_v2 &rhs) const{
    //     return hub_vertex == rhs.hub_vertex &&
    //             parent_vertex == rhs.parent_vertex && hop == rhs.hop &&
    //             distance == rhs.distance;
    // }
};

// 定义哈希函数
// struct DeviceHash {
//     __device__ size_t operator()(int key) const {
//         // 这是一个简单的 hash 函数，你可能需要根据你的需求来修改它
//         key = ((key >> 16) ^ key) * 0x45d9f3b;
//         key = ((key >> 16) ^ key) * 0x45d9f3b;
//         key = (key >> 16) ^ key;
//         return key;
//     }
// };

// namespace std {
// template <> struct hash<hop_constrained_two_hop_label> {
//     __device__ __host__ size_t
//     operator()(const hop_constrained_two_hop_label &label) const {
//     #ifdef __CUDA_ARCH__
//         DeviceHash hash;
//     #else
//         std::hash<int> hash;
//     #endif
//         return hash(label.hub_vertex) ^ (hash(label.parent_vertex) << 1) ^
//                 (hash(label.hop) << 2) ^ (hash(label.distance) << 3);
//     }
//     };
// } // namespace std

#endif // HOP_CONSTRAINED_TWO_HOP_LABELS_H