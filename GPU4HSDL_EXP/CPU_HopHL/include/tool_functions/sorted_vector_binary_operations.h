#pragma once
#include <limits>
#include <stdexcept>
#include <algorithm>

/*
for a sorted vector<pair<int,T>>, this dunction conducted binary (divide and
conquer) operations on this vector;

the int values are unique and sorted from small to large

https://blog.csdn.net/EbowTang/article/details/50770315
*/

template <typename T>
bool sorted_vector_binary_operations_search(
    std::vector<std::pair<int, T>> &input_vector, int key) {

  /*return true if key is in vector; time complexity O(log n)*/

  int left = 0, right = input_vector.size() - 1;

  while (left <= right) {
    int mid = left + ((right - left) /
                      2); // mid is between left and right (may be equal);
    if (input_vector[mid].first == key) {
      return true;
    } else if (input_vector[mid].first > key) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  return false;
}

template <typename T>
T sorted_vector_binary_operations_search_weight(
    std::vector<std::pair<int, T>> &input_vector, int key) {

  /*return std::numeric_limits<T>::max() if key is not in vector; time
   * complexity O(log n)*/

  int left = 0, right = input_vector.size() - 1;

  while (left <= right) {
    int mid = left + ((right - left) /
                      2); // mid is between left and right (may be equal);
    if (input_vector[mid].first == key) {
      return input_vector[mid].second;
    } else if (input_vector[mid].first > key) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  return std::numeric_limits<T>::max();
}

template <typename T>
int sorted_vector_binary_operations_search_position(
    std::vector<std::pair<int, T>> &input_vector, int key) {

  /*return -1 if key is not in vector; time complexity O(log n)*/

  int left = 0, right = input_vector.size() - 1;

  while (left <= right) {
    int mid = left + ((right - left) / 2);
    if (input_vector[mid].first == key) {
      return mid;
    } else if (input_vector[mid].first > key) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  return -1;
}

template <typename T>
void sorted_vector_binary_operations_erase(
    std::vector<std::pair<int, T>> &input_vector, int key) {

  /*erase key from vector; time complexity O(log n + size()-position ), which is
  O(n) in the worst case, as the time complexity of erasing an element from a
  vector is the number of elements behind this element*/

  if (input_vector.size() > 0) {
    int left = 0, right = input_vector.size() - 1;

    while (left <= right) {
      int mid = left + ((right - left) / 2);
      if (input_vector[mid].first == key) {
        input_vector.erase(input_vector.begin() + mid);
        break;
      } else if (input_vector[mid].first > key) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
  }
}
template <typename T>
int sorted_vector_binary_operations_insert(std::vector<std::pair<int, T>> &input_vector, int key, T load) {
    // 使用 std::lower_bound 查找适当的插入位置
    auto it = std::lower_bound(input_vector.begin(), input_vector.end(), std::make_pair(key, load),
                               [](const std::pair<int, T>& a, const std::pair<int, T>& b) {
                                   return a.first < b.first;
                               });

    // 如果找到了匹配的键，则更新其值
    if (it != input_vector.end() && it->first == key) {
        it->second = load;
        return std::distance(input_vector.begin(), it); // 返回位置
    }

    // 否则，在计算出的位置插入新元素
    it = input_vector.insert(it, {key, load});
    return std::distance(input_vector.begin(), it); // 返回新插入元素的位置
}
