#pragma once
#include<vector>
#include<string>
#include<iostream>
#include <fstream>

template<typename T>
void binary_save_vector(std::string path, std::vector<T>& myVector)
{
    std::ofstream FILE(path, std::ios::out | std::ofstream::binary);

    // Store its size
    int size = myVector.size();
    FILE.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Store its contents
    if (size == 0)
    {
        return;
    }
    FILE.write(reinterpret_cast<const char*>(&myVector[0]), myVector.size() * sizeof(T)); // T cannot be bool!
    FILE.close();
}

template<typename T>
void binary_read_vector(std::string path, std::vector<T>& myVector)
{
    std::vector<T>().swap(myVector);

    std::ifstream FILE(path, std::ios::in | std::ifstream::binary);

    int size = 0;
    FILE.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!FILE)
    {
        std::cout << "Unable to open file " << path << std::endl << "Please check the file location or file name." << std::endl; // throw an error message
        exit(1); // end the program
    }
    myVector.resize(size);
    T f;
    for (int k = 0; k < size; ++k) {
        FILE.read(reinterpret_cast<char*>(&f), sizeof(f));
        myVector[k] = f;
    }
    std::vector<T>(myVector).swap(myVector);
}



/*
---------an example main file-----------
#include <text_mining/binary_save_read_vector.h>

int main()
{
    example_binary_save_read_vector();
}
------------------------------------
*/

void example_binary_save_read_vector() {

    std::vector<int> a = { 1, 2, 4 };
    binary_save_vector("b.txt", a);
    std::vector<int> b;
    binary_read_vector("b.txt", b);
    std::cout << b[2] << std::endl;

}

