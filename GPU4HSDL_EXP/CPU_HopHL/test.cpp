
#include "HBPLL/test.h"
#include <string>
#include <filesystem> 
int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "./test dataset upper_k algo query_path output is_clean" << std::endl;
        return 1;
    }
  
    std::string dataset = argv[1];
    int upper_k = std::stoi(argv[2]);
    int algo = std::stoi(argv[3]);
    std::string query_path = argv[4];
    std::string output = argv[5];
    int is_clean = std::stoi(argv[6]);
    std::string dataset_name = std::filesystem::path(dataset).stem().string();

    // Remove trailing ".e" if present
    if (dataset_name.size() > 2 && dataset_name.substr(dataset_name.size() - 2) == ".e") {
        dataset_name = dataset_name.substr(0, dataset_name.size() - 2);
    }

    Res result;
    std::string algo_name = (algo == 1) ? "2023WWW" : "HSDL";
    std::string clean = (is_clean ==1 )? "_Clean":"";
    if(algo == 0)
    {
        test_HSDL(dataset, query_path, upper_k, algo, result,1);
        algo_name = "HSDL";
    }
    else if(algo == 1)
    {
        test_HSDL(dataset, query_path, upper_k, algo, result,0);
        algo_name = "2023WWW";
    }
    else if(algo == 2 )
    {
        dijkstra_hopconstrained(dataset,query_path,upper_k,result);
        algo_name = "HopDijkstra";
    }

    
    
    std::ofstream out(output, std::ios::app); // 以追加模式打开文件
    if (!out.is_open()) {
        std::cerr << "无法打开输出文件。" << std::endl;
        return 1;
    }

    
    
    // 追加写入结果到文件
    out << algo_name<< "," << "CPU,"<<dataset_name<<","<<upper_k<<","<<result.index_time << "," 
        << result.size << "," << result.before_clean_size<<","<<result.query_time << "," <<result.before_clean_query_time<<","
        << result.clean_time << std::endl;

    out.close();
    return 0;
}