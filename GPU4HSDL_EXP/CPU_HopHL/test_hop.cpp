#include "HBPLL/test.h"
using namespace std;
int main()
{
    string dataset = "/home/pengchang/new-data/as-skitter/as-skitter.e";
    string query = "/home/pengchang/new-data/as-skitter/as-skitter.query";
    int upper = 2;
    Res r1,r2;
    //dijkstra_hopconstrained(dataset,query,upper,r1);
    dijkstra_hopconstrained_speed_up(dataset,query,upper,r2);
    
    printf("r1: %lf\n",r1.query_time);
    printf("r2: %lf\n",r2.query_time);
    //dijkstra_hopconstrained_correctness(dataset,query,upper,r1);

}