#include<bits/stdc++.h>
using namespace std;
ifstream fin("p2p-Gnutella31.txt");
ofstream fout("p2p-Gnutella31.e");
map<int,int> mp_node;
map<pair<int,int>,int> mp_edge;
int cnt,cnt_edge;
int edge1[10000005],edge2[10000005];
int main(){
	srand(time(0));
	string line;
	getline(fin,line);
	getline(fin,line);
	getline(fin,line);
	getline(fin,line);
	int a,b;
	while(fin>>a>>b){
		if(mp_node.find(a)==mp_node.end()){
			mp_node[a]=cnt++; 
		}
		if(mp_node.find(b)==mp_node.end()){
			mp_node[b]=cnt++; 
		}
		if(mp_edge.find(make_pair(a,b))==mp_edge.end()){
			mp_edge[make_pair(a,b)]=1;
			mp_edge[make_pair(b,a)]=1;
			edge1[cnt_edge]=a,edge2[cnt_edge]=b;
			cnt_edge++;
		}
	}
	fout<<"Endpoint1 Endpoint2 Random-Weight (vertex ID from 0 to |V|-1; |V|= "<<cnt<<"; |E|= "<<cnt_edge<<")"<<endl;
	for(int i=0;i<cnt_edge;i++){
		fout<<edge1[i]<<" "<<edge2[i]<<" "<<((rand()%100)+1)<<endl;
	}
	return 0;
}
/*

Endpoint1 Endpoint2 Random-Weight (vertex ID from 0 to |V|-1; |V|= 30855; |E|= 577873)

*/
