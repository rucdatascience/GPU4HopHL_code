#ifndef _LOUVAIN_H
#define _LOUVAIN_H
#include "graph/graph_v_of_v.h"
#include "definition/hub_def.h"
typedef struct _louvain Louvain;
#define MAX_CIRCLE 20
typedef struct _node {
  int count;     // node number of current cluster int clsid;
  int clsid;     // the upper cluster id
  int next;      // the next node which belong to the same upper cluster
  int prev;      // the prev node which belong to the same upper cluster
  int first;     // the first child of current community
  int eindex;    // first neighbor index
  double kin;    // current node in weight
  double kout;   // current node out weight
  double clskin; // the kin value for new community
  double clstot; // nodes which belong to the same cluster have the same clstot;
} Node;

typedef struct _edge {
  int left; // left <------ right
  int right;
  int next;      // next neighbor index for node left
  double weight; // edge weight from right to left
} Edge;

struct _louvain {
  int clen;
  int elen;
  int nlen;
  int olen;
  int *cindex;
  double sumw;
  Node *nodes;
  Edge *edges;
};

Louvain * create_louvain(const char * input);
int learn_louvain(Louvain * lv);
void save_louvain(Louvain * lv);
void free_louvain(Louvain *lv);
Louvain *mycreate_louvain(graph_v_of_v<disType> &G);

#endif //LOUVAIN_H