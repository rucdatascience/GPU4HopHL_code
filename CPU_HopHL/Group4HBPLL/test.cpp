#include "test.h"

int main(int argc,char **argv) {
  int test_algo = -1;
  if (argc > 1) {
    test_algo = atoi(argv[1]);
  } else {
    printf("Usage: %s <test_algo>\n", argv[0]);
    printf("test_algo: 1 - Kmeans\n");
    printf("test_algo: 3 - CT-CORE\n");
    return 0;
  }
  printf("test algo:%d\n",test_algo);
  test_Group4HBPLL(test_algo);
  //test_HSDL();
}