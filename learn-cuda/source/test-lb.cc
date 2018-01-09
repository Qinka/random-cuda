
#include <linear-combination.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
  int row = 2;
  int col = 3;
  if(argc >= 2) {
    sscanf(argv[0],"%d",&row);
    sscanf(argv[1],"%d",&col);
  }
  uint8_t* m1 = new uint8_t[row * col];
  uint8_t* m2 = new uint8_t[row * col];
  uint8_t* m3 = new uint8_t[row * col];

  linear_combination(1,m1,1,m2,row,col,m3);

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%u ",m1[i*col+j]);
    printf("\n");
  }
  printf("+\n");
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%u ",m2[i*col+j]);
    printf("\n");
  }
  printf("=\n");
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%u ",m3[i*col+j]);
    printf("\n");
  }
  return 0;
}
