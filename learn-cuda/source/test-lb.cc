
#include <linear-combination.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char* argv[]) {
  int row = 2;
  int col = 3;
  float coe1 = 1;
  float coe2 = 1;
  if(argc >= 3) {
    sscanf(argv[1],"%d",&row);
    sscanf(argv[2],"%d",&col);
  }
  if(argc >= 5) {
    sscanf(argv[3],"%f",&coe1);
    sscanf(argv[4],"%f",&coe2);
  }
  uint8_t* m1 = new uint8_t[row * col];
  uint8_t* m2 = new uint8_t[row * col];
  uint8_t* m3 = new uint8_t[row * col];

  srand((int)time(0));

  for (int i = 0; i < row * col; ++i) {
    m1[i] = 1;
    m2[i] = 2;
    m3[i] = 5;
  }


  linearCombination(coe1,m1,coe2,m2,row * col,m3);

  
  for (int i = 0; i < row; ++i) 
    for (int j = 0; j < col; ++j)
      if (m3[i*col+j] != 1 * coe1 + 2 * coe2)
        printf("(%d,%d) = %u is error\n",i,j,m3[i*col+j]);
 

  /*
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%3u ",m1[i*col+j]);
    printf("\n");
  }
  printf("+\n");
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%3u ",m2[i*col+j]);
    printf("\n");
  }
  printf("=\n");
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j)
      printf("%3u ",m3[i*col+j]);
    printf("\n");
    }*/
  return 0;
}
