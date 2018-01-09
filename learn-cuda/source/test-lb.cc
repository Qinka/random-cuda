
#include <linear-combination.h>
#include <stdio.h>

int main() {
  uint8_t m1[] = {1,2,3,4,5,6};
  uint8_t m2[] = {6,5,4,3,2,1};
  uint8_t m3[] = {9,9,9,9,9,9};

  linear_combination(1,m1,1,m2,2,3,m3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j)
      printf("%u ",m1[i*3+j]);
    printf("\n");
  }
  printf("+\n");
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j)
      printf("%u ",m2[i*3+j]);
    printf("\n");
  }
  printf("=\n");
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j)
      printf("%u ",m3[i*3+j]);
    printf("\n");
  }
  return 0;
}
