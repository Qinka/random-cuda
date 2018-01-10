

/**
 * linear combination for 2-D matrix
 * Copyright 2018 (C) Johann Lee <me@qinka.pro>
 */


#ifndef _LINEAR_COMBINATION_C_
#define _LINEAR_COMBINATION_C_

#include <linear-combination.h>
#include <math.h>
#include <stdint.h>

int linearCombination(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int size, uint8_t* m3) {
  for (int i = 0; i < size; i++) {
    float tmp = coe1 * m1[i] + coe2 * m2[i];
    m3[i] = (uint8_t)(fmaxf(fminf(tmp,255),0));
  }
  return 0;
}


#endif // _LINEAR_COMBINATION_C_
