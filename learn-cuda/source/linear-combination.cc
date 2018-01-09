
/**
 * linear combination for 2-D matrix
 * Copyright 2018 (C) Johann Lee <me@qinka.pro>
 */


#ifndef _LINEAR_COMBINATION_H_
#define _LINEAR_COMBINATION_H_

#include <linear-combination.h>
#include <math.h>
#include <stdint.h>


int linear_combination(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int row, int col, uint8_t* m3) {
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++) {
      float tmp = coe1 * m1[i*col + j] + coe2 * m2[i*col + j];
      m3[i*col + j] = (uint8_t)(fmaxf(fminf(tmp,255),0));
    }
  return 0;
}


#endif // _LINEAR_COMBINATION_H_
