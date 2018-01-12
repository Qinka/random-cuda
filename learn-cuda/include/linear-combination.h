/**
 * linear combination for 2-D matrix
 * Copyright 2018 (C) Johann Lee <me@qinka.pro>
 */


#ifndef _LINEAR_COMBINATION_H_
#define _LINEAR_COMBINATION_H_

#include <stdint.h>

#ifndef EXPORT_C_API
#ifdef WIN32
# define EXPORT_C_API extern "C" _declspec(dllexport)
#else // WIN32
# define EXPORT_C_API extern "C"
#endif // WIN32
#endif // ! EXPORT_C_API




// Error codes
#define Success 0

/**
 * linear_combination
 * @return status code
 * @param coe1 coefficient 1
 * @param m1 matrix 1
 * @param coe2 coefficient 2
 * @param m2 matrix 2
 * @param row rows
 * @param col columns
 * @param m3 matrix of result
 *
 * m3 = coe1 * m1 + coe2 * m2
 */
EXPORT_C_API
int linearCombination(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int _size, uint8_t* m3);


#endif // _LINEAR_COMBINATION_H_
