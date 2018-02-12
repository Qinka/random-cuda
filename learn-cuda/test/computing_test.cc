
#include <linear-combination.h>
#include <stdint.h>
#include <iostream>
#include<cstdlib>
#include<ctime>

using namespace std;

#define N (1000000)
#define T (1000)


int main () {

  uint8_t *a = new uint8_t[N];
  uint8_t *b = new uint8_t[N];
  uint8_t *c = new uint8_t[N];
  cout << "begin" << endl;


  for (int i = 0; i < T; ++i) {
    srand((unsigned)time(NULL));
    for (int j = 0; j < N; ++j) {
      a[j] = rand() % 255;
      b[j] = rand() % 255;
    }
    linearCombination(0.5,a,0.5,b,N,c);
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
