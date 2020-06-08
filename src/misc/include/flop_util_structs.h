#ifndef __FLOP_UTIL_STRUCTS__
#define __FLOP_UTIL_STRUCTS__
#include <stdint.h>

typedef struct flop_counter_t {
  unsigned long int __pad1;
  unsigned long int __pad2;
  unsigned long int copy;
  unsigned long int flop;
  int tid;
  unsigned long int potrf;
  unsigned long int trsm;
  unsigned long int syrk;
  unsigned long int update;
  unsigned long int geqrfu;
  unsigned long int geqrfv;
  unsigned long int geqrfv_gemm;
  unsigned long int gesvd;
  unsigned long int gesvd_gemm;
  unsigned long int gesvd_lasetu;
  unsigned long int gesvd_lasetv;
  unsigned long int gesvd_lacpyu;
  unsigned long int gesvd_lacpyv;
  unsigned long int gesvd_dscal;
  unsigned long int ormqru;
  unsigned long int ormqru_laset;
  unsigned long int ormqru_lacpy;
  unsigned long int ormqrv;
  unsigned long int ormqrv_laset;
  unsigned long int ormqrv_lacpy;
  unsigned long int omatcopy;

  int numthreads;
  unsigned long int __pad3;
  unsigned long int __pad4;
} flop_counter;
#define FLOP_NUMTHREADS 256

#endif
