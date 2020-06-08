#ifndef __HCORE__
#define __HCORE__

#ifdef MKL
  #include <mkl.h>
  #include <mkl_lapack.h>
  //#pragma message("MKL is used")
#else
  #ifdef ARMPL
    #include <armpl.h>
  #else
    #include <cblas.h>
  #endif
  #ifdef LAPACKE_UTILS
    #include <lapacke_utils.h>
  #endif
  #include <lapacke.h>
  //#pragma message("MKL is NOT used")
#endif

#ifndef hcore_min
#define hcore_min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef hcore_max
#define hcore_max(a, b) ((a) < (b) ? (b) : (a))
#endif



#define HCORE_NoTrans         111
#define HCORE_Trans           112
#define HCORE_ConjTrans       113

#define HCORE_Upper           121
#define HCORE_Lower           122
#define HCORE_UpperLower      123

#define HCORE_NonUnit         131
#define HCORE_Unit            132

#define HCORE_Left            141
#define HCORE_Right           142

typedef int  HCORE_enum;
#endif
