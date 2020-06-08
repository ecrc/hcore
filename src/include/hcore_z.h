/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hcore_z.h
 *
 *  HCORE kernels
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2018-11-08
 **/
#ifndef _HCORE_Z_H_
#define _HCORE_Z_H_

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
#include "flop_util_structs.h"



#include "hcore.h"
    /** ****************************************************************************
     *  Declarations of serial kernels - alphabetical order
     **/
    void HCORE_zgemm(HCORE_enum transA, int transB,
            int M, int N,
            double alpha, 
            double *AU, 
            double *AV, 
            double *Ark, 
            int LDA,
            double *BU,
            double *BV,
            double *Brk,
            int LDB,
            double beta,
            double *CU,
            double *CV,
            double *Crk,
            int LDC,
            int rk,
            int maxrk,
            double acc,
            double* work,
            flop_counter* flops
                );
    void HCORE_zgemm_fast(HCORE_enum transA, int transB,
            int M, int N, 
            double alpha,
            double *AU,
            double *AV,
            double *Ark,
            int LDA,
            double *BU,
            double *BV,
            double *Brk,
            int LDB,
            double beta,
            double *CU,
            double *CV,
            double *Crk,
            int LDC,
            int rk,
            int maxrk,
            double acc,
            double* work
            );
    void HCORE_zgemmbdcd(HCORE_enum transA, HCORE_enum transB,
            int M, int N,
            double alpha, 
            double *AU, 
            double *AV, 
            double *Ark, 
            int LDA,
            double *BD, 
            int LDB,
            double beta, 
            double *CD, 
            int LDC,
            double *work, 
            flop_counter* flops
            );
    void HCORE_zsyrk(HCORE_enum uplo, HCORE_enum trans,
            int N, int K,
            double alpha,
            const double *AU, int LDAU,
            const double *AV, int LDAV,
            double beta,
            double *CD, int LDCD,
            double* work,
            flop_counter* flops
            );
    void HCORE_zuncompress(HCORE_enum transA, HCORE_enum transB,
            int M, int N,
            double alpha, 
            double *AU, 
            double *Ark, 
            int LDA,
            double *BV,
            double *Brk,
            int LDB,
            double beta,
            double *CD,
            int LDC
            );
#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
