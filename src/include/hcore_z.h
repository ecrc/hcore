/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hcore_z.h
 *
 *  HiCMA HCORE kernels
 *  HiCMA is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2018-11-08
 **/
#ifndef _HICMA_HCORE_Z_H_
#define _HICMA_HCORE_Z_H_

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
        double _Complex alpha,
        double _Complex *AU,
        double _Complex *AV,
        double *Ark,
        int LDA,
        double _Complex *BU,
        double _Complex *BV,
        double *Brk,
        int LDB,
        double _Complex beta,
        double _Complex *CU,
        double _Complex *CV,
        double *Crk,
        int LDC,
        int rk,
        int maxrk,
        double acc,
        double _Complex* work,
        flop_counter* flops
            );
void HCORE_zgemmbdcd(HCORE_enum transA, HCORE_enum transB,
        int M, int N,
        double _Complex alpha,
        double _Complex *AU,
        double _Complex *AV,
        double *Ark,
        int LDA,
        double _Complex *BD,
        int LDB,
        double _Complex beta,
        double _Complex *CD,
        int LDC,
        double _Complex *work
        );
void HCORE_zgytlr(int m, int n,
        double _Complex *AU,
        double _Complex *AV,
        double _Complex *AD,
        double *Ark,
        int lda,
        int ldu,
        int ldv,
        int bigM, int m0, int n0, unsigned long long int seed,
        int maxrank, double tol,
        int compress_diag,
        double _Complex *Dense
        );
void HCORE_zuncompress(HCORE_enum transA, HCORE_enum transB,
        int M, int N,
        double _Complex alpha,
        double _Complex *AU,
        double *Ark,
        int LDA,
        double _Complex *BV,
        double *Brk,
        int LDB,
        double _Complex beta,
        double _Complex *CD,
        int LDC
        );

#endif
