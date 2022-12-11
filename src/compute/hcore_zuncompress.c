/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file hcore_zuncompress.c
 *
 *  HiCMA HCORE routines
 *  HiCMA is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 * @precisions normal z -> c d s
 **/
#include <assert.h>
#include "hcore.h"
#include <stdio.h>

//FIXME PREVIOUS DECLARION OF CBLAS_SADDR ~/hicma/chameleon/build/include/chameleon/coreblas/include/coreblas.h

int gemmfrk_print_index = 0;
int gemmfrk_print_mat = 0;

/*
 * CD=AU*BV'. 
 * Rank of tile AU is in Ark.
 * Rank of tile BV is in Brk.
 * Multiplied tiles must have same rank.
 * CD is dense output.
*/

#undef CBLAS_SADDR
#define CBLAS_SADDR(_val) &(_val)
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
                       )
{
    /*printf("%d|M:%d N:%d K:%d LDA:%d LDB:%d LDC:%d rk:%d acc:%e a:%e b:%e\n",*/
    /*__LINE__, M, N, K, LDA, LDB, LDC, rk, acc, alpha, beta);*/
    
    /*
     * NOTES:
     * assumptions on matrix dimensions are marked as //ASSUMPTION
     * I am currently allocating and freeing temporary buffers.
     * They are marked as //ALLOCATE
     */
    /*printf("%d %d|%g->%d. %g %g, %g %g\n",  */
    /*__LINE__, __COUNTER__,*/
    /*Crk[0], new_Crk, CU[0], CU[1], CV[0], CV[1]);*/
    //hcore_dgemm(Aij, ik, Ajk, -1, rank, acc);
    int64_t _Ark = (int64_t)(Ark[0]);
    int64_t _Brk = (int64_t)(Brk[0]);
    
    if(gemmfrk_print_index){
        printf("Ark:%d Brk:%d M:%d N:%d K:%d  LDA:%d LDB:%d LDC:%d\n",
               _Ark, _Brk, M, N, _Ark, LDA, LDB, LDC  );
    }
    assert(_Ark == _Brk); // ranks must be same
    double _Complex one = (double _Complex)1.0;
    double _Complex zero = (double _Complex)0.0;
    if(gemmfrk_print_mat){
        /*hcfrk_printmat(AU,  M, _Ark, LDA); */
        /*hcfrk_printmat(BV,  N, _Brk, LDB); */
    }
    cblas_zgemm(
                CblasColMajor,
                CblasNoTrans, CblasTrans,
                M, N, _Ark,
                CBLAS_SADDR(one),  AU,  LDA,
                BV,  LDB,
                CBLAS_SADDR(zero), CD, LDC);
    if(gemmfrk_print_mat){
        /*hcfrk_printmat(CD,  M, N, LDC); */
    }
    
}

