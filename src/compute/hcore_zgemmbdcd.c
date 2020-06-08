/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hcore_zgemmbdcd.c
 *
 *  HCORE routines
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2018-11-08
 * @precisions normal z -> c d s
 **/
#include <assert.h>
#include "hcore.h"
#include <stdio.h>
#include "flop_util_structs.h"
#include "flop_counts.h"

#undef CBLAS_SADDR
#define CBLAS_SADDR(_val) (_val)

int gemmbdcd_print_index = 0;
int gemmbdcd_print_mat = 0;

/*
 * CD=AU*BV'. 
 * Rank of tile AU is in Ark.
 * Rank of tile BV is in Brk.
 * Multiplied tiles must have same rank.
 * CD is dense output.
 * Updates flops->update
*/
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
        flop_counter* flops)
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
    int64_t LD_AVBD = _Ark; //hcore_min(_Ark, LDB);
    int64_t sizeofAVBD = LD_AVBD * N * sizeof(double); 
    
    if(gemmbdcd_print_index){
        printf("Ark:%d M:%d N:%d K:%d  LDA:%d LDB:%d LDC:%d sizeofAVBD=%d transA:%d transB:%d alpha:%g beta:%g\n",
                _Ark, M, N, _Ark, LDA, LDB, LDC, sizeofAVBD, transA, transB, alpha, beta  );
    }
    double one = 1.0, zero = 0.0, minusone = -1.0;
    if(gemmbdcd_print_mat){
        /*hcfrk_printmat(AU,  M, _Ark, LDA); */
        /*hcfrk_printmat(BV,  N, _Brk, LDB); */
    }
    double *AVBD = malloc(sizeofAVBD);
    unsigned long int gemmflop = 0;
    if ( transA == HCORE_Trans) {
        /*printf("AUBD=AU^TxBD m,n,k:%d,%d,%d ld:%d,%d,%d\n", _Ark, N, M, LDA, LDB, LDB);*/
        cblas_dgemm(
                CblasColMajor,
                CblasTrans, CblasNoTrans,
                _Ark, N, M,
                CBLAS_SADDR(one),  AV,  LDA,
                BD,  LDB,
                CBLAS_SADDR(zero), AVBD, LD_AVBD);
        /*printf("CD=AVxAUBD m,n,k:%d,%d,%d ld:%d,%d,%d\n", LDA, N, _Ark, LDA, LDB, LDC);*/
        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                LDA, N, _Ark,
                CBLAS_SADDR(alpha),  AU,  LDA,
                AVBD, LD_AVBD,
                CBLAS_SADDR(beta), CD, LDC);
        gemmflop += flop_counts('m', _Ark, N, M, 0);
        gemmflop += flop_counts('m', LDA, N, _Ark, 0);
    } else {
        /*printf("AVBD=AV^TxBD m,n,k:%d,%d,%d ld:%d,%d,%d\n", _Ark, N, LDB, LDA, LDB, LDB);*/
        cblas_dgemm(
                CblasColMajor,
                CblasTrans, CblasNoTrans,
                _Ark, N, LDB,
                CBLAS_SADDR(one),  AV,  LDA,
                BD,  LDB,
                CBLAS_SADDR(zero), AVBD, LD_AVBD);
        /*printf("CD=AUxAVBD m,n,k:%d,%d,%d ld:%d,%d,%d\n", M, N, _Ark, LDA, LDB, LDC);*/
        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, _Ark,
                CBLAS_SADDR(alpha),  AU,  LDA,
                AVBD, LD_AVBD,
                CBLAS_SADDR(beta), CD, LDC);
        gemmflop += flop_counts('m', _Ark, N, LDB, 0);
        gemmflop += flop_counts('m', M, N, _Ark, 0);
    }
    free(AVBD);
    flops->update += gemmflop;
    if(gemmbdcd_print_mat){
        /*hcfrk_printmat(CD,  M, N, LDC); */
    }
}

