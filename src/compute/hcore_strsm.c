/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file hcore_dtrsm.c
 *
 *  HCORE routines
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include "hcore.h"
#include <assert.h>
#include "flop_util_structs.h"
#include "flop_counts.h"

/*
 * Performs regular trsm.
 * Updates flops->trsm.
 */
void HCORE_dtrsm(HCORE_enum side, 
        HCORE_enum uplo,
        HCORE_enum transA,
        HCORE_enum diag,
        int M, int Brank,
        float alpha,
        float *A, int LDA,
        float *BV, int LDBV,
        flop_counter* flops)
{
    cblas_strsm(
        CblasColMajor,
        side, uplo,
        transA, diag,
        M,
        Brank,
        alpha, A, LDA,
        BV, LDBV);
    if(side == CblasLeft)
        flops->trsm += flop_counts('t', M, Brank, 1, 0);
    else if(side == CblasRight)
        flops->trsm += flop_counts('t', M, Brank, 2, 0);
    else
        assert(0=="side is not CblasLeft or CblasRight");
}


