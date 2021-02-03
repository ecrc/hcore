/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hcore_dpotrf.c
 *
 *  HCORE routines
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include "hcore.h"
#include "flop_util_structs.h"
#include "flop_counts.h"

/*
 * Performs regular potrf.
 * Updates flops->potrf.
 */
void HCORE_dpotrf(HCORE_enum uplo, 
        int N,
        double *A, int LDA,
        int *info,
        flop_counter* flops)
{
    *info = LAPACKE_dpotrf_work(
        LAPACK_COL_MAJOR,
        uplo,
        N, A, LDA );
    flops->potrf += flop_counts('c', N, 0, 0, 0);
}


