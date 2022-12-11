/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file hcore_dgemmcd.c
 *
 *  HCORE routines
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include <assert.h>
#include "hcore.h"
#include <stdio.h>
#include "flop_util_structs.h"
#include "flop_counts.h"

#undef CBLAS_SADDR
#define CBLAS_SADDR(_val) (_val)

int gemmcd_print_index = 0;
int gemmcd_print_mat = 0;
int gemmcd_use_scratch = 0;//TODO use scratch

/*
 * C  = A * -1.0 B^T
 * CD = AU AV^T * -1.0 (BU BV^T)^T. 
 * Rank of tile A is in Ark.
 * Rank of tile B is in Brk.
 * CD is dense output.
 * Updates flops->update
 */
void HCORE_dgemmcd(HCORE_enum transA, HCORE_enum transB, /*TODO implement all combinations*/
        int M,
        double alpha, //TODO use in this function 
        double *Au, 
        double *Av, 
        int Arank, 
        int LDA,
        double *Bu, 
        double *Bv, 
        int Brank, 
        int LDB,
        double beta, //TODO use in this function
        double *CD, 
        int LDC,
        int maxrank,
        double *work,
        flop_counter* flops)
{
    size_t mb_x_maxrank = (size_t)M * maxrank;
    size_t maxrank_x_maxrank = (size_t)maxrank * maxrank;
    double *tmp_mbr;
    double *tmp_rr;
    if(gemmcd_use_scratch == 1) {
        tmp_mbr = work;
        tmp_rr  = work + mb_x_maxrank;
    } else {
        tmp_mbr = (double*) calloc(mb_x_maxrank, sizeof(double)); //  mb x maxrank 
        tmp_rr = (double*) calloc(maxrank_x_maxrank, sizeof(double)); //  maxrank x maxrank 
    }
    /* tmp_rr = trans(Av) * Bv */
    cblas_dgemm(CblasColMajor, HCORE_Trans, HCORE_NoTrans,
            Arank, Brank, M,
            (double) 1.0, Av, M,
            Bv, M,
            (double) 0.0, tmp_rr, Arank);

    if( Arank > Brank ) {
        /* tmp_mbr = Au * tmp_rr */
        cblas_dgemm(CblasColMajor, HCORE_NoTrans, HCORE_NoTrans,
                M, Brank, Arank,
                (double) 1.0, Au, M,
                tmp_rr, Arank,
                (double) 0.0, tmp_mbr, M);

        /* C = C - tmp_mbr * trans(Bu) */
        cblas_dgemm(CblasColMajor, HCORE_NoTrans, HCORE_Trans,
                M, M, Brank,
                (double)-1.0, tmp_mbr, M,
                Bu, M,
                (double) 1.0, CD, M);
    } else {
        /* tmp_mbr = tmp_rr * trans(Bu) */
        cblas_dgemm(CblasColMajor, HCORE_NoTrans, HCORE_Trans,
                Arank, M, Brank,
                (double) 1.0, tmp_rr, Arank,
                Bu, M,
                (double) 0.0, tmp_mbr, Arank);

        /* C = C - Au * tmp_mbr */
        cblas_dgemm(CblasColMajor, HCORE_NoTrans, HCORE_NoTrans,
                M, M, Arank,
                (double)-1.0, Au, M,
                tmp_mbr, Arank,
                (double) 1.0, CD, M);
    }
    if(gemmcd_use_scratch == 1) {
    } else {
        free(tmp_mbr);
        free(tmp_rr);
    }
}

