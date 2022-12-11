/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hcore_d.h
 *
 *  HCORE kernels
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2018-11-08
 **/
#ifndef _HCORE_D_H_
#define _HCORE_D_H_
#ifdef __cplusplus
extern "C" {
#endif
#include "flop_util_structs.h"
#include "hcore.h"
/** ****************************************************************************
 *  Declarations of serial kernels - alphabetical order
 **/
void HCORE_dgemm(HCORE_enum transA, int transB,
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
void HCORE_dgemm_fast(HCORE_enum transA, int transB,
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
void HCORE_dgemmcd(
        HCORE_enum transA, //TODO notrans hardcoded, implement all combinations
        HCORE_enum transB, //TODO trans hardcoded, implement all combinations
        int M,
        double alpha, //TODO -1.0 hardcoded, use this parameter
        double *Au,
        double *Av,
        int Arank,
        int LDA,
        double *Bu,
        double *Bv,
        int Brank,
        int LDB,
        double beta, //TODO 1.0 hardcoded, use this parameter
        double *CD,
        int LDC,
        int maxrank,
        double *work,
        flop_counter* flops);
void HCORE_dgemmbdcd(HCORE_enum transA, HCORE_enum transB,
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
void HCORE_dpotrf(HCORE_enum uplo,
        int N,
        double *A, int LDA,
        int *info,
        flop_counter* flops);
void HCORE_dsyrk(HCORE_enum uplo, HCORE_enum trans,
        int N, int K,
        double alpha,
        const double *AU, int LDAU,
        const double *AV, int LDAV,
        double beta,
        double *CD, int LDCD,
        double* work,
        flop_counter* flops
        );
void HCORE_dtrsm(HCORE_enum side,
        HCORE_enum uplo,
        HCORE_enum transA,
        HCORE_enum diag,
        int M, int Brank,
        double alpha,
        double *A, int LDA,
        double *BV, int LDBV,
        flop_counter* flops);
void HCORE_duncompress(HCORE_enum transA, HCORE_enum transB,
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
/**
 * The following functions are used to perform
 * HCORE_dgemm(). These functions perform 
 * first half and second half of HCORE_dgemm()
 * so that matrix sizes can be dynamically adjusted
 * by user after ranks are revealed by SVD.
 */
void HCORE_dgemm_qr_svd_b_dense(int transA, int transB,
        int M, int N,
        double alpha,
        double *AU,
        double *AV,
        int *Ark,
        int LDA,
        double *B,
        int LDB,
        double beta,
        double *CU,
        double *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        double acc,
        double* work,
        flop_counter* flops,
        /** parameters that will be passed to HCORE_dgemm_ormqr */
        double*      *_p_work_new     ,
        double*      *_p__CU          ,
        double*      *_p__CV          ,
        int          *_p_CU_ncols     ,
        int          *_p_new_UVrk     ,
        double*      *_p_newU         ,
        int          *_p_ld_newU      ,
        double*      *_p_qrtauA       ,
        int          *_p_CV_ncols     ,
        double*      *_p_newV         ,
        int          *_p_ld_newV      ,
        double*      *_p_qrtauB       ,
        int          *_p_use_CUV_clone,
        double*      *_p_CUclone      ,
        int          *_p_ld_CUclone   ,
        double*      *_p__CU_save     ,
        double*      *_p_CVclone      ,
        int          *_p_ld_CVclone   ,
        double*      *_p__CV_save
);
void HCORE_dgemm_qr_svd(int transA, int transB,
        int M, int N,
        double alpha,
        double *AU,
        double *AV,
        int *Ark,
        int LDA,
        double *BU,
        double *BV,
        int *Brk,
        int LDB,
        double beta,
        double *CU,
        double *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        double acc,
        double* work,
        flop_counter* flops,
        /** parameters that will be passed to HCORE_dgemm_ormqr */
        double*      *_p_work_new     ,
        double*      *_p__CU          ,
        double*      *_p__CV          ,
        int          *_p_CU_ncols     ,
        int          *_p_new_UVrk     ,
        double*      *_p_newU         ,
        int          *_p_ld_newU      ,
        double*      *_p_qrtauA       ,
        int          *_p_CV_ncols     ,
        double*      *_p_newV         ,
        int          *_p_ld_newV      ,
        double*      *_p_qrtauB       ,
        int          *_p_use_CUV_clone,
        double*      *_p_CUclone      ,
        int          *_p_ld_CUclone   ,
        double*      *_p__CU_save     ,
        double*      *_p_CVclone      ,
        int          *_p_ld_CVclone   ,
        double*      *_p__CV_save
);
void HCORE_dgemm_ormqr(int transA, int transB,
        int M, int N,
        double alpha,
        double *AU,
        double *AV,
        int *Ark,
        int LDA,
        double beta,
        double *CU,
        double *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        double acc,
        double* work,
        flop_counter* flops,
        /** parameters coming from HCORE_dgemm_qr_svd */
        double* _CU,
        double* _CV,
        int CU_ncols,
        int new_UVrk,
        double* newU,
        int ld_newU,
        double* qrtauA,
        int CV_ncols,
        double* newV,
        int ld_newV,
        double* qrtauB,
        int use_CUV_clone,
        double* CUclone,
        int ld_CUclone,
        double* _CU_save,
        double* CVclone,
        int ld_CVclone,
        double* _CV_save
);

void HCORE_dgemm_dense(HCORE_enum transA, int transB,
                       int M, int N, int K,
                       double alpha,
                       const double *A,
                       int LDA,
                       const double *B,
                       int LDB,
                       double beta,
                       double *C,
                       int LDC);

#ifdef __cplusplus
}
#endif
#endif
