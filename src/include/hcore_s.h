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
#ifndef _HCORE_S_H_
#define _HCORE_S_H_
#ifdef __cplusplus
extern "C" {
#endif
#include "flop_util_structs.h"
#include "hcore.h"
/** ****************************************************************************
 *  Declarations of serial kernels - alphabetical order
 **/
void HCORE_sgemm(HCORE_enum transA, HCORE_enum transB,
        int M, int N,
        float alpha, 
        float *AU, 
        float *AV, 
        float *Ark, 
        int LDA,
        float *BU,
        float *BV,
        float *Brk,
        int LDB,
        float beta,
        float *CU,
        float *CV,
        float *Crk,
        int LDC,
        int rk,
        int maxrk,
        float acc,
        float* work,
        flop_counter* flops
        );
void HCORE_sgemm_fast(HCORE_enum transA, int transB,
        int M, int N, 
        float alpha,
        float *AU,
        float *AV,
        float *Ark,
        int LDA,
        float *BU,
        float *BV,
        float *Brk,
        int LDB,
        float beta,
        float *CU,
        float *CV,
        float *Crk,
        int LDC,
        int rk,
        int maxrk,
        float acc,
        float* work
        );
void HCORE_sgemmcd(
        HCORE_enum transA, //TODO notrans hardcoded, implement all combinations
        HCORE_enum transB, //TODO trans hardcoded, implement all combinations
        int M,
        float alpha, //TODO -1.0 hardcoded, use this parameter
        float *Au, 
        float *Av, 
        int Arank, 
        int LDA,
        float *Bu, 
        float *Bv, 
        int Brank, 
        int LDB,
        float beta, //TODO 1.0 hardcoded, use this parameter
        float *CD, 
        int LDC,
        int maxrank,
        float *work,
        flop_counter* flops);
void HCORE_sgemmbdcd(HCORE_enum transA, HCORE_enum transB,
        int M, int N,
        float alpha, 
        float *AU, 
        float *AV, 
        float *Ark, 
        int LDA,
        float *BD, 
        int LDB,
        float beta, 
        float *CD, 
        int LDC,
        float *work, 
        flop_counter* flops
        );
void HCORE_spotrf(HCORE_enum uplo, 
        int N,
        float *A, int LDA,
        int *info,
        flop_counter* flops);
void HCORE_ssyrk(HCORE_enum uplo, HCORE_enum trans,
        int N, int K,
        float alpha,
        const float *AU, int LDAU,
        const float *AV, int LDAV,
        float beta,
        float *CD, int LDCD,
        float* work,
        flop_counter* flops
        );
void HCORE_strsm(HCORE_enum side, 
        HCORE_enum uplo,
        HCORE_enum transA,
        HCORE_enum diag,
        int M, int Brank,
        float alpha,
        float *A, int LDA,
        float *BV, int LDBV,
        flop_counter* flops);
void HCORE_suncompress(HCORE_enum transA, HCORE_enum transB,
        int M, int N,
        float alpha, 
        float *AU, 
        float *Ark, 
        int LDA,
        float *BV,
        float *Brk,
        int LDB,
        float beta,
        float *CD,
        int LDC
        );
/** 
 * The following functions are used to perform
 * HCORE_sgemm(). These functions perform 
 * first half and second half of HCORE_sgemm()
 * so that matrix sizes can be dynamically adjusted
 * by user after ranks are revealed by SVD.
 */
void HCORE_sgemm_qr_svd_b_dense(int transA, int transB,
        int M, int N,
        float alpha,
        float *AU,
        float *AV,
        int *Ark,
        int LDA,
        float *B,
        int LDB,
        float beta,
        float *CU,
        float *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        float acc,
        float* work,
        flop_counter* flops,
        /** parameters that will be passed to HCORE_sgemm_ormqr */
        float*      *_p_work_new     ,
        float*      *_p__CU          ,
        float*      *_p__CV          ,
        int          *_p_CU_ncols     ,
        int          *_p_new_UVrk     ,
        float*      *_p_newU         ,
        int          *_p_ld_newU      ,
        float*      *_p_qrtauA       ,
        int          *_p_CV_ncols     ,
        float*      *_p_newV         ,
        int          *_p_ld_newV      ,
        float*      *_p_qrtauB       ,
        int          *_p_use_CUV_clone,
        float*      *_p_CUclone      ,
        int          *_p_ld_CUclone   ,
        float*      *_p__CU_save     ,
        float*      *_p_CVclone      ,
        int          *_p_ld_CVclone   ,
        float*      *_p__CV_save     
);
void HCORE_sgemm_qr_svd(int transA, int transB,
        int M, int N,
        float alpha,
        float *AU,
        float *AV,
        int *Ark,
        int LDA,
        float *BU,
        float *BV,
        int *Brk,
        int LDB,
        float beta,
        float *CU,
        float *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        float acc,
        float* work,
        flop_counter* flops,
        /** parameters that will be passed to HCORE_sgemm_ormqr */
        float*      *_p_work_new     ,
        float*      *_p__CU          ,
        float*      *_p__CV          ,
        int          *_p_CU_ncols     ,
        int          *_p_new_UVrk     ,
        float*      *_p_newU         ,
        int          *_p_ld_newU      ,
        float*      *_p_qrtauA       ,
        int          *_p_CV_ncols     ,
        float*      *_p_newV         ,
        int          *_p_ld_newV      ,
        float*      *_p_qrtauB       ,
        int          *_p_use_CUV_clone,
        float*      *_p_CUclone      ,
        int          *_p_ld_CUclone   ,
        float*      *_p__CU_save     ,
        float*      *_p_CVclone      ,
        int          *_p_ld_CVclone   ,
        float*      *_p__CV_save     
);
void HCORE_sgemm_ormqr(int transA, int transB,
        int M, int N,
        float alpha,
        float *AU,
        float *AV,
        int *Ark,
        int LDA,
        float beta,
        float *CU,
        float *CV,
        int *Crk,
        int LDC,
        int rk,
        int storage_maxrank,
        int maxrk, /*this is compmaxrank*/
        float acc,
        float* work,
        flop_counter* flops,
        /** parameters coming from HCORE_sgemm_qr_svd */
        float* _CU,
        float* _CV,
        int CU_ncols,
        int new_UVrk,
        float* newU,
        int ld_newU,
        float* qrtauA,
        int CV_ncols,
        float* newV,
        int ld_newV,
        float* qrtauB,
        int use_CUV_clone,
        float* CUclone,
        int ld_CUclone,
        float* _CU_save,
        float* CVclone,
        int ld_CVclone,
        float* _CV_save
);

void HCORE_sgemm_dense(HCORE_enum transA, int transB,
                       int M, int N, int K,
                       float alpha,
                       const float *A,
                       int LDA,
                       const float *B,
                       int LDB,
                       float beta,
                       float *C,
                       int LDC);
#ifdef __cplusplus
}
#endif
#endif
