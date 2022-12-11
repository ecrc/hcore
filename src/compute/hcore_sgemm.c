/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file hcore_sgemm.c
 *
 * This file contains HCORE_sgemm() kernel and its internal functions.
 * The main operation is C=alpha A B + beta C.
 * A, B and C are tile-low rank (TLR) matrices input to HCORE_sgemm() and HCORE_sgemm_qr_svd(). 
 * A and C are TLR matrices and B is dense matrix input to HCORE_sgemm HCORE_sgemm_qr_svd_b_dense()
 * and HCORE_sgemm_ormqr(). 
 * A low rank tile T is stored as TU|TV and T can be obtained via performing TU*TV^T.
 *
 * HCORE_sgemm() is split into HCORE_sgemm_qr_svd()/HCORE_sgemm_qr_svd_b_dense() and HCORE_sgemm_ormqr()
 * to leave the dynamic allocation of C to user after new rank is revealed by SVD.
 *
 * HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include <assert.h>
#ifdef LAPACKE_UTILS
#include <lapacke_utils.h>
#endif
#include "hcore.h"
#include <stdio.h>

#undef CBLAS_SADDR
#define CBLAS_SADDR(_val) (_val)

#include "flop_util_structs.h"
#include "flop_counts.h"

int sgemm_use_trmm = 0; /** Usage of TRMM instead of GEMM in HCORE_sgemm() */
extern int use_scratch; /** Determines how buffers are allocated 
                            1: use work array given by caller, 
                            0: allocate in the function */
int sgemm_print_index = 0; /** Verbose mode. Print tile indices only in HCORE_sgemm() */
int sgemm_print_mat = 0;   /** Verbose mode. Print tile contents only in HCORE_sgemm() */
int shc_nelm_limit = 512;  /** Maximum number of elements of a tile to print */
/**
 * Print a tile
 */
void shc_printmat(float * A, int m, int n, int ld){
    printf("M:%d N:%d LD:%d\n[", m, n, ld);
    int i, j, nelm = 0;
    for(i=0;i<m;i++){
        printf("[");
        for(j=0;j<n;j++){
            printf("%+.2e", A[j*ld+i]);
            if(j!=n-1){
                //printf(", ");
            }
            //printf("%g ", A[j*tld(descZ)+i]);
            //printf("%g\t", A[j*descZ->n+i]);
            //printf("(%d,%d,%d) %g\t", i,j,descZ->mb,A[j*descZ->mb+i]);
            //printf("(%d,%d,%d) %g\t", i,j,descZ->n,A[j*descZ->n+i]);
            //printf("(%d,%d,%d) %g [%d %d %d]\t", i,j,descZ->n,A[j*descZ->n+i], descZ->m, descZ->lm, descZ->ln);
            nelm++;
            if(nelm >= shc_nelm_limit){
                printf("\n");
                return;
            }
        }
        printf("]");
        if(i!=m-1){
            printf(",");
            printf("\n");
        }
    }
    printf("]\n");
}
/**
 * Performs QA*RA=QR(CU|AU)
 */
void __sqra(int _M,
        int maxrank,
        float* _CU, int ld_CU, int _Crk,
        int* pnew_CU_ncols,
        float* _AU, int ld_AU, int _Ark,
        float alpha, float beta, float* qrtauA,
        flop_counter* flops)
{
    int info;
    int AU_nrows = _M;// assume that _M == nrows of A
    int AU_ncols = _Ark;
    int CU_ncols = _Crk;
    if ((_Ark + _Crk) > 2*maxrank){
        fprintf(stderr, "%s %s %d: Sum of ranks (%d) is too big! _Ark:%d _Crk:%d maxrank:%d (x2: %d)\n",
                __FILE__, __func__, __LINE__, (_Ark + _Crk), _Ark, _Crk, maxrank, 2*maxrank);
        exit(-1);
    }
    int nelm_AU = AU_nrows * AU_ncols;
    int incOne = 1;
    //hc_printmat(_CU, _M, _M, ld_CU);
    //hc_printmat(_CU, _M, _M, ld_CU);
    //ERRONEOUS. here is no ld!!!
    if(sgemm_print_index){
        printf("%s %s %d: Sum of ranks:%d _Ark:%d _Crk:%d maxrank:%d (x2: %d)\n",
                __FILE__, __func__, __LINE__, (_Ark + _Crk), _Ark, _Crk, maxrank, 2*maxrank);
        printf(" QRA\t|%d\t|nelm_AU:%d alpha:%g CU_ncols:%d ld_CU:%d CU_ncols*ld_CU:%d\n", __LINE__, nelm_AU, alpha, CU_ncols, ld_CU, CU_ncols*ld_CU);
    }
    cblas_scopy(nelm_AU, _AU, incOne,  &_CU[CU_ncols*ld_CU], incOne);
    //shc_printmat(_CU, _M, _M, ld_CU);
    float d_one = (float)1.0;
    if(alpha != d_one){
        cblas_sscal(nelm_AU, CBLAS_SADDR(alpha), &_CU[CU_ncols*ld_CU], incOne);
    }
    if(beta != d_one){
        cblas_sscal(_M * _Crk, CBLAS_SADDR(beta), _CU, incOne);
    }
    //shc_printmat(_CU, _M, _M, ld_CU);
    int CU_nrows = _M;
    /*printf("__qra: CU_ncols:%d new_CU_ncols:%d\n", CU_ncols, (CU_ncols+_Ark));*/
    CU_ncols += _Ark;
    *pnew_CU_ncols = CU_ncols; //CHANGED VALUE, RETURN VALUE
    if(sgemm_print_index){
        printf(" QRA\t|%d\t|CU_nrows:%d CU_ncols:%d ld_CU:%d  QRA:%p\n", __LINE__, CU_nrows, CU_ncols, ld_CU, _CU);
    }
    info = LAPACKE_sgeqrf(
            LAPACK_COL_MAJOR, CU_nrows, CU_ncols, _CU, ld_CU, qrtauA);
    unsigned long int qrflop = flop_counts('q', CU_nrows, CU_ncols, 0, 0);
    flops->update += qrflop;// + multMinusOne;
    //printf("%d %d %d _M:%d\n", CU_nrows, CU_ncols, ld_CU, _M);
    //hc_printmat(_CU, _M, _M, ld_CU);
    /*hc_printmat(qrtauA, 1, _M, 1); */
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_sgeqrf(1:CU_nrows:%d 2:CU_ncols:%d 3:_CU:%p 4:ld_CU:%d 5:qrtauA:%p) info=%d maxrank:%d\n",
                __FILE__, __LINE__, CU_nrows, CU_ncols, _CU, ld_CU, qrtauA, info, maxrank);
        exit(-1);
    }
    //hc_printmat(_AU, _M, _M, ld_AU);
}

/*
 * Performs QR([CV|(AV^T*(BU*BV^T)^T)]).
 * CV, AV and BV are stored as transposed.
 * Step 1: F=AV^T*BV        
 * Step 2: G=F*BU^T=BU*F^T  
 * Step 3: CV=CV|G   (done in the gemm at Step 2)
 * Step 4: QB*RB=QR(CV)    
 *
 * Note that G is stored as transposed.
 */
void __sqrb(
        int _M,
        int maxrank,
        float* _CV, int ld_CV, int _Crk,
        int* pnew_CV_ncols,
        float* _AV, int ld_AV, int _Ark,
        float* _BU, int ld_BU,
        float* _BV, int ld_BV, int _Brk,
        float* qrtauB, float* AcolBcolT,
        flop_counter* flops)
{
    int info;
    /// B2 = ( (iA{2} * iB{2}') * iB{1}');
    assert(AcolBcolT != NULL);
    int ld_AcolBcolT = maxrank; //ASSUMPTION

    float alpha = 1.0;
    float beta = 0.0;
    int AV_ncols = _Ark; //ASSUMPTION
    int BV_nrows = _M;   //ASSUMPTION
    int BV_ncols = _Brk; //ASSUMPTION
    // AV*BV^T
    if(sgemm_print_index){
        printf(" QRB\t|%d\t| (AV*BV^T)       M,N,K:%d,%d,%d  LDA,LDB,LDC:%d,%d,%d alpha:%g beta:%g\n", __LINE__,AV_ncols, BV_ncols, BV_nrows,ld_AV,ld_BV,ld_AcolBcolT, alpha, beta);
    }
    /* Step 1: F=AV^T*BU gemm */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            AV_ncols, BV_ncols, BV_nrows, CBLAS_SADDR(alpha),
            _AV, ld_AV, _BV, ld_BV, CBLAS_SADDR(beta),
            AcolBcolT, ld_AcolBcolT);
    flops->update += flop_counts('m', AV_ncols, BV_ncols, BV_nrows,0);
    int AcolBcolT_nrows = AV_ncols;
    int AcolBcolT_ncols = BV_ncols;

    int BU_nrows = _M;     //ASSUMPTION
    int BU_ncols = _Brk;  //ASSUMPTION
    int CV_ncols = _Crk;
    if ((AcolBcolT_nrows + _Crk) > 2*maxrank){
        fprintf(stderr, "%s %s %d: Sum of two ranks (%d) is too big! \
                AcolBcolT:%d _Crk:%d maxrank:%d (x2: %d)\n",
                __FILE__, __func__, __LINE__, (AcolBcolT_nrows + _Crk), AcolBcolT, _Crk, maxrank, 2*maxrank);
        exit(-1);
    }
    // (AV*BV^T) * BU^T
    if(sgemm_print_index){
        printf(" QRB\t|%d\t| (AV*BV^T) * BU^T  M,N,K:%d,%d,%d  LDA,LDB,LDC:%d,%d,%d alpha:%g beta:%g   CV_ncols:%d AcolBcolT_nrows:%d ldcB:%d\n",
                __LINE__, BU_nrows,AcolBcolT_nrows, BU_ncols,ld_BU,ld_AcolBcolT,ld_CV, alpha, beta, CV_ncols, AcolBcolT_nrows, ld_CV);
    }
    /* Step 2: G=P*BV^T  gemm */
    /* Step 3: CV=CV|G   done in the gemm at Step 2*/
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            BU_nrows, AcolBcolT_nrows, BU_ncols,
            CBLAS_SADDR(alpha),
            _BU, ld_BU,
            AcolBcolT, ld_AcolBcolT,
            CBLAS_SADDR(beta),
            &_CV[CV_ncols*ld_CV], ld_CV);
    CV_ncols        += AcolBcolT_nrows;
    *pnew_CV_ncols   = CV_ncols;       //CHANGED VALUE, RETURN VALUE
    flops->update += flop_counts('m', BU_nrows, AcolBcolT_nrows, BU_ncols,0);  
    int CV_nrows = _M;             //ASSUMPTION

    if(sgemm_print_index){
        printf(" QRB\t|%d\t|CV_nrows:%d CV_ncols:%d ld_CV:%d QRB:%p\n",
                __LINE__, CV_nrows, CV_ncols, ld_CV, _CV);
    }
    /* Step 4: QR(CV)    potrf */
    info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, CV_nrows, CV_ncols,
            _CV, ld_CV, qrtauB);
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dgeqrf(1:CV_nrows:%d 2:CV_ncols:%d 3:_CV:%p 4:ld_CV:%d 5:qrtauB:%p) info=%d maxrank:%d\n",
                __FILE__, __LINE__, CV_nrows, CV_ncols, _CV, ld_CV, qrtauB,info,  maxrank);
        exit(-1);
    }
    flops->update += flop_counts('q', CV_nrows, CV_ncols, 0, 0);  
}

/*
 * Performs QR([CV|(AV^T*B^T)]).
 * CV and AV are stored as transposed.
 * Step 1: G=AV^T*B^T=B*AV 
 * Step 2: CV=CV|G   (done in the gemm at Step 1)
 * Step 3: QB*RB=QR(CV)   
 *
 * Note that G is stored as transposed.
 */
void __sqrb_b_dense(
        int _M,
        int maxrank,
        float* _CV, int ld_CV, int _Crk,
        int* pnew_CV_ncols,
        float* _AV, int ld_AV, int _Ark,
        float* _B,  int ld_B,
        float* qrtauB,
        flop_counter* flops ){
    int info;
    int CV_ncols = _Crk;

    float alpha = 1.0;
    float beta = 0.0;
    int AV_ncols = _Ark; //ASSUMPTION
    // (AV*BV^T) * BU^T
    if(sgemm_print_index){
        printf(" QRB\t|%d\t| AV*B^T  M,N,K:%d,%d,%d  LDA,LDB,LDC:%d,%d,%d alpha:%g beta:%g   CV_ncols:%d ldcB:%d\n",
                __LINE__, _Ark, _M, _M, ld_AV, ld_B, ld_CV, alpha, beta, CV_ncols, ld_CV);
    }
    /* Step 1: G=AV^T*B^T=B*AV */
    /* Step 2: CV=CV | G   done in the gemm at Step 2*/

    /* trans(AV) * trans(B) */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            _M, 
            _Ark, 
            _M,
            (alpha),
            _B, ld_B,
            _AV, ld_AV,
            (beta),
            &_CV[CV_ncols*ld_CV], ld_CV);
    flops->update += flop_counts('m', _M, _Ark, _M, 0);
    CV_ncols        += _Ark;
    *pnew_CV_ncols   = CV_ncols;       //CHANGED VALUE, RETURN VALUE
    int CV_nrows = _M;             //ASSUMPTION

    if(sgemm_print_index){
        printf(" QRB\t|%d\t|CV_nrows:%d CV_ncols:%d ld_CV:%d QRB:%p\n",
                __LINE__, CV_nrows, CV_ncols, ld_CV, _CV);
    }
    /* Step 3: QR(CV) */
    info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, CV_nrows, CV_ncols,
            _CV, ld_CV, qrtauB);
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dgeqrf(1:CV_nrows:%d 2:CV_ncols:%d 3:_CV:%p 4:ld_CV:%d 5:qrtauB:%p) info=%d maxrank:%d\n",
                __FILE__, __LINE__, CV_nrows, CV_ncols, _CV, ld_CV, qrtauB,info,  maxrank);
        exit(-1);
    }
    flops->update += flop_counts('q', CV_nrows, CV_ncols, 0, 0);  
}

/**
 * Performs U*(S*V)=SVD( RA RB^T ) to reveal new rank of C.
 * Returns U and (S*V) as V.
 *
 * Uses CV for TRMM.
 */
void __ssvd(
        int _M,
        int maxrank,
        float* _CU, int ld_CU,
        float* _CV, int ld_CV, int _Crk,
        float* _U, int ld_U,
        float* _V, int ld_V, int* pnew_UVrk,
        int rank,
        float acc,
        float* _rA, float* _rB, float* _T, float* sigma, float* svdsuperb,
        flop_counter* flops) 
{
    int info;
    int nb        = _M;   //ASSUMPTION
    int CU_nrows  = _M;   //ASSUMPTION
    int CU_ncols  = _Crk; //ASSUMPTION
    /// rA     START
    int rA_nrows  = hcore_min(CU_nrows, CU_ncols);
    int rA_ncols  = CU_ncols;    //ASSUMPTION
    int maxncolsR = 2*_Crk; //nb;          //ASSUMPTION
    int ld_rA     = rA_nrows;

    if(rA_nrows != rA_ncols && sgemm_use_trmm == 1){
        printf("TRMM cannot be used because R coming from QR factorization of A is not square nrows: %d ncols:%d \n", rA_nrows, rA_ncols);
        exit(1);
    }
    assert(_rA != NULL);

    if(sgemm_print_index){
        printf(" SVD\t|%d\t| copy rA rA_nrows:%d rA_ncols:%d ld_CU:%d ld_rA:%d CU:%p rA:%p\n",
              __LINE__, rA_nrows, rA_ncols, ld_CU, ld_rA, _CU, _rA);
    }
    float zero = 0.0;
    char chlow = 'L';
    slaset_(&chlow,
            &rA_nrows, &rA_ncols, &zero, &zero, _rA, &ld_rA);
    char chup = 'U';
    slacpy_(&chup,
            &rA_nrows, &rA_ncols,
            _CU, &ld_CU,
            _rA, &ld_rA);
    if(sgemm_print_mat){
        printf("%d\t|_CU and _rA\n", __LINE__);
        shc_printmat(_CU,  _M, _Crk, ld_CU);
        shc_printmat(_rA,  rA_nrows, rA_ncols, ld_rA);
    }
    /// rA     END
    /// rB    START
    int CV_nrows = _M;         //ASSUMPTION
    int CV_ncols = _Crk;       //ASSUMPTION
    int rB_nrows = hcore_min(CV_nrows, CV_ncols);
    int rB_ncols = CV_ncols;
    int ld_rB    = rB_nrows;     //ASSUMPTION
    assert(rA_ncols == rB_ncols);
    // trmm does not access lower part but gemm reads.
    // However lower part is never written.
    if(sgemm_print_index){
        printf(" SVD\t|%d\t| copy rB rB_nrows:%d rB_ncols:%d ld_rB:%d ld_CV:%d rB:%p CV:%p\n",
              __LINE__, rB_nrows, rB_ncols, ld_rB, ld_CV, _rB, _CV);
    }
    if(sgemm_use_trmm == 0){
        assert(_rB != NULL);
        slaset_(&chlow,
                &rB_nrows, &rB_ncols, &zero, &zero, _rB, &ld_rB);
        slacpy_(&chup,
                &rB_nrows, &rB_ncols, _CV, &ld_CV, _rB, &ld_rB);
    } else {
        _rB = _CV;
        ld_rB = ld_CV;
    }
    if(sgemm_print_mat){
        printf("%d\t|_CV and _rB\n", __LINE__);
        shc_printmat(_CV,  _M, _Crk, ld_CV);
        shc_printmat(_rB,  rB_nrows, rB_ncols, ld_rB);
    }
    /// rB    END
    /// SVD   START
    int T_nrows = rA_ncols;
    int T_ncols = rA_ncols;
    int ld_T    = T_nrows;
    float alpha = 1.0;
    float beta  = 0.0;
    if(sgemm_use_trmm == 1){
        cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, rA_nrows, rA_ncols,  alpha, _rB, ld_rB, _rA, ld_rA); //FIXME Correctness of rA_ncols is not checked
        _T = _rA;
        flops->update += flop_counts('r', rA_nrows, rA_ncols, 2 /** right */, 0);  
    } else {
        assert(_T != NULL);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                rA_nrows, rB_nrows, rB_ncols, CBLAS_SADDR(alpha), _rA, ld_rA,
                _rB, ld_rB, CBLAS_SADDR(beta), _T, ld_T);
    }
    if(sgemm_print_index){
        printf(" SVD\t|%d\t| T=rA*rB^T  rA_nrows:%d rB_nrows:%d rB_ncols:%d ld_rA:%d ld_rB:%d ld_T:%d alpha:%g beta:%g\n",
              __LINE__, rA_nrows, rB_nrows, rB_ncols, ld_rA, ld_rB, ld_T, alpha, beta);
    }
    if(sgemm_print_mat){
        shc_printmat(_T,  T_nrows, T_ncols, ld_T);
    }
    // Singular values are ALWAYS floating point numbers.
    assert(sigma != NULL);
    int size_sigma = T_nrows;
    assert(svdsuperb != NULL);
    if(sgemm_print_index){
        printf(" SVD\t|%d\t| svd(T)    (3.m)T_nrows:%d (4.n)T_ncols:%d ld_T:%d ld_U:%d (11.ldvt)ld_V:%d _T:%p (zero based parameter indices)\n",
              __LINE__, T_nrows, T_ncols,  ld_T, ld_U, ld_V, _T);
    }
    info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'A', 'A',
            T_nrows, T_ncols, _T, ld_T, sigma,
            _U, ld_U, _V, ld_V,
            svdsuperb);
    if(info != 0){
        /**
         * Parameter index numbering: http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html#ga84fdf22a62b12ff364621e4713ce02f2
  286       ELSE IF( m.LT.0 ) THEN
  287          info = -3
  288       ELSE IF( n.LT.0 ) THEN
  289          info = -4
  290       ELSE IF( lda.LT.max( 1, m ) ) THEN
  291          info = -6
  292       ELSE IF( ldu.LT.1 .OR. ( wntuas .AND. ldu.LT.m ) ) THEN
  293          info = -9
  294       ELSE IF( ldvt.LT.1 .OR. ( wntva .AND. ldvt.LT.n ) .OR.
  295      $         ( wntvs .AND. ldvt.LT.minmn ) ) THEN
  296          info = -11
         */
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dgesvd() info=%d"
                "3:T_nrows=%d, 4:T_ncols=%d, 5:_T=%p, 6:ld_T=:%d, 7:sigma=%p,"
                "8:_U=%p, 9:ld_U=%d, 10:_V=%p, 11:ld_V:%d,"
                "12:svdsuperb:%p"
                "\n",
                __FILE__, __LINE__, info,
                T_nrows, T_ncols, _T, ld_T, sigma,
                _U, ld_U, _V, ld_V,
                svdsuperb);
        exit(-1);
    }
    flops->update += flop_counts('s', T_nrows, 0,0,0);
    int U_nrows, U_ncols, V_nrows, V_ncols;
    U_nrows = U_ncols = V_nrows = V_ncols = T_nrows;

    if(sgemm_print_mat){
        shc_printmat(_U,  U_nrows, U_ncols, ld_U);
        shc_printmat(sigma,  1, size_sigma, 1);
        shc_printmat(_V,  V_nrows, V_ncols, ld_V);
        printf("%d %e\n", rank, acc);
    }
    int finalrank = -1;
    //double relacc = (acc*sigma[0]*acc);
    float relacc = (acc);
    //double relacc = (acc*acc);
    if(rank != 0) { /// truncate according to rank
        finalrank = rank;
        if(rank > size_sigma)
            finalrank = size_sigma;
    }
    else{ /// truncate according to acc
        int newrank = size_sigma;
        int i;
        for(i=1;i<size_sigma;i++){
            if(sigma[i] < relacc)
            {
                newrank=i;
                break;
            }
        }
        finalrank = newrank;
    }
    if(finalrank > maxrank){
        fprintf(stderr, "%s %s %d: Rank after truncation is too big! finalrank:%d maxrank:%d\n", __FILE__, __func__, __LINE__, finalrank, maxrank);
        exit(-1);
    }

    if(sgemm_print_index){int i;
        printf("rank:%d acc:%.2e   relac:%.2e size_sigma:%d final_rank:%d:  ",
                rank,   acc, relacc, size_sigma, finalrank);
        for(i=0;i<size_sigma;i++){
            printf("%d:%.2e ", i,sigma[i]);
        }
        printf("\n");
    }
    if(sgemm_print_index){
        printf("size_sigma:%d finalrank:%d %.2e\n", size_sigma, finalrank, acc);
    }
    U_ncols = finalrank;
    V_nrows = finalrank;
    /// SVD    END
    int rank_V = finalrank;
    int k;
    for(k = 0; k < rank_V; k++){
        float diagval = sigma[k];
        cblas_sscal(V_ncols, CBLAS_SADDR(diagval), &_V[k], ld_V);
    }
    {
        unsigned long ulVncols = V_ncols;
        unsigned long nmult = rank_V * ulVncols;
        flops->update += nmult;  
    }
    if(sgemm_print_index){
        printf(" SVD\t|%d\t| S*V     V_ncols:%d ld_V:%d _V:%p\n",
              __LINE__, V_ncols,  ld_V, _V);
    }
    if(sgemm_print_mat){
        shc_printmat(_V,  V_nrows, V_ncols, ld_V);
    }
    *pnew_UVrk = finalrank;
}

/**
 * Performs CU=QA*U using dormqr
 */
void __snewu(
        int _M,
        int ncols_qA,
        float* _CU, int ld_CU, int _Crk,
        float* _U,  int ld_U,  int _Urk,
        float* qrtauA,
        flop_counter* flops) 
{
    int info = 0;
    int CU_nrows = _M;
    int CU_ncols = ncols_qA;
    int U_nrows = ncols_qA; //ASSUMPTION
    int U_ncols = _Urk; //ASSUMPTION
    int nrows = CU_nrows - U_nrows;
    float zero  = 0.0;
    if(sgemm_print_index){
        printf(" NEWU\t|%d\t|    zero     nrows:%d U_ncols:%d ld_U:%d _Crk:%d CU_ncols:%d _Urk:%d   CU_nrows:%d U_nrows:%d diff:%d\n",
              __LINE__, nrows, U_ncols,  ld_U, _Crk, CU_ncols,  _Urk, CU_nrows, U_nrows, nrows);
    }
    //info = LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', nrows, U_ncols,
    //        zero, zero, &_U[U_nrows], ld_U);

    char uplo = 'A';
    slaset_( &uplo, &nrows, &U_ncols, &zero, &zero, &_U[U_nrows], &ld_U );
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dlaset() info=%d\n",
                __FILE__, __LINE__, info);
    }
    if(sgemm_print_mat){
        shc_printmat(_U,  _M, _M, ld_U);
    }

    //zunmqr
    int min_MK = hcore_min(CU_nrows, ncols_qA); //number of reflectors, number of rows of qA
    info = LAPACKE_sormqr(LAPACK_COL_MAJOR, 'L', 'N',
            CU_nrows, U_ncols, min_MK, _CU, ld_CU, qrtauA, _U, ld_U);
    if(sgemm_print_index){
        printf(" NEWU\t|%d\t|    ormqr     CU_nrows (new U_nrows):%d U_ncols:%d min_MK:%d ld_CU:%d ld_U:%d\n",
              __LINE__, CU_nrows, U_ncols, min_MK, ld_CU, ld_U);
    }
    if(sgemm_print_mat){
        shc_printmat(_U,  _M, _M, ld_U);
    }
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dormqr() info=%d\n",
                __FILE__, __LINE__, info);
        printf(" NEWU\t|%d\t|    ormqr     3:CU_nrows (new U_nrows):%d 4:U_ncols:%d 5:min_MK:%d 7:ld_CU:%d 10:ld_U:%d U_nrows:%d\n",
              __LINE__, CU_nrows, U_ncols, min_MK, ld_CU, ld_U, U_nrows);
        if(0){
            int i, j, ssend, ldarr;
            float *arr;
            arr = _CU; ldarr = ld_CU; ssend = ncols_qA;
            for(i=0;i<50;i++){
                for(j=0;j<3;j++){
                    printf("%.3e ", arr[j*ldarr+i]);
                }
                printf("...");
                for(j=ssend-4;j<ssend;j++){
                    printf("%.3e ", arr[j*ldarr+i]);
                }
                printf("\n");
            }
            printf("\n");
            arr = _U; ldarr = ld_U; ssend = U_ncols;
            for(i=0;i<50;i++){
                for(j=0;j<3;j++){
                    printf("%.3e ", arr[j*ldarr+i]);
                }
                printf("...");
                for(j=ssend-4;j<ssend;j++){
                    printf("%.3e ", arr[j*ldarr+i]);
                }
                printf("\n");
            }
            for(j=0;j<U_ncols;j++){
                for(i=0; i < CU_nrows; i++){
                    float val = _U[j*ld_U+i];
                    if(val != val){
                        printf("%d,%d is nan (%g) CU_nrows:%d U_ncols:%d ld_U:%d\n", i, j, val,  CU_nrows, U_ncols, ld_U);
                    }
                }
            }
        }
        exit(-1);
    }
    flops->update += flop_counts('o', CU_nrows, U_ncols, min_MK, 1);  
    U_nrows = CU_nrows;
    LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', U_nrows, U_ncols,
            _U, ld_U, _CU, ld_CU);
    if(sgemm_print_index){
        printf(" NEWU\t|%d\t|    copy     U_nrows:%d U_ncols:%d ld_CU:%d ld_U:%d\n",
                __LINE__, U_nrows, U_ncols, ld_CU, ld_U);
        if(0){
            int i, j;
            for(i=0;i<100;i++){
                for(j=0;j<3;j++){
                    printf("%.3e ", _CU[j*ld_CU+i]);
                }
                printf("...");
                for(j=U_ncols-4;j<U_ncols;j++){
                    printf("%.3e ", _CU[j*ld_CU+i]);
                }
                printf("\n");
            }
            getc(stdin);
        }
    }
}

/**
 * Performs CV=V*QB using dormqr
 */
void __snewv(
        int _M,
        int ncols_qB,
        float* _CV, int ld_CV, int _Crk,
        float* _V,  int ld_V,  int _Vrk,
        float* qrtauB,
        flop_counter* flops) 
{
    int info;
    int CV_nrows = _M;
    int CV_ncols = ncols_qB;
    int V_nrows = _Vrk; //ASSUMPTION
    int V_ncols = ncols_qB; //ASSUMPTION
    int ncols = CV_nrows - V_ncols;
    float zero  = 0.0;
    if(sgemm_print_index){
        printf(" NEWV\t|%d\t|    zero     V_nrows:%d ncols:%d ld_V:%d _Crk:%d CV_ncols:%d _Vrk:%d CV_nrows:%d V_ncols:%d diff:%d\n",
              __LINE__, V_nrows,  ncols, ld_V, _Crk, CV_ncols,  _Vrk, CV_nrows, V_ncols, ncols);
    }
    //LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', V_nrows, ncols,
    //        zero, zero, &(_V[V_ncols*ld_V]), ld_V);
    char uplo = 'A';
    size_t iv = V_ncols*ld_V;
    slaset_( &uplo, &V_nrows, &ncols, &zero, &zero, &(_V[iv]), &ld_V );
    if(sgemm_print_mat){
        shc_printmat(_V,  _M, _M, ld_V);
    }
    if(sgemm_print_index){
        printf(" NEWV\t|%d\t|    ormqr     V_nrows:%d CV_nrows:%d ncols_qB:%d ld_CV:%d ld_V:%d\n",
              __LINE__, V_nrows, CV_nrows, ncols_qB, ld_CV, ld_V);
    }
    //zunmqr
    int min_MK = hcore_min(_M, ncols_qB); //number of reflectors, number of rows of qB
    info = LAPACKE_sormqr(LAPACK_COL_MAJOR, 'R', 'T',
            V_nrows, CV_nrows, min_MK, _CV, ld_CV, qrtauB, _V, ld_V);
    if(sgemm_print_mat){
        shc_printmat(_V,  _M, _M, ld_V);
    }
    if(info != 0){
        fprintf(stderr,
                "%s %d ERROR in LAPACKE_dormqr() info=%d\n",
                __FILE__, __LINE__, info);
        exit(-1);
    }
    flops->update += flop_counts('o', V_nrows, CV_nrows, min_MK, 2);  
    V_ncols  = CV_nrows;
    CV_ncols = V_nrows;
    if(sgemm_print_index){
        printf(" NEWV\t|%d\t|    trans    V_nrows:%d V_ncols:%d ld_V:%d ld_CV%d\n",
              __LINE__, V_nrows,  V_ncols, ld_V, ld_CV);
    }
    LAPACKE_sge_trans(LAPACK_COL_MAJOR, V_nrows, V_ncols,
            _V, ld_V, _CV, ld_CV);
    if(sgemm_print_index){
        printf(" NEWV\t|%d\t|    copy     V_nrows:%d V_ncols:%d ld_CV:%d ld_V:%d\n",
              __LINE__, V_nrows, V_ncols, ld_CV, ld_V);
        if(0){
            int i, j;
            for(i=0;i<100;i++){
                for(j=0;j<3;j++){
                    printf("%.3e ", _CV[j*ld_CV+i]);
                }
                printf("...");
                for(j=V_ncols-4;j<V_ncols;j++){
                    printf("%.3e ", _CV[j*ld_CV+i]);
                }
                printf("\n");
            }
            getc(stdin);
        }
    }
    if(sgemm_print_mat){
        shc_printmat(_CV,  _M, _M, ld_CV);
    }
}

/**
 * The main operation is C=alpha A B + beta C.
 * A, B and C are tile-low rank (TLR) matrices input to HCORE_sgemm(). 
 * A low rank tile T is stored as TU|TV and T can be obtained via performing TU*TV^T.
 */
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
        flop_counter* flops)
{
    if(sgemm_print_index){
		printf("%d:%s work:%p ", __LINE__, __func__, work);
		printf("M:%d N:%d  LDA:%d LDB:%d LDC:%d rk:%d maxrk:%d acc:%e a:%e b:%e\n",
				M, N, LDA, LDB, LDC, rk, maxrk, acc, alpha, beta);
	}

    int new_Crk = 0;
    /*
     * NOTES:
     * assumptions on matrix dimensions are marked as //ASSUMPTION
     */
    /*printf("%d %d|%g->%d. %g %g, %g %g\n",  */
            /*__LINE__, __COUNTER__,*/
            /*Crk[0], new_Crk, CU[0], CU[1], CV[0], CV[1]);*/
    //hcore_sgemm(Aij, ik, Ajk, -1, rank, acc);
    int _Ark = (int)(Ark[0]);
    int _Brk = (int)(Brk[0]);
    int _Crk = (int)(Crk[0]);
    if(_Ark == 0 || _Brk == 0 || _Crk == 0){
        fprintf(stderr, "%s %d: _Ark=%d _Brk=%d _Crk=%d. These rank values should not be zero.\n", __FILE__, __LINE__, _Ark, _Brk, _Crk);
        exit(-1);
    }

    int _M = M; int _N = N;
    float* _CU = CU; int ld_CU = LDC;
    float* _CV = CV; int ld_CV = LDC; int* pnew_Crk = &new_Crk;
    float* _AU = AU; int ld_AU = LDA;
    float* _AV = AV; int ld_AV = LDA;
    float* _BU = BU; int ld_BU = LDB;
    float* _BV = BV; int ld_BV = LDB;
    int rank = rk;
    { // FIXME remove these extra braces


    //printf("%s %s %d maxrank=%d\n", __FILE__, __func__, __LINE__, maxrk);
    char chall = 'A';
    float* CUclone = NULL;
    size_t CUclone_nelm =  _M * 2 * maxrk;
    int ld_CUclone = _M;
    int use_CUV_clone = 1;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CUclone = work;
            work += CUclone_nelm;
        } else {
            CUclone = malloc(CUclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CU, &ld_CU,
                CUclone, &ld_CUclone);
    }

    float* CVclone = NULL;
    size_t CVclone_nelm = _M * 2 * maxrk;
    int ld_CVclone = _M;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CVclone = work;
            work += CVclone_nelm;
        } else {
            CVclone = malloc(CVclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CV, &ld_CV,
                CVclone, &ld_CVclone);
    }
    float* _CU_save = _CU;
    float* _CV_save = _CV;

    if(use_CUV_clone == 1) {
        _CU = CUclone;
        _CV = CVclone;
    }
    int nb    = _M;//ASSUMPTION
    float* qrtauA = NULL;
    size_t qrtauA_nelm = nb;
    if(use_scratch){
        qrtauA = work;
    } else {
        qrtauA = malloc(qrtauA_nelm * sizeof(float));
    }
    assert(qrtauA != NULL);
    float* qrtauB = NULL;
    size_t qrtauB_nelm = nb;
    if(use_scratch){
        qrtauB = work + qrtauA_nelm;
    } else {
        qrtauB = malloc(qrtauB_nelm * sizeof(float));
    }
    assert(qrtauB != NULL);



    int CU_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    __sqra(_M, maxrk, _CU, ld_CU, _Crk, &CU_ncols, _AU, ld_AU, _Ark, alpha, beta, qrtauA, flops);
    //output: _CU contains [_CU _AU]. use _Crk+_Ark as number of cols
    assert(CU_ncols == (_Crk + _Ark));

    int CV_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    float* qrb_aubut = NULL;
    size_t qrb_aubut_nelm = maxrk * maxrk;
    if(use_scratch){
       qrb_aubut = work + qrtauA_nelm + qrtauB_nelm;
    } else {
       qrb_aubut = malloc(qrb_aubut_nelm * sizeof(float));
    }
    __sqrb(_M, maxrk, _CV, ld_CV, _Crk, &CV_ncols,
            _AV, ld_AV, _Ark, _BU, ld_BU, _BV, ld_BV, _Brk, qrtauB, qrb_aubut, flops);
    if(CU_ncols == 0 || CV_ncols == 0){
        fprintf(stderr, "%s %d: CU_ncols=%d CV_ncols=%d. These values should not be zero.\n", __FILE__, __LINE__, CU_ncols, CV_ncols);
        exit(-1);
    }
    if(use_scratch == 0) {
        free(qrb_aubut);
    }
    assert(CU_ncols == CV_ncols);


    int max_nb_maxrk = hcore_max(nb, maxrk);
    int ld_newU = max_nb_maxrk; //ASSUMPTION
    int ld_newV = maxrk; //ASSUMPTION
    int new_UVrk;
    float* newU = NULL;
    size_t newU_nelm = max_nb_maxrk * maxrk;
    if(use_scratch){
        newU = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm;
    } else {
        newU = malloc(newU_nelm * sizeof(float));
    }
    float* newV = NULL;
    size_t newV_nelm = nb * maxrk;

    if(use_scratch){
        newV = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm;
    } else {
        newV = malloc(newV_nelm *  sizeof(float));
    }
    assert(newU != NULL);
    assert(newV != NULL);
    float *svd_rA = NULL;
    int svd_rA_nrows  = hcore_min(_M, CU_ncols);
    int svd_rA_ncols  = CU_ncols;    //ASSUMPTION
    size_t svd_rA_nelm;
    if(sgemm_use_trmm == 1){ // allocate more because rA will be output of trmm(rA*rB^T)
        svd_rA_nelm = svd_rA_nrows * svd_rA_nrows;
    } else {
        svd_rA_nelm = svd_rA_nrows * svd_rA_ncols;
    }
    if(use_scratch == 1) {
        svd_rA = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm;
    } else {
        svd_rA = malloc(svd_rA_nelm * sizeof(float));
    }
    float *svd_rB = NULL;
    int svd_rB_nrows = hcore_min(_M, CV_ncols);
    int svd_rB_ncols = CV_ncols;
    size_t svd_rB_nelm;
    if(sgemm_use_trmm == 0) {
        svd_rB_nelm = svd_rB_nrows * svd_rB_ncols;
        if(use_scratch == 1) {
            svd_rB = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm;
        } else {
            svd_rB = malloc(svd_rB_nelm * sizeof(float));
        }
    } else {
        svd_rB_nelm = 0; //This variable will be used for calculating offset for work array
    }
    float *svd_T = NULL;
    int svd_T_nrows = svd_rA_ncols;
    int svd_T_ncols = svd_rA_ncols;
    size_t svd_T_nelm;
    if(sgemm_use_trmm == 0) {
        svd_T_nelm = svd_T_nrows * svd_T_ncols;
        if(use_scratch == 1) {
            svd_T = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm;
        } else {
            svd_T = malloc(svd_T_nelm * sizeof(float));
        }
    } else {
        svd_T_nelm = 0; //This variable will be used for calculating offset for work array
    }

    float *svd_sigma  = NULL;
    float *svd_superb = NULL;
    size_t svd_sigma_nelm  = svd_T_nrows;
    size_t svd_superb_nelm = svd_T_nrows;
    if(use_scratch == 1){
        svd_sigma  = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm;
        svd_superb = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm + svd_sigma_nelm;
    } else {
        svd_sigma  = malloc(svd_sigma_nelm  * sizeof(float));
        svd_superb = malloc(svd_superb_nelm * sizeof(float));
    }
    if(ld_newV < CU_ncols){
        fprintf(stderr, "%s %d: Increase maxrank. %d is not enough ld_newV:%d CU_ncols:%d\n", __FILE__, __LINE__, maxrk, ld_newV, CU_ncols);
        exit(-1);
    }
    __ssvd(
            _M,
            maxrk,
            _CU,  ld_CU,
            _CV,  ld_CV, CU_ncols,
            newU,   ld_newU,
            newV,   ld_newV,  &new_UVrk ,
            rank, acc,
            svd_rA, svd_rB, svd_T, svd_sigma, svd_superb,
            flops
         );
    if(use_scratch == 0) {
        free(svd_rA);
    }
    if(use_scratch == 0 && sgemm_use_trmm == 0){
        free(svd_rB);
    //    free(svd_T);
    }
    int ncols_qA = CU_ncols;
    //__newu(_CU, qrtauA, newU);
    __snewu(_M,
            ncols_qA,
            _CU, ld_CU, _Crk,
            newU, ld_newU, new_UVrk,
            qrtauA,
            flops
            );
    //Warning: number of columns of _CU is new_UVrk now!

    int ncols_qB = CV_ncols;
    //__newv(_CV, qrtauB, newV);
    __snewv(_M,
            ncols_qB,
            _CV, ld_CV, _Crk,
            newV, ld_newV, new_UVrk,
            qrtauB,
            flops
            );
    *pnew_Crk = new_UVrk;
    //Warning: number of columns of _CV is new_UVrk now!

    //printf("%d->%d\n", _Crk, *pnew_Crk);

    if(use_CUV_clone == 1) {
        slacpy_(&chall,
                &_M, &new_UVrk,
                CUclone, &ld_CUclone,
                _CU_save, &ld_CU
                );

        slacpy_(&chall,
                &_M, &new_UVrk,
                CVclone, &ld_CVclone,
                _CV_save, &ld_CV
                );
        if(use_scratch == 0) {
            free(CUclone);
            free(CVclone);
        }
    }


    if(use_scratch == 0) {
        free(qrtauA);
        free(qrtauB);
        free(newU);
        free(newV);
        free(svd_sigma);
        free(svd_superb);
    }

    } // FIXME remove these extra braces

    int old_Crk = Crk[0];
    if(sgemm_print_index){
        printf("Ark:%d Brk:%d Crk[0]:%d %g RANK CHANGE: %d->%d\n",
                _Ark, _Brk, Crk[0], Crk[0], old_Crk, *pnew_Crk);
    }
    Crk[0] = new_Crk;
    if(sgemm_print_index){
        int casted_Crk = (int)(Crk[0]);
        printf("casted_Crk:%d Ark:%d Brk:%d Crk[0]:%d %g RANK CHANGE: %d->%d\n",
                casted_Crk, _Ark, _Brk, Crk[0], Crk[0], old_Crk, new_Crk);
    }
}

/**
 * The main operation is C=alpha A B + beta C.
 * This is the first half of HCORE_sgemm().
 * Second half is performed by HCORE_sgemm_ormqr(). 
 * A and C are stored as U*V^T. 
 * B is stored as dense         
 * alpha and beta are scalars.
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
)
{
    if(sgemm_print_index){
		printf("%d:%s work:%p ", __LINE__, __func__, work);
		printf("M:%d N:%d  LDA:%d LDB:%d LDC:%d rk:%d maxrk:%d acc:%e a:%e b:%e\n",
				M, N, LDA, LDB, LDC, rk, maxrk, acc, alpha, beta);
	}

    int new_Crk = 0;
    /*
     * NOTES:
     * assumptions on matrix dimensions are marked as //ASSUMPTION
     */
    /*printf("%d %d|%g->%d. %g %g, %g %g\n",  */
            /*__LINE__, __COUNTER__,*/
            /*Crk[0], new_Crk, CU[0], CU[1], CV[0], CV[1]);*/
    //hcore_sgemm(Aij, ik, Ajk, -1, rank, acc);
    int _Ark = (int)(Ark[0]);
    int _Crk = (int)(Crk[0]);
    if(_Ark == 0 || _Crk == 0){
        fprintf(stderr, "%s %d: _Ark=%d _Crk=%d. These rank values should not be zero.\n", __FILE__, __LINE__, _Ark, _Crk);
        exit(-1);
    }

    int _M = M; int _N = N;
    float* _CU = CU; int ld_CU = LDC;
    float* _CV = CV; int ld_CV = LDC; int* pnew_Crk = &new_Crk;
    float* _AU = AU; int ld_AU = LDA;
    float* _AV = AV; int ld_AV = LDA;
    float* _B  = B;  int ld_B  = LDB;
    int rank = rk;


    //printf("%s %s %d maxrank=%d\n", __FILE__, __func__, __LINE__, maxrk);
    char chall = 'A';
    float* CUclone = NULL;
    size_t CUclone_nelm =  _M * 2 * maxrk;
    int ld_CUclone = _M;
    int use_CUV_clone = 1;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CUclone = work;
            work += CUclone_nelm;
        } else {
            CUclone = malloc(CUclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CU, &ld_CU,
                CUclone, &ld_CUclone);
    }

    float* CVclone = NULL;
    size_t CVclone_nelm = _M * 2 * maxrk;
    int ld_CVclone = _M;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CVclone = work;
            work += CVclone_nelm;
        } else {
            CVclone = malloc(CVclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CV, &ld_CV,
                CVclone, &ld_CVclone);
    }
    float* _CU_save = _CU;
    float* _CV_save = _CV;

    if(use_CUV_clone == 1) {
        _CU = CUclone;
        _CV = CVclone;
    }
    int nb    = _M;//ASSUMPTION
    float* qrtauA = NULL;
    size_t qrtauA_nelm = nb;
    if(use_scratch){
        qrtauA = work;
    } else {
        qrtauA = malloc(qrtauA_nelm * sizeof(float));
    }
    assert(qrtauA != NULL);
    float* qrtauB = NULL;
    size_t qrtauB_nelm = nb;
    if(use_scratch){
        qrtauB = work + qrtauA_nelm;
    } else {
        qrtauB = malloc(qrtauB_nelm * sizeof(float));
    }
    assert(qrtauB != NULL);



    int CU_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    __sqra(_M, maxrk, _CU, ld_CU, _Crk, &CU_ncols, _AU, ld_AU, _Ark, alpha, beta, qrtauA, flops);
    //output: _CU contains [_CU _AU]. use _Crk+_Ark as number of cols
    assert(CU_ncols == (_Crk + _Ark));

    int CV_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    float* qrb_aubut = NULL;
    size_t qrb_aubut_nelm = maxrk * maxrk;
    if(use_scratch){
       qrb_aubut = work + qrtauA_nelm + qrtauB_nelm;
    } else {
       qrb_aubut = malloc(qrb_aubut_nelm * sizeof(float));
    }
    __sqrb_b_dense(_M, maxrk, _CV, ld_CV, _Crk, &CV_ncols,
            _AV, ld_AV, _Ark, _B, ld_B, qrtauB, flops); /* NOT USED, qrb_aubut);*/
    if(CU_ncols == 0 || CV_ncols == 0){
        fprintf(stderr, "%s %d: CU_ncols=%d CV_ncols=%d. These values should not be zero.\n", __FILE__, __LINE__, CU_ncols, CV_ncols);
        exit(-1);
    }
    if(use_scratch == 0) {
        free(qrb_aubut);
    }
    assert(CU_ncols == CV_ncols);


    int ld_newU = nb; //ASSUMPTION
    int ld_newV = 2*maxrk; //ASSUMPTION
    int new_UVrk;
    float* newU = NULL;
    size_t newU_nelm = ld_newU * 2*maxrk;
    if(use_scratch){
        newU = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm;
    } else {
        newU = malloc(newU_nelm * sizeof(float));
    }
    float* newV = NULL;
    size_t newV_nelm = nb * ld_newV;

    if(use_scratch){
        newV = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm;
    } else {
        newV = malloc(newV_nelm *  sizeof(float));
    }
    assert(newU != NULL);
    assert(newV != NULL);
    float *svd_rA = NULL;
    int svd_rA_nrows  = _M < CU_ncols ? _M : CU_ncols;
    int svd_rA_ncols  = CU_ncols;    //ASSUMPTION
    size_t svd_rA_nelm;
    if(sgemm_use_trmm == 1){ // allocate more because rA will be output of trmm(rA*rB^T)
        svd_rA_nelm = svd_rA_nrows * svd_rA_nrows;
    } else {
        svd_rA_nelm = svd_rA_nrows * svd_rA_ncols;
    }
    if(use_scratch == 1) {
        svd_rA = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm;
    } else {
        svd_rA = malloc(svd_rA_nelm * sizeof(float));
    }
    float *svd_rB = NULL;
    int svd_rB_nrows = _M < CV_ncols ? _M : CV_ncols;
    int svd_rB_ncols = CV_ncols;
    size_t svd_rB_nelm;
    if(sgemm_use_trmm == 0) {
        svd_rB_nelm = svd_rB_nrows * svd_rB_ncols;
        if(use_scratch == 1) {
            svd_rB = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm;
        } else {
            svd_rB = malloc(svd_rB_nelm * sizeof(float));
        }
    } else {
        svd_rB_nelm = 0; //This variable will be used for calculating offset for work array
    }
    float *svd_T = NULL;
    int svd_T_nrows = svd_rA_ncols;
    int svd_T_ncols = svd_rA_ncols;
    size_t svd_T_nelm;
    if(sgemm_use_trmm == 0) {
        svd_T_nelm = svd_T_nrows * svd_T_ncols;
        if(use_scratch == 1) {
            svd_T = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm;
        } else {
            svd_T = malloc(svd_T_nelm * sizeof(float));
        }
    } else {
        svd_T_nelm = 0; //This variable will be used for calculating offset for work array
    }

    float *svd_sigma  = NULL;
    float *svd_superb = NULL;
    size_t svd_sigma_nelm  = svd_T_nrows;
    size_t svd_superb_nelm = svd_T_nrows;
    if(use_scratch == 1){
        svd_sigma  = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm;
        svd_superb = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm + svd_sigma_nelm;
    } else {
        svd_sigma  = malloc(svd_sigma_nelm  * sizeof(float));
        svd_superb = malloc(svd_superb_nelm * sizeof(float));
    }
    if(ld_newV < CU_ncols){ /* ld_newV=2*maxrank CU_ncols=Crk+Ark */
        fprintf(stderr, "%s %d: Increase computation maxrank. %d is not enough ld_newV:%d CU_ncols:%d\n", __FILE__, __LINE__, maxrk, ld_newV, CU_ncols);
        exit(-1);
    }
    __ssvd(
            _M,
            maxrk,
            _CU,  ld_CU,
            _CV,  ld_CV, CU_ncols,
            newU,   ld_newU,
            newV,   ld_newV,  &new_UVrk ,
            rank, acc,
            svd_rA, svd_rB, svd_T, svd_sigma, svd_superb, flops
         );
    if(use_scratch == 0) {
        free(svd_rA);
        free(svd_superb);
        free(svd_sigma);
    }
    if(use_scratch == 0 && sgemm_use_trmm == 0){
        free(svd_rB);
    //    free(svd_T);
    }

    /** parameters that will be passed to HCORE_sgemm_ormqr */
    *_p_work_new     =  work         ;
    *_p__CU          =  _CU          ;
    *_p__CV          =  _CV          ;
    *_p_CU_ncols     =  CU_ncols     ;  
    *_p_new_UVrk     =  new_UVrk     ;
    *_p_newU         =  newU         ;
    *_p_ld_newU      =  ld_newU      ;
    *_p_qrtauA       =  qrtauA       ;
    *_p_CV_ncols     =  CV_ncols     ;
    *_p_newV         =  newV         ;
    *_p_ld_newV      =  ld_newV      ;
    *_p_qrtauB       =  qrtauB       ;
    *_p_use_CUV_clone=  use_CUV_clone;
    *_p_CUclone      =  CUclone      ;
    *_p_ld_CUclone   =  ld_CUclone   ;
    *_p__CU_save     =  _CU_save     ;
    *_p_CVclone      =  CVclone      ;
    *_p_ld_CVclone   =  ld_CVclone   ;
    *_p__CV_save     =  _CV_save     ;
}

/**
 * The main operation is C=alpha A B + beta C.
 * This is the first half of HCORE_sgemm().
 * Second half is performed by HCORE_sgemm_ormqr(). 
 * A, B, and C are stored as U*V^T. 
 * alpha and beta are scalars.
 */
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
)
{
    if(sgemm_print_index){
		printf("%d:%s work:%p ", __LINE__, __func__, work);
		printf("M:%d N:%d  LDA:%d LDB:%d LDC:%d rk:%d maxrk:%d acc:%e a:%e b:%e\n",
				M, N, LDA, LDB, LDC, rk, maxrk, acc, alpha, beta);
	}

    int new_Crk = 0;
    /*
     * NOTES:
     * assumptions on matrix dimensions are marked as //ASSUMPTION
     */
    /*printf("%d %d|%g->%d. %g %g, %g %g\n",  */
            /*__LINE__, __COUNTER__,*/
            /*Crk[0], new_Crk, CU[0], CU[1], CV[0], CV[1]);*/
    //hcore_sgemm(Aij, ik, Ajk, -1, rank, acc);
    int _Ark = (int)(Ark[0]);
    int _Brk = (int)(Brk[0]);
    int _Crk = (int)(Crk[0]);
    /*if(_Ark == 0 || _Brk == 0 || _Crk == 0){
        fprintf(stderr, "%s %d: _Ark=%d _Brk=%d _Crk=%d. These rank values should not be zero.\n", __FILE__, __LINE__, _Ark, _Brk, _Crk);
        exit(-1);
    }*/

    int _M = M; int _N = N;
    float* _CU = CU; int ld_CU = LDC;
    float* _CV = CV; int ld_CV = LDC; int* pnew_Crk = &new_Crk;
    float* _AU = AU; int ld_AU = LDA;
    float* _AV = AV; int ld_AV = LDA;
    float* _BU = BU; int ld_BU = LDB;
    float* _BV = BV; int ld_BV = LDB;
    int rank = rk;


    //printf("%s %s %d maxrank=%d\n", __FILE__, __func__, __LINE__, maxrk);
    char chall = 'A';
    float* CUclone = NULL;
    size_t CUclone_nelm =  _M * 2 * maxrk;
    int ld_CUclone = _M;
    int use_CUV_clone = 1;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CUclone = work;
            work += CUclone_nelm;
        } else {
            CUclone = malloc(CUclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CU, &ld_CU,
                CUclone, &ld_CUclone);
    }

    float* CVclone = NULL;
    size_t CVclone_nelm = _M * 2 * maxrk;
    int ld_CVclone = _M;
    if(use_CUV_clone == 1) {
        if(use_scratch){
            CVclone = work;
            work += CVclone_nelm;
        } else {
            CVclone = malloc(CVclone_nelm * sizeof(float));
        }
        slacpy_(&chall,
                &_M, &_Crk,
                _CV, &ld_CV,
                CVclone, &ld_CVclone);
    }
    float* _CU_save = _CU;
    float* _CV_save = _CV;

    if(use_CUV_clone == 1) {
        _CU = CUclone;
        _CV = CVclone;
    }
    int nb    = _M;//ASSUMPTION
    float* qrtauA = NULL;
    size_t qrtauA_nelm = nb;
    if(use_scratch){
        qrtauA = work;
    } else {
        qrtauA = malloc(qrtauA_nelm * sizeof(float));
    }
    assert(qrtauA != NULL);
    float* qrtauB = NULL;
    size_t qrtauB_nelm = nb;
    if(use_scratch){
        qrtauB = work + qrtauA_nelm;
    } else {
        qrtauB = malloc(qrtauB_nelm * sizeof(float));
    }
    assert(qrtauB != NULL);



    int CU_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    __sqra(_M, maxrk, _CU, ld_CU, _Crk, &CU_ncols, _AU, ld_AU, _Ark, alpha, beta, qrtauA, flops);
    //output: _CU contains [_CU _AU]. use _Crk+_Ark as number of cols
    assert(CU_ncols == (_Crk + _Ark));

    int CV_ncols = 0;
    //Warning:In this function, there are assumptions
    //on leading dimensions and number of rows/cols
    float* qrb_aubut = NULL;
    size_t qrb_aubut_nelm = maxrk * maxrk;
    if(use_scratch){
       qrb_aubut = work + qrtauA_nelm + qrtauB_nelm;
    } else {
       qrb_aubut = malloc(qrb_aubut_nelm * sizeof(float));
    }
    __sqrb(_M, maxrk, _CV, ld_CV, _Crk, &CV_ncols,
            _AV, ld_AV, _Ark, _BU, ld_BU, _BV, ld_BV, _Brk, qrtauB, qrb_aubut, flops);
    if(CU_ncols == 0 || CV_ncols == 0){
        fprintf(stderr, "%s %d: CU_ncols=%d CV_ncols=%d. These values should not be zero.\n", __FILE__, __LINE__, CU_ncols, CV_ncols);
        exit(-1);
    }
    if(use_scratch == 0) {
        free(qrb_aubut);
    }
    assert(CU_ncols == CV_ncols);


    int ld_newU = nb; //ASSUMPTION
    int ld_newV = 2*maxrk; //ASSUMPTION
    int new_UVrk;
    float* newU = NULL;
    size_t newU_nelm = ld_newU * 2*maxrk;
    if(use_scratch){
        newU = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm;
    } else {
        newU = malloc(newU_nelm * sizeof(float));
    }
    float* newV = NULL;
    size_t newV_nelm = nb * ld_newV;

    if(use_scratch){
        newV = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm;
    } else {
        newV = malloc(newV_nelm *  sizeof(float));
    }
    assert(newU != NULL);
    assert(newV != NULL);
    float *svd_rA = NULL;
    int svd_rA_nrows  = _M < CU_ncols ? _M : CU_ncols;
    int svd_rA_ncols  = CU_ncols;    //ASSUMPTION
    size_t svd_rA_nelm;
    if(sgemm_use_trmm == 1){ // allocate more because rA will be output of trmm(rA*rB^T)
        svd_rA_nelm = svd_rA_nrows * svd_rA_nrows;
    } else {
        svd_rA_nelm = svd_rA_nrows * svd_rA_ncols;
    }
    if(use_scratch == 1) {
        svd_rA = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm;
    } else {
        svd_rA = malloc(svd_rA_nelm * sizeof(float));
    }
    float *svd_rB = NULL;
    int svd_rB_nrows = _M < CV_ncols ? _M : CV_ncols;
    int svd_rB_ncols = CV_ncols;
    size_t svd_rB_nelm;
    if(sgemm_use_trmm == 0) {
        svd_rB_nelm = svd_rB_nrows * svd_rB_ncols;
        if(use_scratch == 1) {
            svd_rB = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm;
        } else {
            svd_rB = malloc(svd_rB_nelm * sizeof(float));
        }
    } else {
        svd_rB_nelm = 0; //This variable will be used for calculating offset for work array
    }
    float *svd_T = NULL;
    int svd_T_nrows = svd_rA_ncols;
    int svd_T_ncols = svd_rA_ncols;
    size_t svd_T_nelm;
    if(sgemm_use_trmm == 0) {
        svd_T_nelm = svd_T_nrows * svd_T_ncols;
        if(use_scratch == 1) {
            svd_T = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm;
        } else {
            svd_T = malloc(svd_T_nelm * sizeof(float));
        }
    } else {
        svd_T_nelm = 0; //This variable will be used for calculating offset for work array
    }

    float *svd_sigma  = NULL;
    float *svd_superb = NULL;
    size_t svd_sigma_nelm  = svd_T_nrows;
    size_t svd_superb_nelm = svd_T_nrows;
    if(use_scratch == 1){
        svd_sigma  = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm;
        svd_superb = work + qrtauA_nelm + qrtauB_nelm + qrb_aubut_nelm + newU_nelm + newV_nelm + svd_rA_nelm + svd_rB_nelm + svd_T_nelm + svd_sigma_nelm;
    } else {
        svd_sigma  = malloc(svd_sigma_nelm  * sizeof(float));
        svd_superb = malloc(svd_superb_nelm * sizeof(float));
    }
    if(ld_newV < CU_ncols){ /* ld_newV=2*maxrank CU_ncols=Crk+Ark */
        fprintf(stderr, "%s %d: Increase computation maxrank. %d is not enough ld_newV:%d CU_ncols:%d\n", __FILE__, __LINE__, maxrk, ld_newV, CU_ncols);
        exit(-1);
    }
    __ssvd(
            _M,
            maxrk,
            _CU,  ld_CU,
            _CV,  ld_CV, CU_ncols,
            newU,   ld_newU,
            newV,   ld_newV,  &new_UVrk ,
            rank, acc,
            svd_rA, svd_rB, svd_T, svd_sigma, svd_superb, flops
         );
    if(use_scratch == 0) {
        free(svd_rA);
        free(svd_superb);
        free(svd_sigma);
    }
    if(use_scratch == 0 && sgemm_use_trmm == 0){
        free(svd_rB);
    //    free(svd_T);
    }

    /** parameters that will be passed to HCORE_sgemm_ormqr */
    *_p_work_new     =  work         ;
    *_p__CU          =  _CU          ;
    *_p__CV          =  _CV          ;
    *_p_CU_ncols     =  CU_ncols     ;  
    *_p_new_UVrk     =  new_UVrk     ;
    *_p_newU         =  newU         ;
    *_p_ld_newU      =  ld_newU      ;
    *_p_qrtauA       =  qrtauA       ;
    *_p_CV_ncols     =  CV_ncols     ;
    *_p_newV         =  newV         ;
    *_p_ld_newV      =  ld_newV      ;
    *_p_qrtauB       =  qrtauB       ;
    *_p_use_CUV_clone=  use_CUV_clone;
    *_p_CUclone      =  CUclone      ;
    *_p_ld_CUclone   =  ld_CUclone   ;
    *_p__CU_save     =  _CU_save     ;
    *_p_CVclone      =  CVclone      ;
    *_p_ld_CVclone   =  ld_CVclone   ;
    *_p__CV_save     =  _CV_save     ;
}

/**
 * Performs CU=QA*U and CV=V*QB using dormqr.
 */
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
)
{
    char chall = 'A';
    int new_Crk = new_UVrk;
    int _Crk = (int)(Crk[0]);
    int _M = M; int _N = N;
    int ld_CU = LDC;
    int ld_CV = LDC;
    int ncols_qA = CU_ncols;
    if(new_UVrk > storage_maxrank) {
        fprintf(stderr, "%s %d: Increase storage maxrank %d. Observed rank for U and V is %d\n", __FILE__, __LINE__, storage_maxrank, new_UVrk);
        exit(-1);
    }
    __snewu(_M,
            ncols_qA,
            _CU, ld_CU, _Crk,
            newU, ld_newU, new_UVrk,
            qrtauA, flops
            );
    int ncols_qB = CV_ncols;
    __snewv(_M,
            ncols_qB,
            _CV, ld_CV, _Crk,
            newV, ld_newV, new_UVrk,
            qrtauB, flops
            );
    if(use_CUV_clone == 1) {
        slacpy_(&chall,
                &_M, &new_UVrk,
                CUclone, &ld_CUclone,
                _CU_save, &ld_CU
                );

        slacpy_(&chall,
                &_M, &new_UVrk,
                CVclone, &ld_CVclone,
                _CV_save, &ld_CV
                );
        if(use_scratch == 0) {
            free(CUclone);
            free(CVclone);
        }
    }
    if(use_scratch == 0) {
        free(qrtauA);
        free(qrtauB);
        free(newU);
        free(newV);
    }
    int old_Crk = Crk[0];
    if(sgemm_print_index){
        printf("Crk[0]:%d %g RANK CHANGE: %d->%d\n",
                Crk[0], Crk[0], old_Crk, new_Crk);
    }
    Crk[0] = new_Crk;
    if(sgemm_print_index){
        int casted_Crk = (int)(Crk[0]);
        printf("casted_Crk:%d  Crk[0]:%d %g RANK CHANGE: %d->%d\n",
                casted_Crk, Crk[0], Crk[0], old_Crk, new_Crk);
    }
}

void HCORE_sgemm_dense(HCORE_enum transA, int transB,
                       int M, int N, int K,
                       float alpha,
                       const float *A,
                       int LDA,
                       const float *B,
                       int LDB,
                       float beta,
                       float *C,
                       int LDC) {

    cblas_sgemm(CblasColMajor, (CBLAS_TRANSPOSE )transA, (CBLAS_TRANSPOSE)transB, M, N, K,
                alpha,A, LDA, B, LDB, beta, C, LDC);

}


