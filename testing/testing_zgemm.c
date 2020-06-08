/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file testing_zsyrk.c
 *
 *  Testing for HCORE_zsyrk().
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2020-06-01
 **/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "hcore.h"
#include "hcore_z.h"
#include "flop_util_structs.h"
#ifdef LAPACKE_UTILS
#include <lapacke_utils.h>
#endif

int use_scratch = 0; /** dynamically allocate buffers */
int print_matrices = 1;

void _print_array(double* array, int M, int N, int LDA){
    printf("M:%d N:%d LDA:%d\n", M, N, LDA);
    if(0){
        for(int i = 0; i < M*N; i++){
            printf("%.2e ", array[i]);
        }
        printf("\n");
    }
    for(int i=0; i<M; i++) {
        if(i==0) printf("[");
        else printf(" ");
        for(int j=0; j<N; j++) {
            if(j==0) printf("[");
            printf("%+.3e", array[j*LDA+i]);
            if(j==N-1) printf("]");
            else printf(",");
        }
        if(i==M-1) printf("]");
        else printf(",\n");
    }
    printf("\n");
}
#define print_array(A, M, N, LDA) do { \
	if(print_matrices != 0) { \
    	printf("%s:%d ", __FILE__, __LINE__); \
    	_print_array(A, M, N, LDA); \
	} \
} while(0);

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

void diff_matrix(double *A, double *B, int NB, int BB) {
    int X, Y, x, y, i;
    int N = NB*BB;

    for (Y = 0; Y < BB; Y++) {
        for (y = 0; y < NB; y++) {
            for (X = 0; X < BB; X++) {
                //if(y==0)printf(" X:%d\n", X);
                for (x = 0; x < NB; x++) {

                    double a, b, c, d, e;
                    a = fabs(A[(Y*NB+y) + (X*NB+x)*N]);
                    b = fabs(B[(Y*NB+y) + (X*NB+x)*N]);
                    c = max(a, b);
                    d = min(a, b);
                    e = (c - d) / d;
                    if (d == 0.0) e = 0.0;

                    printf("%c", e == 0.0 || e < 0.1 ? '.' : '#');
                    //printf("%.4lf-%.4lf ", A[(Y*NB+y) + (X*NB+x)*N], B[(Y*NB+y) + (X*NB+x)*N]);
                    //if (x == 3) x = NB-5;
                    //if (x == 7) x = NB-1;
                }
                printf("  |");
            }
            printf("\n");
            //if (y == 3) y = NB-5;
            //if (y == 7) y = NB-1;
        }
        if (Y < BB-1)
            for (i = 0; i < BB*12; i++) printf("=");
        printf("\n");
    }
    printf("\n");
}

double get_rnorm( double* mat1, double* mat2, int ld1,     int ld2,     int nrows,    int ncols   ) {
    double rnorm = 0.0;
    assert(ncols == ld1);
    double* mat1clone = calloc(nrows*ncols, sizeof(*mat1clone)); // WARNING LD is ignored
    memcpy(mat1clone, mat1, nrows*ncols*sizeof(*mat1clone));
    int i, j;
    /* Compute the Residual || reference_C - C|| */
    for(i = 0; i < nrows; i++){
        for(j = 0; j < ncols; j++){
    //        printf("%+.3e|%+.3e  ", mat1clone[i+j*ld1], mat2[i+j*ld2]);
            mat1clone[i+j*ld1] -= mat2[i+j*ld2];
        }
    //    printf("\n");
    }
    char norm = 'I';
    double *work = (double *)malloc(nrows*sizeof(double));
    rnorm = dlange(&norm, &nrows, &ncols, mat1clone, &ld1, work);
    free(work);
    free(mat1clone);
    return rnorm;
}

int generate_compress(
        int nrows,
        int ncols,
        int ld,
        double* matrix,
        double* _U,
        double* _V,
        int storage_maxrank, 
        int maxMNK,
        double acc_threshold, 
        int* iseed,
        int* prank
        ){
        int print_mat = 0;
        int minNrowsNcols = min(nrows, ncols);

        /** Temporary U and V for storing output of SVD and transpose V */
        size_t nelm_tUV = maxMNK * maxMNK;
        double* _tU = calloc(nelm_tUV, sizeof(*_tU));
        double* _tV = calloc(nelm_tUV, sizeof(*_tV));
        double* _tUV = calloc(nelm_tUV, sizeof(*_tUV));

        int info;
        /**
          lapack_int LAPACKE_dlatms (int matrix_layout, lapack_int m, lapack_int n, char dist, lapack_int * iseed, char sym, double * d, lapack_int mode, double cond, double dmax, lapack_int kl, lapack_int ku, char pack, double * a, lapack_int lda);
          */
        int nelm_d = maxMNK; /** Allocate more for singular values */
        double* _d = calloc(nelm_d, sizeof(*_d)); 
        int i;
        for(i = 0; i < nelm_d; i++){
            _d[i] = pow(10, -1*i);
        }
        if(print_mat) {
            printf("Singular values which will be used by dlatms in generation of A, B, and C: ");
            print_array(_d, 1, nelm_d, 1);
        }
        double* _S = calloc(nelm_d, sizeof(*_d)); 
        double* superb = calloc(nelm_d, sizeof(*_d)); 

        /** Generate */
        info = LAPACKE_dlatms(LAPACK_COL_MAJOR, nrows, ncols, 'U', iseed, 'N', _d, 0, -1, -1, nrows-1, ncols-1, 'N', matrix, ld);
        if(info != 0) {printf("Error in LAPACKE_dlatms. Info=%d\n", info);return -1;}
        if(print_mat) {
            printf("dense matrix\n");
            print_array(matrix, nrows, ncols, ld);
        }
        char chall = 'A';
        assert(nrows <= ld);
        double* matrixclone = calloc(ld*ncols, sizeof(*matrixclone));
        dlacpy_(&chall, &nrows, &ncols, matrix, &ld, matrixclone, &ld);
        /** Compress */
        /** https://software.intel.com/en-us/node/521150 */
        info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', nrows, ncols, matrixclone, ld, _S, _tU, ld, _tV, ld, superb);
        free(matrixclone);
        if(info != 0) {printf("Error in LAPACKE_dgesvd. Info=%d\n", info);return -1;}
        int k;
        /** S * V */
        for(k = 0; k < ncols /** ? */; k++){
            double diagval = _S[k];
            cblas_dscal(nrows, diagval, &_tV[k], ld);
        }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nrows, ncols, nrows, 1.0, _tU, ld, _tV, ld, 0.0, _tUV, ld);
        if(print_mat) {
            printf("UV\n");
            print_array(_tUV, nrows, ncols, ld);


            printf("tU matrix\n");
            print_array(_tU, nrows, nrows, ld);
            printf("tV matrix\n");
            print_array(_tV, ncols, ncols, ld);
            printf("Singular values\n");
            print_array(_S, 1, minNrowsNcols, 1);
        }
    
        int rank = 0;
        while(_S[rank] >= acc_threshold && rank < minNrowsNcols){rank++;}
        if( rank > storage_maxrank) {
            rank = storage_maxrank;
        }
        *prank = rank;
        printf("Rank:%d storage_maxrank:%d minNrowsNcols:%d\n", rank, storage_maxrank, minNrowsNcols);
        if(print_mat) {
            printf("Truncated singular values\n");
            print_array(_S, 1, rank, 1);
        }

        dlacpy_(&chall, &nrows, &rank, _tU, &ld, _U, &ld);
        if(print_mat) {
            printf("U matrix\n");
            print_array(_U, nrows, rank, ld);
        }
        LAPACKE_dge_trans(LAPACK_COL_MAJOR, rank, ncols, _tV, ld, _V, ld);
        if(print_mat) {
            printf("V^T matrix\n");
            print_array(_V, ncols, rank, ld);
        }
        

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                nrows, ncols, rank, 1.0, _U, ld, _V, ld,
                0.0, _tUV, ld);
        if(print_mat) {
            printf("UV^T matrix\n");
            print_array(_tUV, nrows, ncols, ld);
        }
        double rnorm = get_rnorm(matrix, _tUV, ld, ld, nrows, ncols);
        printf("inf norm(dense-appU*appV):%e\n", rnorm);

    
        free(_d);
        free(_S);
        free(superb);
        free(_tU);
        free(_tV);
        free(_tUV);
        return 0; 
}

/**
 * This program tests HCORE_zgemm(). 
 * The matrices are generated using LAPACKE_dlatms() (https://software.intel.com/en-us/node/534990).
 *
 * The following steps are performed:
 *
 * 1. Generate A using LAPACKE_dlatms() 
 *  . Generate B using LAPACKE_dlatms() 
 *  . Generate C using LAPACKE_dlatms() 
 *  . Perform refC = 1.0 C - 1.0 A B using cblas_dgemm()
 *  . Perform tlrC = 1.0 (CU CV^T) - 1.0 (AU AV^T) (BU BV^T) using HCORE_zgemm()
 */
int main(){
    int print_mat = 0;
    printf("testing HCORE_zgemm() in HCORE library\n");
    int _M = 10;
    int _N = _M; //10; //8;
    int _K = _M; //10; //9;
    int minMK = min(_M, _K);
    int maxMN = max(_M, _N);
    int maxMNK = max(maxMN, _K);
    int storage_maxrank = 6; 
    double acc_threshold = 1.0e-4;
    int rank_threshold = 0; /** not used for now */
    int ld_A = _M;
    int ld_B = _K;
    int ld_C = _M; 

    size_t nelm_A = _M * _K;
    double* _A = calloc(nelm_A, sizeof(*_A));

    size_t nelm_B = _K * _N;
    double* _B = calloc(nelm_B, sizeof(*_B));
    
    size_t nelm_C = _M * _N;
    double* _C = calloc(nelm_C, sizeof(*_C));
    
    size_t nelmsize_C = nelm_C * sizeof(*_C);
    double* reference_C = calloc(nelm_C, sizeof(*_C));
    
    size_t nelm_AU = _M * storage_maxrank;
    double* _AU = calloc(nelm_AU, sizeof(*_AU));
    size_t nelm_AV = max(_K, ld_A) * storage_maxrank; /** LD for AV is set to LD of A*/
    double* _AV = calloc(nelm_AV, sizeof(*_AV));
    
    size_t nelm_BU = _K * storage_maxrank;
    double* _BU = calloc(nelm_BU, sizeof(*_BU));
    size_t nelm_BV = _N * storage_maxrank;
    double* _BV = calloc(nelm_BV, sizeof(*_BV));

    size_t nelm_CU = _M * storage_maxrank;
    double* _CU = calloc(nelm_CU, sizeof(*_CU));
    size_t nelm_CV = _N * storage_maxrank;
    double* _CV = calloc(nelm_CV, sizeof(*_CV));

    int Ark = 0;
    int Brk = 0;
    int Crk = 0;
    int iseed[4] = {0,0,0,1};
    /** A */
    printf("============== A ==============\n");
    generate_compress(_M, _K, ld_A, _A, _AU, _AV, storage_maxrank, maxMNK, acc_threshold, iseed, &Ark);
    if(print_mat) {
        print_array(_AU, _M, Ark, ld_A);
        print_array(_AV, _M, Ark, ld_A);
    }
    /** B */
    printf("============== B ==============\n");
    generate_compress(_K, _N, ld_B, _B, _BU, _BV, storage_maxrank, maxMNK, acc_threshold, iseed, &Brk);
    if(print_mat) {
        print_array(_BU, _N, Brk, ld_B);
        print_array(_BV, _N, Brk, ld_B);
    }
    /** C */
    printf("============== C ==============\n");
    generate_compress(_M, _N, ld_C, _C, _CU, _CV, storage_maxrank, maxMNK, acc_threshold, iseed, &Crk);
    if(print_mat) {
        print_array(_CU, _M, Crk, ld_C);
        print_array(_CV, _M, Crk, ld_C);
    }

    memcpy(reference_C, _C, nelmsize_C);

    if(print_mat) {
        print_array(_A, _M, _K, ld_A);
        print_array(_B, _K, _N, ld_B);
        print_array(_C, _M, _N, ld_C);
        print_array(reference_C, _M, _N, ld_C);
    }
	double alpha = 1.0;
	double beta = 1.0;
	HCORE_enum transA = CblasNoTrans;
	HCORE_enum transB = CblasTrans;
	/*C := alpha * A * B^T + beta * C */
	cblas_dgemm(CblasColMajor, 
		transA /*A*/, 
		transB /*B^T*/, 
        _M, _N, _K,
		alpha, _A, ld_A, _B, ld_B, beta,
		reference_C, ld_C);
    if(print_mat) {
        printf("final reference C matrix\n");
        print_array(reference_C, _M, _N, ld_C);
    }

    double _Ark = (double) Ark; 
    double _Brk = (double) Brk; 
    double _Crk = (double) Crk; 
	flop_counter flops;
	HCORE_zgemm(transA, transB, // FIXME handle these in HCORE, not in HiCMA codelets 
            _M, _N,  // FIXME add K to HCORE_zgemm()
            alpha, 
            _AU, _AV, &_Ark, ld_A,
            _BU, _BV, &_Brk, ld_B,
            beta,
            _CU, _CV, &_Crk, ld_C,
            rank_threshold,
            _M,
            //storage_maxrank,
            acc_threshold,
            NULL,
            &flops);
    int newCrank = (int) _Crk;
    printf("Crank: %d -> %d\n", Crk, newCrank);
            
    if(print_mat) {
        print_array(_CU, _M, newCrank, ld_C);
        print_array(_CV, _M, newCrank, ld_C);
    }
	/** C = CU * CV^T */
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		_M, _N, newCrank, 1.0, _CU, ld_C, _CV, ld_C,
		0.0, _C, ld_C);
    if(print_mat) {
        printf("final C matrix\n");
        print_array(_C, _M, _N, ld_C);
        diff_matrix(_C, reference_C, _M, 1); /** Assumes M-by-M matrix*/
    }

    double rnorm = get_rnorm(reference_C, _C, ld_C, ld_C, _M, _N);
    assert(rnorm < 10 * acc_threshold); 
    printf("inf norm(C - appC):%e\n", rnorm);

    free(reference_C);
    free(_A);
    free(_AU);
    free(_AV);
	free(_B);
	free(_BU);
	free(_BV);
	free(_C);
	free(_CU);
    free(_CV);
    return 0;
}
