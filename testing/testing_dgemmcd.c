/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file testing_dsyrk.c
 *
 *  Testing for HCORE_dgemmcd().
 *  HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "hcore.h"
#include "hcore_d.h"
#include "flop_util_structs.h"

int use_scratch = 0; /** dynamically allocate buffers */
int print_matrices = 0;

void _print_array(double* array, int M, int N, int LDA){
    printf("M:%d N:%d LDA:%d\n", M, N, LDA);
    for(int i = 0; i < M*N; i++){
        printf("%.2e ", array[i]);
    }
    printf("\n");
    for(int i=0; i<M; i++) {
        if(i==0) printf("[");
        for(int j=0; j<N; j++) {
            if(j==0) printf("[");
            printf("%.1e", array[j*LDA+i]);
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
    	printf("%s:%d", __FILE__, __LINE__); \
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

                    printf("%c", e == 0.0 || e < 0.001 ? '.' : '#');
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

/**
 * This program tests HCORE_dgemmcd(). The following steps are performed:
 *
 * 1. Generate random matrix C
 * 2. Generate random matrix AU
 * 3. Generate random matrix AV
 * 2. Generate random matrix BU
 * 3. Generate random matrix BV
 * 4. Compute A = AU AV^T
 * 4. Compute B = AU AV^T
 * 5. Perform refC = 1.0 C - 1.0 A B^T using cblas_dgemm()
 * 6. Perform tlrC = 1.0 C - 1.0 (AU AV^T) (BU BV^T)^T using HCORE_dgemmcd()
 * 7. Compute inifinity norm of difference of |refC-tlrC| 
 */
int main(){
    printf("testing HCORE_dsyrk() in HCORE library\n");
    int nrows = 10;
    int rankA = 7;
    int rankB = 6;
    int ld_C = nrows; 
    int ld_A = nrows;
    int ld_AU = nrows;
    int ld_AV = nrows;
    int ld_B = nrows;
    int ld_BU = nrows;
    int ld_BV = nrows;


    size_t nelm_C = nrows * nrows;
    double* _C = calloc(nelm_C, sizeof(*_C));

    size_t nelm_A = nrows * nrows;
    double* _A = calloc(nelm_A, sizeof(*_A));
    
	size_t nelm_AU = nrows * rankA;
    double* _AU = calloc(nelm_AU, sizeof(*_AU));
    double* _AV = calloc(nelm_AU, sizeof(*_AV));
    
    size_t nelm_B = nrows * nrows;
    double* _B = calloc(nelm_B, sizeof(*_B));
    
	size_t nelm_BU = nrows * rankB;
    double* _BU = calloc(nelm_BU, sizeof(*_BU));
    double* _BV = calloc(nelm_BU, sizeof(*_BV));

    size_t nelmsize_C = nelm_C * sizeof(*_C);
    double* reference_C = calloc(nelm_C, sizeof(*_C));

    int i, j;
    for(i = 0; i < nrows; i++){
        for(j = 0; j <= i; j++){
            _C[i+j*ld_C] = rand() / (double)RAND_MAX;
        }
    }
    memcpy(reference_C, _C, nelmsize_C);
    for(i = 0; i < nrows; i++){
        for(j = 0; j < rankA; j++){
            _AU[i+j*ld_AU] = rand() / (double)RAND_MAX;
            _AV[i+j*ld_AV] = rand() / (double)RAND_MAX;
        }
        for(j = 0; j < rankB; j++){
            _BU[i+j*ld_BU] = rand() / (double)RAND_MAX;
            _BV[i+j*ld_BV] = rand() / (double)RAND_MAX;
        }
    }
	/** A = AU * AV */
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		nrows, nrows, rankA, 1.0, _AU, ld_AU, _AV, ld_AV,
		0.0, _A, ld_A);
	/** B = BU * BV */
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		nrows, nrows, rankB, 1.0, _BU, ld_BU, _BV, ld_BV,
		0.0, _B, ld_B);
    printf("initial C matrix\n");
    print_array(_C, nrows, nrows, ld_C);
    printf("A matrix\n");
    print_array(_A, nrows, nrows, ld_A);
    printf("B matrix\n");
    print_array(_B, nrows, nrows, ld_B);

	double alpha = -1.0;
	double beta = 1.0;
	/*C := alpha * A * A^T + beta * C */
	cblas_dgemm(CblasColMajor, CblasNoTrans, 
		CblasTrans /*A B^T*/, 
		nrows /*nrows of C*/, 
		nrows /*ncols of C*/, 
		nrows /*ncols of A*/,
		alpha, _A, ld_A, _B, ld_B, beta,
		reference_C, ld_C);
    printf("final reference C matrix\n");
    print_array(reference_C, nrows, nrows, ld_C);

	flop_counter flops;
    HCORE_dgemmcd(CblasNoTrans, CblasTrans,
            nrows, alpha, _AU, _AV, rankA, ld_A,
            _BU, _BV, rankB, ld_B,
            beta, _C, ld_C, nrows, NULL, &flops);
    printf("final C matrix\n");
    print_array(_C, nrows, nrows, ld_C);
	diff_matrix(_C, reference_C, nrows, 1);

    /* Compute the Residual || reference_C - C|| */
    for(i = 0; i < nrows; i++){
        for(j = 0; j < nrows; j++){
            reference_C[i+j*ld_C] -= _C[i+j*ld_C];
        }
    }
	char norm = 'I';
    double *work = (double *)malloc(nrows*sizeof(double));
    double residual_norm = dlange(&norm, &nrows, &nrows, reference_C, &ld_C, work);
	printf("Infinity norm of |reference_C - C| = %.2e\n", residual_norm);
    double eps = dlamch("Epsilon");
    printf("The relative machine precision (eps) = %.2e \n", eps);
	assert(residual_norm < 1000 * eps); 
	free(_C);
	free(reference_C);
	free(_A);
	free(_AU);
	free(_AV);
	free(_B);
	free(_BU);
	free(_BV);
    return 0;
}
