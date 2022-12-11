/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and 
 *                     All rights reserved.
 **/
/**
 * @file descsame.c
 *
 * This file contains the functions for checking whether two matrices are same or not.
 * 
 * HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.2
 * @author Kadir Akbudak
 * @date 2020-12-17
 **/
#include "auxdescutil.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
/** 
 * This function isused to compare to descriptors in terms of size and numerical values.
 * Do not use this function with MPI
 */
void check_same_array(double *L, double *R, int nelm, int line, char *file){
    int i;
    int error_encountered = 0;
    for(i=0; i<nelm; i++){
        double valL = L[i];
        double valR = R[i];
        double diff = fabs(valL - valR);
        double thresh = 1e-12;
        //double thresh = 1e-2;
        if(diff > thresh ){
            printf("Elm:%d diff:%g val:%.14e %.14e\t",  i,  diff, valL, valR);
            error_encountered = 1;
            //exit(1);
            break;
        }
    }
    if (error_encountered == 0) {
        printf("arrays are same at line %d of %s\n", line, file);
    } else {
        printf("arrays are NOT SAME !!!!!!!!!!!!!!!!!!!!!!  at line %d of %s\n", line, file);
        exit(1);
    }
}
