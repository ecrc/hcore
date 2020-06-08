/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file auxdescutil.h
 *
 * This file contains the declarations of auxiliary functions for printing matrices.
 *
 * HCORE is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2019-11-14
 **/
#ifndef __AUXDESCUTIL__
#define __AUXDESCUTIL__
#include <stdio.h>
#include "hcore.h"

#define tld(d) (d->mb)
#define tsa(d,i,j) (((j)*(d->mt)+(i))*(d->mb)*(d->nb))

void printmat(double * A, int64_t m, int64_t n, int64_t ld, int irs, int ics);
void printmat_format(double * A, int64_t m, int64_t n, int64_t ld, int irs, int ics, int format);
void _printmat(double * A, int64_t m, int64_t n, int64_t ld);
void check_same_array(double *L, double *R, int nelm, int line, char *file);

#endif
