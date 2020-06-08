The HCORE Library
===========================================================
The **HCORE** library provides BLAS operations for matrices
in low rank format, i.e., A=UV. HCORE is part of the HiCMA
library (https://github.com/ecrc/hicma).


Features of HCORE 0.1.0
-----------------------------
* Matrix Compression 
* Matrix-Matrix Multiplication (HCORE_GEMM)
* Symmetric Rank-K update (HCORE_SYRK)
* Double Precision
* Testing Suite (soon) and Examples


Current Research
----------------
* Hardware Accelerators
* Support for Multiple Precisions
* Autotuning: Tile Size, Fixed Accuracy and Fixed Ranks
* Support for OpenMP
* Support for HODLR, H, HSS and H2 


External Dependencies
---------------------
HCORE depends on the following libraries:
* BLAS
* LAPACK

Installation
------------

Please see INSTALL.md for information about installing and testing.


References
-----------
1. K. Akbudak, H. Ltaief, A. Mikhalev, and D. E. Keyes, *Tile Low Rank Cholesky Factorization for 
Climate/Weather Modeling Applications on Manycore Architectures*, **International Supercomputing 
Conference (ISC17)**, June 18-22, 2017, Frankfurt, Germany.

2. K. Akbudak, H. Ltaief, A. Mikhalev, A. Charara, and D. E. Keyes, *Exploiting Data Sparsity for Large-Scale Matrix Computations*, **Euro-Par 2018**, August 27-31, 2018, Turin, Italy.

3. Q. Cao, Y. Pei, T. Herauldt, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief, D. E. Keyes, and J. Dongarra, *Performance Analysis of Tile Low-Rank Cholesky Factorization Using PaRSEC Instrumentation Tools*, **2019 IEEE/ACM International Workshop on Programming and Performance Visualization Tools (ProTools)**, Denver, CO, USA, 2019, pp. 25-32.

4. Q. Cao, Y. Pei, K. Akbudak, A. Mikhalev, G. Bosilca, H. Ltaief, D. E. Keyes, and J. Dongarra, *Extreme-Scale Task-Based Cholesky Factorization Toward Climate and Weather Prediction Applications*, **The Platform for Advanced Scientific Computing (PASC 2020)**.

5. N. Al-Harthi, R. Alomairy, K. Akbudak, R. Chen, H. Ltaief, H. Bagci, and D. E. Keyes, *Solving Acoustic Boundary Integral Equations Using High Performance Tile Low-Rank LU Factorization*, **International Supercomputing Conference (ISC 2020)**.


![Handout](docs/HiCMA-handout-SC17.png)
