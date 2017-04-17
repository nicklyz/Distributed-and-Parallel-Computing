#include "const.h"
#include "stdio.h"
#include "string.h"
#include <omp.h>

void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k, ii, jj, kk;
	float tmp, acc00, acc01, acc10, acc11;
	// prefill Matrix C with 0s
	memset(C, 0, sizeof(C[0][0]) * ni * nj);
	// create blocks for improvement
	size_t ib = 64;
	size_t kb = 32;
	/*
	for (ii = 0; ii < ni; ii += ib) {
		for (kk = 0; kk < nk; kk += kb) {
			for (j = 0; j < nj; j += 2) {
				for (i = ii; i < ii+ib; i += 2) {
					if (kk = 0) {
						acc00 = acc01 = acc10 = acc11 = 0;
					}
					else {
						acc00 = C[i+0][j+0];
						acc01 = C[i+0][j+1];
						acc10 = C[i+1][j+0];
						acc11 = C[i+1][j+1];
					}
					for (k = kk; k < kk + kb; k++) {
					//for (k = 0; k < nk; k++) {
						acc00 += A[i+0][k] * B[k][j+0];
						acc01 += A[i+0][k] * B[k][j+1];
						acc10 += A[i+1][k] * B[k][j+0];
						acc11 += A[i+1][k] * B[k][j+1];
					}
					C[i+0][j+0] = acc00;
					C[i+0][j+1] = acc01;
					C[i+1][j+0] = acc10;
					C[i+1][j+1] = acc11;
				}
			}
		}
	}
	*/
	/*
	// for (jj=0; jj<nj; jj+=_block_size) {
		for (kk=0; kk<nk; kk+=_block_size) {
			// #pragma omp parallel for private(j, k)
			for (i=0; i<ni; i++) {
				// for (j=jj; j<((jj+_block_size)>nj ? nj : (jj+_block_size)); j++) {
				for (j=0; j<nj; j++) {
					tmp = 0;
					for (k=kk; k < ((kk+_block_size)>nk ? nk : (kk+_block_size)); k++) {
					//for (k=0; k<nk; k++) {
						tmp += A[i][k] * B[k][j];
					}
					C[i][j] += tmp;
				}
			}
		}
	// }
	*/
}

