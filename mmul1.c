#include "const.h"
#include <omp.h>

void mmul1(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{	
	int i, j, k;
	#pragma omp parallel private(i) shared(j, k, A, B, C)
	{
	#pragma omp for schedule(static)
	for (i=0; i<ni; i++) {
		for (k=0; k<nk; k++) {
			for (j=0; j<nj; j++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
	}
}

