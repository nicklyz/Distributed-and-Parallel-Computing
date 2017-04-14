#include "const.h"

void mmul1(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k;
	for (i=0; i<ni; i++) {
		for (j=0; j<nj; j++) {
			C[i][j] = 0;
			for (k=0; k<nk; k++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

