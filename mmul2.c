#include "const.h"
#include "stdio.h"
#include "string.h"

void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k;
	// prefill Matrix C with 0s
	memset(C, 0, sizeof(C[0][0]) * ni * nj);
	// create blocks for improvement
	int _block_size = 16;
	for (i=0; i<ni; i++) {
		for (k=0; k<nk; k++) {
			for (j=0; j<nj; j++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

