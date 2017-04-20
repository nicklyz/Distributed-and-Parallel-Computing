#include "const.h"
#include "stdio.h"
#include "string.h"
#include <stdlib.h>
#include <omp.h>

void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k, ii, jj, kk;
	// prefill Matrix C with 0s
	memset(C, 0, sizeof(C[0][0]) * ni * nj);
	// create blocks for improvement
	const int BS = 64;
	#pragma omp parallel private (i, j, k, ii, jj, kk) shared(A, B, C)
	{
	#pragma omp for schedule(static)
	for (i = 0; i < ni; i += BS)
		for (j = 0; j < nj; j += BS)
		{
			float buff[BS][BS];
			memset(buff, 0, sizeof(float) * BS * BS);
			for (k = 0; k < nk; k += BS)
				for (ii = 0; ii < BS; ++ii)
					for (kk = 0; kk < BS; ++kk) 
						for (jj = 0; jj < BS; ++jj) 
							buff[ii][jj] += A[i+ii][k+kk] * B[k+kk][j+jj];
			// put buffer back to c
			for (ii = 0; ii < BS; ++ii)
				memcpy(&C[i+ii][j], &buff[ii][0], sizeof(float) * BS);
		}
	} // end pragma
}
