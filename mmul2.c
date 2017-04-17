#include "const.h"
#include "stdio.h"
#include "string.h"
#include <omp.h>

void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
	int i, j, k, ii, jj, kk;
	float acc00, acc01, acc10, acc11;
	// prefill Matrix C with 0s
	memset(C, 0, sizeof(C[0][0]) * ni * nj);
	// create blocks for improvement
	int ib = 128;
	int kb = 128;
	#pragma omp parallel for private (i, j, k, acc00, acc01, acc10, acc11)
	for (ii = 0; ii < ni; ii += ib)
	{
	    for (kk = 0; kk < nk; kk += kb)
	    {
	        for (j=0; j < nj; j += 2)
	        {
	            for(i = ii; i < ii + ib; i += 2 )
	            {
			        #pragma omp critical
	                {
                    if (kk == 0)
	                    acc00 = acc01 = acc10 = acc11 = 0;
	                else
	                {
	                    acc00 = C[i + 0][j + 0];
	                    acc01 = C[i + 0][j + 1];
	                    acc10 = C[i + 1][j + 0];
	                    acc11 = C[i + 1][j + 1];
	                }
	                for (k = kk; k < kk + kb; k++)
	                {
	                    acc00 += A[i + 0][k] * B[k][j + 0];
	                    acc01 += A[i + 0][k] * B[k][j + 1];
	                    acc10 += A[i + 1][k] * B[k][j + 0];
	                    acc11 += A[i + 1][k] * B[k][j + 1];
	                }
	                C[i + 0][j + 0] = acc00;
	                C[i + 0][j + 1] = acc01;
	                C[i + 1][j + 0] = acc10;
	                C[i + 1][j + 1] = acc11;
			        }
	            }
	        }
	    }
	}
}
