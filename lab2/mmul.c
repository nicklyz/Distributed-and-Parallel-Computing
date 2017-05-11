#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BS 64

void mmul(float *A, float *B, float *C, int n)
{
	int pnum, pid;
	MPI_Comm_size(MPI_COMM_WORLD, &pnum);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	float* la = (float*) malloc( sizeof(float) * n * n/pnum );
	float* lb;
	float* lc = (float*) malloc( sizeof(float) * n * n/pnum );

	if ( pid == 0 )
	{
		lb = B;
	} 
	else 
	{
		lb = (float*) malloc( sizeof(float) * n * n);
	}
	// broadcast matrix lb to all processors
	// printf("%d: Sending data\n", pid);
	// Broadcast is not scalable for larger B
	MPI_Bcast(lb, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(A, n * n/pnum, MPI_FLOAT, la, n * n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// computation
	int i, j, k, ii, jj, kk;
	// init lc to zeros
	memset(lc, 0, sizeof(float) * n * n / pnum);
	float ra, r0, r1, r2, r3;
	
	for (i=0; i<n/pnum; i+=BS)
	{
		for (j=0; j<n; j+= BS)
		{
			// local buffer for C
			float buf[BS][BS];
			memset(buf, 0, sizeof(float) * BS * BS);
			
			for (k=0; k<n; k+=BS)
			{
				// local buffer for A and B
				float Abuf[BS][BS];
				float Bbuf[BS][BS];
				for (ii=0; ii< (BS>(n/pnum) ? n/pnum : BS); ii++)
				{
					for (jj=0; jj<BS; jj++)
					{
						Abuf[ii][jj] = la[(i+ii)*n + k+jj];
						Bbuf[ii][jj] = lb[(k+ii)*n + j+jj];
					}
				}
				// computation; 4 multiplication at a time
				for (ii=0; ii<(BS>(n/pnum) ? n/pnum : BS); ii++)
				{
					for (kk=0; kk<BS; kk++)
					{
						ra=Abuf[ii][kk];
						for (jj=0; jj<BS; jj+=4)
						{
							r0 = Bbuf[kk][jj];
							r1 = Bbuf[kk][jj+1];
							r2 = Bbuf[kk][jj+2];
							r3 = Bbuf[kk][jj+3];
							r0 *= ra;
							r1 *= ra;
							r2 *= ra;
							r3 *= ra;
							buf[ii][jj] += r0;
							buf[ii][jj+1] += r1;
							buf[ii][jj+2] += r2;
							buf[ii][jj+3] += r3; 
						}
					}
				}
			}
			// put accumulator values back to lc
			for (ii=0; ii<(BS > n/pnum ? n/pnum : BS); ii++)
			{
				for (jj=0; jj<BS; jj++)
				{
					lc[(i+ii)*n+j+jj] = buf[ii][jj];
				}
			}
		}
	}
	// printf("%d: Collecting data\n", pid);
	MPI_Gather(lc, n*n/pnum, MPI_FLOAT, C, n*n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	free(la);
	if (pid != 0)
	{
		free(lb);
	}
	free(lc);
}
