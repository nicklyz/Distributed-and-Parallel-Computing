#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BS 64
// Please modify this function

void mmul(float *A, float *B, float *C, int n, int pid, int pnum)
{
	float* la = (float*) malloc( sizeof(float) * n * n/pnum );
	float* lb = (float*) malloc( sizeof(float) * n * n);
	float* lc = (float*) malloc( sizeof(float) * n * n/pnum );

	if ( pid == 0 )
	{
		lb = B;
	}

	// broadcast matrix lb to all processors
	printf("%d: Sending data\n", pid);
	// Broadcast is not scalable for larger B
	MPI_Bcast(lb, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(A, n * n/pnum, MPI_FLOAT, la, n * n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// computation
	int i, j, k, jj, kk;
	for (i=0; i<n/pnum; i++) 
	for (j=0; j<n; j++)
		lc[i*n+j] = 0;

	for (j=0; j<n/BS; ++j)
	{
		for (k=0; k<n/BS; ++k)
		{
			for(i=0; i<n/pnum; ++i)
			{
				for (kk=0; kk<BS; kk++) 
				{
					for (jj=0; jj<BS; jj++) 
					{
						lc[i*n+j*BS+jj] += la[i*n+k*BS+kk] * lb[k*BS*n+kk*n+j*BS+jj];
					}
				}
			}
		}
	}
	
	printf("%d: Collecting data\n", pid);
	MPI_Gather(lc, n*n/pnum, MPI_FLOAT, C, n*n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

}
