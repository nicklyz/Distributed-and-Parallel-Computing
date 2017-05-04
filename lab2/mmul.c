#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  MPI_Bcast(lb, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(A, n * n/pnum, MPI_FLOAT, la, n * n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);

  // computation
  int i, j, k;
  for (i=0; i<n/pnum; i++) {
    for (j=0; j<n; j++)
      lc[i*n+j] = 0;
    for (k=0; k<n; k++) {
      for (j=0; j<n; j++) {
	       lc[i*n+j] += la[i*n+k]*lb[k*n+j];
      }
    }
  }

  printf("%d: Collecting data\n", pid);
  MPI_Gather(lc, n*n/pnum, MPI_FLOAT, C, n*n/pnum, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

}
