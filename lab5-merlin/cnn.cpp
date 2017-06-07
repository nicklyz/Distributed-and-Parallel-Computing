#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "cnn.h"
#include "cnn_host.h"

void conv(float C_tmp[NUM][INIMROW_A][INIMROW_A], float Cin[NUM][INIMROW_A][INIMROW_A],
          float weight[NUM][NUM][KERNEL][KERNEL]);

int main()
{
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float C_tmp[NUM][INIMROW_A][INIMROW_A];
	static float Cin[NUM][INIMROW_A][INIMROW_A];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	int i, j, h, w;
	// OpenCL host program
	fprintf(stderr, "Start cnn computation\n");
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	// --- Please add your code below ---
	// load bias
	for(i = 0; i < NUM; i++) {
		for(h = 0; h < IMROW; h++) {
      			for(w = 0; w < IMROW; w++)
        			C_tmp[i][h][w] = bias[i];
		}
    	}

	conv(C_tmp, Cin, weight);

	// max pooling
	for (i = 0; i < NUM; i++) { 
		for (h = 0; h < OUTIMROW; h++) {
     			for (w = 0; w < OUTIMROW; w++) {
        			float local_max = C_tmp[i][2 * h][2 * w];
        			local_max = fmax(local_max, C_tmp[i][2 * h + 1][2 * w]);
        			local_max = fmax(local_max, C_tmp[i][2 * h + 1][2 * w + 1]);
        			local_max = fmax(local_max, C_tmp[i][2 * h][2 * w + 1]);
        			Cout[i][h][w] = local_max;
      			}
		}
    	}

	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}
