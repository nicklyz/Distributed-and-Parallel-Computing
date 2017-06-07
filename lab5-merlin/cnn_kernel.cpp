#include <stdio.h>
#include <math.h>
#include "cnn.h"

#define TSI 16
#define TSJ 16
#define TSH 16
#define TSW 16

// Sequential CNN implementation
#pragma ACCEL kernel
void conv(float C_tmp[NUM][INIMROW_A][INIMROW_A], float Cin[NUM][INIMROW_A][INIMROW_A],
    float weight[NUM][NUM][KERNEL][KERNEL])
{

  int i, p, q;
  int j, h, w;
  int i0, j0, h0, w0;
  int ii, jj, hh, ww;

/*
    static float C_tmp[INIMROW_A][INIMROW_A];

    for(h = 0; h < IMROW; h++) {
      for(w = 0; w < IMROW; w++)
        C_tmp[h][w] = bias[i];
    }
*/
    // Convolution
  for (h0 = 0; h0 < IMROW / TSH; h0++) {
    for (w0 = 0; w0 < IMROW / TSW; w0++) {
      for (p = 0; p < KERNEL; p++) {
	for (q = 0; q < KERNEL; q++) {
#pragma ACCEL pipeline
	for (j = 0; j < NUM; j++) {
	for (i = 0; i < NUM; i++) {
	  // on-chip computation
	  //for (p = 0; p < KERNEL; p++) {
	    //for (q = 0; q < KERNEL; q++) {
#pragma ACCEL parallel flatten
	      for (hh = 0; hh < TSH; hh++) {
	        for (ww = 0; ww < TSW; ww++) {
		  h = h0 * TSH + hh;
		  w = w0 * TSW + ww;
	          //for (i = 0; i < NUM; i++) {
		    //for (j = 0; j < NUM; j++) {
		      C_tmp[i][h][w] += weight[i][j][p][q] * Cin[j][h+p][w+q];
	  }}}}}}
  }}
// }}

/*
  for(i = 0; i < NUM; i++) {
    for(j = 0; j < NUM; j++) {
      for(h = 0; h < IMROW; h++) {
        for(w = 0; w < IMROW; w++) {
	  	float acc = 0;
#pragma ACCEL parallel flatten
		for(p = 0; p < KERNEL; p++) {
          	  for(q = 0; q < KERNEL; q++) {
          	    acc += weight[i][j][p][q] * Cin[j][h + p][w + q];
          	  }
          	}
	  	C_tmp[i][h][w] += acc;
	}
      }
    }
  }
*/
    // ReLU
  for (i = 0; i < NUM; i++) {
    for (h = 0; h < IMROW; h++) {
      for (w = 0; w < IMROW; w++) {
        C_tmp[i][h][w] = fmax(0, C_tmp[i][h][w]);
      }	
    }
  }
/*
    // Max pooling
    for (h = 0; h < OUTIMROW; h++) {
      for (w = 0; w < OUTIMROW; w++) {
        float local_max = C_tmp[2 * h][2 * w];
        local_max = fmax(local_max, C_tmp[2 * h + 1][2 * w]);
        local_max = fmax(local_max, C_tmp[2 * h + 1][2 * w + 1]);
        local_max = fmax(local_max, C_tmp[2 * h][2 * w + 1]);
        Cout[i][h][w] = local_max;
      }
    }
*/


#if PRINT
  printf("\n");
#endif

}



