#include <stdio.h>
#include <math.h>
#include "cnn.h"


// Sequential CNN implementation
#pragma ACCEL kernel
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW_A][INIMROW_A],
    float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{

  int i,p, q;
  int j, h, w;
  for(i = 0; i < NUM; i++) {

    static float C_tmp[INIMROW_A][INIMROW_A];

    for(h = 0; h < IMROW; h++) {
      for(w = 0; w < IMROW; w++)
        C_tmp[h][w] = bias[i];
    }

    // Convolution
#if PRINT
    if (i % (NUM/20) == 0) {
      printf(".");
      fflush(stdout);
    }
#endif
    for(j = 0; j < NUM; j++) {
      for(h = 0; h < IMROW; h++) {
        for(w = 0; w < IMROW; w++) {
#pragma ACCEL parallel flatten
          for(p = 0; p < KERNEL; p++) {
            for(q = 0; q < KERNEL; q++)
              C_tmp[h][w] += weight[i][j][p][q] * Cin[j][h + p][w + q];
          }

        }
      }
    }

    // ReLU
    for (h = 0; h < IMROW; h++) {
      for (w = 0; w < IMROW; w++) {
        C_tmp[h][w] = fmax(0, C_tmp[h][w]);
      }	
    }

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
  }


#if PRINT
  printf("\n");
#endif

}



