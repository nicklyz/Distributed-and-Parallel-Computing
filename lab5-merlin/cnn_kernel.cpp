#include <stdio.h>
#include <math.h>
#include "cnn.h"

#define TSI 16
#define TSJ 16
#define TSH 4
#define TSW 4

// Sequential CNN implementation
#pragma ACCEL kernel
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW_A][INIMROW_A],
    float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{

    int i, p, q;
    int j, h, w;
    int ii, jj, hh, ww;
    int i0, j0, h0, w0;

    // for(i = 0; i < NUM; i++) {
    //     static float C_tmp[INIMROW_A][INIMROW_A];
    //
    //     for(h = 0; h < IMROW; h++) {
    //         for(w = 0; w < IMROW; w++) {
    //             C_tmp[h][w] = bias[i];
    //         }
    //     }
    // }

    for (h0 = 0; h0 < IMROW / TSH; h0++) {
        for (w0 = 0; w0 < IMROW / TSW; w0++) {
            for (i0 = 0; i0 < NUM / TSI; i0++) {
                // load output feature maps
                float C_tmp[TSI][TSH][TSW];
                for (ii = 0; ii < TSI; ii++) {
                    i = i0 * TSI + ii;
                    for (hh = 0; hh < TSH; hh++) {
                        for (ww = 0; ww < TSW; ww++) {
                            C_tmp[ii][hh][ww] = bias[i];
                }}}

                for (j0 = 0; j0 < NUM / TSJ; j0++) {
                    // load weights
                    float w_tmp[TSI][TSJ][KERNEL][KERNEL];
                    for (ii = 0; ii < TSI; ii++) {
                        i = i0 * TSI + ii;
                        for (jj = 0; jj < TSJ; jj++) {
                            j = j0 * TSJ + jj;
                            for (p = 0; p < KERNE; p++) {
                                for (q = 0; q < KERNEL; q++) {
                                    w_tmp[ii][jj][p][q] = weight[i][j][p][q];
                    }}}}
                    // load input feature maps
                    float Cin_tmp[TSJ][TSH+KERNEL-1][TSW+KERNEL-1];
                    for (jj = 0; jj < TSJ; jj++) {
                        j = j0 * TSJ + jj;
                        for (hh = 0; hh < TSH+KERNEL-1; hh++) {
                            for (ww = 0; ww < TSH+KERNEL-1; ww++) {
                                Cin_tmp[jj][hh][ww] = Cin[j][hh][ww];
                    }}}

                    // on-chip data computation
                    for (p = 0; p < KERNEL; p++) {
                        for (q = 0; q < KERNEL; q++) {
                            for (hh = 0; hh < TSH; hh++) {
                                for (ww = 0; ww < TSW; ww++) {
                                    for (ii = 0; ii < TSI; ii++) {
                                        for (jj = 0; jj < TSJ; jj++) {
                                            C_tmp[ii][hh][ww] += w_tmp[ii][jj] *
                                                Cin_tmp[jj][hh+p][ww+q];
                    }}}}}}
                }
                // ReLU
                for (ii = 0; ii < TSI; ii++) {
                    for (hh = 0; hh < TSH; hh++) {
                        for (ww = 0; ww < TSW; ww++) {
                            C_tmp[ii][hh][ww] = fmax(0, C_tmp[ii][hh][ww]);
                }}}

                // Max pooling
                for (ii = 0; ii < TSI; ii++) {
                    for (hh = 0; hh < TSH; hh+=2) {
                        for (ww = 0; ww < TSW; ww+=2) {
                            float local_max = C_tmp[hh][ww];
                            local_max = fmax(local_max, C_tmp[hh + 1][ww]);
                            local_max = fmax(local_max, C_tmp[hh + 1][ww + 1]);
                            local_max = fmax(local_max, C_tmp[hh][ww + 1]);
                            Cout[i][hh/2][ww/2] = local_max;
                }}}
                // store output feature maps
    }}}


    //p
    //  q
    //    h
    //      w
    //        i
    //          j
    // for(j0 = 0; j0 < NUM / TSJ; j0++) {
    //   for(h0 = 0; h0 < IMROW/TSH; h0++) {
    //     for(w0 = 0; w0 < IMROW/TSW; w0++) {
    // #pragma ACCEL pipeline
    //         for(hh = 0; hh < TSH; hh++) {
    //     for(ww = 0; ww < TSW; ww++) {
    // 	h = h0 * TSH + hh;
    // 	w = w0 * TSW + ww;
    // 	float sum = 0;
    // #pragma ACCEL parallel flatten
    // 	for(p = 0; p < KERNEL; p++) {
    //       	  for(q = 0; q < KERNEL; q++) {
    //   	              for(jj = 0; jj < TSJ; jj++) {
    //                       j = j0 * TSJ + jj;
    //                       sum += weight[i][j][p][q] * Cin[j][h + p][w + q];
    // 	      }
    // 	    }
    //     	  }
    //   	C_tmp[h][w] += sum;
    //     }
    //   }
    //     }
    //   }
    // }

    // // ReLU
    // for (h = 0; h < IMROW; h++) {
    //   for (w = 0; w < IMROW; w++) {
    //     C_tmp[h][w] = fmax(0, C_tmp[h][w]);
    //   }
    // }

    // Max pooling
    // for (h = 0; h < OUTIMROW; h++) {
    //   for (w = 0; w < OUTIMROW; w++) {
    //     float local_max = C_tmp[2 * h][2 * w];
    //     local_max = fmax(local_max, C_tmp[2 * h + 1][2 * w]);
    //     local_max = fmax(local_max, C_tmp[2 * h + 1][2 * w + 1]);
    //     local_max = fmax(local_max, C_tmp[2 * h][2 * w + 1]);
    //     Cout[i][h][w] = local_max;
    //   }
    // }
    // }


    // #if PRINT
    // printf("\n");
    // #endif

}
