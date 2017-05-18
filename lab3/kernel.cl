#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5
#define NKK 6400
#define IMROW2 50176
#define KK 25
#define INIMROW2 51984

__kernel                                            
void conv(	
	__global float *Cin,
	__global float *weight, 
	__global float *Cconv)                        
{                                                                                               
   	// Get the work-item's unique ID            
   	int i = get_global_id(0);
	//int j = get_global_id(1);
	//printf("CPU: %d, %d\n", i, j);
		
	// Convolution
	/*
	float c_local[IMROW][IMROW];
	for (int h = 0; h < IMROW; h++) {
		for (int w = 0; w < IMROW; w++) {
			c_local[h][w] = Cconv[i*IMROW2 + h*IMROW + w] ;
		}
	}
	*/
	for(int j = 0; j < NUM; j++) {
		// local buffer for w[i][j]
		__local float w_local[KERNEL][KERNEL];
		for (int p = 0; p < KERNEL; p++) {
			for (int q = 0; q < KERNEL; q++) {
				w_local[p][q] = weight[i*NKK + j*KK + p*KERNEL + q];
			}
		}
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w+=4) {
				float4 c4 = vload4(0, &Cconv[i*IMROW2 + h*IMROW + w]);
				for(int p = 0; p < KERNEL; p++) {
					for(int q = 0; q < KERNEL; q++) {
						float4 tmp4 = vload4(0, &Cin[j*INIMROW2 + (h + p)*INIMROW + (w + q)]);
						c4 = c4 + tmp4 * w_local[p][q];
						// c_local[h][w] += w_local[p][q] * 
						//	Cin[j*INIMROW*INIMROW + (h + p)*INIMROW + (w + q)];
					}
				}
				vstore4(c4, 0, &Cconv[i*IMROW2 + h*IMROW + w]);
			}
		}
	}
	
	// write back to Cconv
	/*
	for (int h = 0; h < IMROW; h++) {
		for (int w = 0; w < IMROW; w++) {
			Cconv[i*IMROW*IMROW + h*IMROW + w] = c_local[h][w];
		}
	}
	*/
}                                                   
/*
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < NUM; j++) {
			for(int h = 0; h < IMROW; h++) {
				for(int w = 0; w < IMROW; w++) {
					for(int p = 0; p < KERNEL; p++) {
						for(int q = 0; q < KERNEL; q++)
							C[i][h][w] += weight[i][j][p][q] * Cin[j][h + p][w + q];
					}
				}
			}
		}
	}
*/
