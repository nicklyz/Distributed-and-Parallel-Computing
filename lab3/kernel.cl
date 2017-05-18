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
void conv(__global float *Cin,
		 __global float *weight, 
		 __global float *Cconv)                        
{                                                                                               
   	// Get the work-item's unique ID            
   	int i = get_global_id(0);
		
	// Convolution
	float c_local[IMROW][IMROW];
	for (int h = 0; h < IMROW; h++) {
		for (int w = 0; w < IMROW; w++) {
			c_local[h][w] = Cconv[i*IMROW*IMROW + h*IMROW + w] ;
		}
	}
	
	for(int j = 0; j < NUM; j++) {
		// local buffer for w[i][j]
		__local float w_local[KERNEL][KERNEL];
		for (int p = 0; p < KERNEL; p++) {
			for (int q = 0; q < KERNEL; q++) {
				w_local[p][q] = weight[i*NKK + j*KK + p*KERNEL + q];
			}
		}
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++) {
				__private float tmp;
				tmp = c_local[h][w];
				for(int p = 0; p < KERNEL; p++) {
					for(int q = 0; q < KERNEL; q++) {
						tmp = tmp + w_local[p][q] * 
							Cin[j*INIMROW*INIMROW + (h + p)*INIMROW + (w + q)];
					}
				}
				c_local[h][w] = tmp;
			}
		}
	}
	
	// write back to Cconv
	for (int h = 0; h < IMROW; h++) {
		for (int w = 0; w < IMROW; w++) {
			Cconv[i*IMROW*IMROW + h*IMROW + w] = c_local[h][w];
		}
	}
}
                                                  
