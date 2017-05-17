#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5

#define PBLOCK 256
#define BLOCK 32

__kernel                                            
void conv(__global float *Cin,
		 __global float *weight, 
		 __global float *Cconv)                        
{                                                                                               
   	// Get the work-item's unique ID            
   	int i = get_global_id(0);
	//int j = get_global_id(1);
	//printf("CPU: %d, %d\n", i, j);
		
// Convolution
	float c_local[IMROW][IMROW];
	for (int h = 0; h < IMROW; h++) {
		for (int w = 0; w < IMROW; w++) {
			c_local[h][w] = Cconv[i*IMROW*IMROW + h*IMROW + w] ;
		}
	}
	for(int j = 0; j < NUM; j++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++) {
				for(int p = 0; p < KERNEL; p++) {
					for(int q = 0; q < KERNEL; q++)
						c_local[h][w] += 
							weight[i*NUM*KERNEL*KERNEL +j*KERNEL*KERNEL + p*KERNEL + q] * 
							Cin[j*INIMROW*INIMROW + (h + p)*INIMROW + (w + q)];
				}
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
