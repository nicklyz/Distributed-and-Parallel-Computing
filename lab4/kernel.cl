#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5
#define NKK 6400
#define IMROW2 50176
#define KK 25
#define INIMROW2 51984
#define BLOCK_SIZE 32

__kernel
void conv(	__global float *Cin,
		__global float *weight,
		__global float *bias,
		__global float *Cconv)
{
   	// Get the work-item's unique ID
	// int dim = get_work_dim();
	// printf("Numer of dimensions in kernel: %d\n", dim);

	// Block Index
	int bi = get_group_id(0);
	int bh = get_group_id(1);
	int bw = get_group_id(2);
	
	// Thread Index
	// int ti = get_local_id(0);
	int th = get_local_id(1);
	int tw = get_local_id(2);
	
	// Index
	int i = bi;
	int h = bh * BLOCK_SIZE + th;
	int w = bw * BLOCK_SIZE + tw;

	// Assign weight
	float C = bias[i];

	__local float Cin_local[BLOCK_SIZE+KERNEL-1][BLOCK_SIZE+KERNEL-1];
	// Convolution
	for (int j = 0; j < NUM; j++) {
		Cin_local[th][tw] = Cin[j*INIMROW2 + h*INIMROW * w];		
		if (tw == BLOCK_SIZE-1 && th == BLOCK_SIZE-1) {
			for (int p=1; p<KERNEL; p++) {
				for (int q=1; q<KERNEL; q++) {
					Cin_local[th+p][tw+q] = 
						Cin[j*INIMROW2 + (h+p)*INIMROW + w+q];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int p = 0; p < KERNEL; p++) {
			for (int q = 0; q < KERNEL; q++) {
				C += weight[i*NKK + j*KK + p*KERNEL + q]
				//	* Cin[j*INIMROW2 + (h+p)*INIMROW + w+q];
					* Cin_local[th+p][tw+q];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	// RELU
	Cconv[i*IMROW2 + h*IMROW + w] = fmax(0.0f, C);
}
