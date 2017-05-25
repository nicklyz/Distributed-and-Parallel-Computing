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
#define JBLOCK 8

__kernel
void conv(	__global float *Cin,
		__global float *weight,
		__global float *bias,
		__global float *Cconv)
{
   	// Get the work-item's unique ID
	int i = get_global_id(0);
	int h = get_global_id(1);
	int w = get_global_id(2);

	int bi = get_group_id(0);
	int bh = get_group_id(1);
	int bw = get_group_id(2);
	
	// Thread Index
	// int ti = get_local_id(0);
	int th = get_local_id(1);
	int tw = get_local_id(2);
	int local_id = th * BLOCK_SIZE + tw;

	// Assign weight
	float C = bias[i];

	__local float Cin_local[JBLOCK][BLOCK_SIZE+KERNEL-1][BLOCK_SIZE+KERNEL-1];
	__local float w_local[JBLOCK][KERNEL][KERNEL];

	// Convolution
	for (int j = 0; j < NUM; j+=JBLOCK) {
		for (int jj = 0; jj < JBLOCK; jj++) {
			// read first 32 * 32 elements from 36 * 36 Kernel block
			int hh = local_id / 36;
			int ww = local_id % 36;
			int j0 = j + jj;
			Cin_local[jj][hh][ww] = 
				Cin[j0*INIMROW2 + (bh * BLOCK_SIZE + hh) *INIMROW + (bw * BLOCK_SIZE + ww)];
		}
 		
		// read the rest of the elements
		if (local_id < 272) {
			for (int jj = 0; jj < JBLOCK; jj++) {
				int hh = (local_id + 1024) / 36;
				int ww = (local_id + 1024) % 36;
				int j0 = j + jj;
				Cin_local[jj][hh][ww] = 
					Cin[j0*INIMROW2 + (bh * BLOCK_SIZE + hh) *INIMROW + (bw * BLOCK_SIZE + ww)];
			}
		}
		if (th < 5 && tw < 5) {
			for (int jj = 0; jj < JBLOCK; jj++) {
				int j0 = j + jj;
				w_local[jj][th][tw] = weight[i*NKK + j0*KK + th*KERNEL + tw];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int jj = 0; jj < JBLOCK; jj++) {
			for (int p = 0; p < KERNEL; p++) {
				for (int q = 0; q < KERNEL; q++) {
					C += w_local[jj][p][q] * Cin_local[jj][th+p][tw+q];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	// RELU
	Cconv[i*IMROW2 + h*IMROW + w] = fmax(0.0f, C);
}
