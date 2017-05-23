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
	int dim = get_work_dim();
	printf("Numer of dimensions in kernel: %d\n", dim);

	// Block Index
	int bi = get_group_id(0);
	int bh = get_group_id(1);
	int bw = get_group_id(2);
	
	// Thread Index
	int ti = get_local_id(0);
	int th = get_local_id(1);
	int tw = get_local_id(2);
	
	// Index
	int i = bi * BLOCK_SIZE + ii;
	int h = bh * BLOCK_SIZE + th;
	int w = bw * BLOCK_SIZE + tw;

	// local C buffer
	__local float C; 
	
	// Assign weight
	C = bias[i];
	
	// Convolution
	
	// RELU

	Cconv[i*IMROW2 + h*IMROW + w] = C;
}
