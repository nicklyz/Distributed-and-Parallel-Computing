#define KERNEL 5
#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112

#define weight(i,j,p,q) weight[(i)*NUM*KERNEL*KERNEL + (j)*KERNEL*KERNEL + (p)*KERNEL + (q)]
#define Cin(j,h,w) Cin[(j)*INIMROW*INIMROW+(h)*INIMROW+(w)]
#define Cout(i,h,w) Cout[(i)*OUTIMROW*OUTIMROW+(h)*OUTIMROW+(w)]

#define max(a,b) (a>b)?(a):(b)

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void cnn_kernel(
  __global float* Cin,
  __global float* weight,
  __global float* bias,
  __global float* Cout
){
	static float C[NUM][IMROW][IMROW];

	for(int i = 0; i < NUM; i++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++)
				C[i][h][w] = bias[i];
		}
	}

// Convolution
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < NUM; j++) {
			for(int h = 0; h < IMROW; h++) {
				for(int w = 0; w < IMROW; w++) {
					for(int p = 0; p < KERNEL; p++) {
						for(int q = 0; q < KERNEL; q++)
							C[i][h][w] += weight(i,j,p,q) * Cin(j,h + p,w + q);
					}
				}
			}
		}
	}

// ReLU
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < IMROW; h++) {
			for (int w = 0; w < IMROW; w++) {
				C[i][h][w] = max(0, C[i][h][w]);
			}	
		}
	}

// Max pooling
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < OUTIMROW; h++) {
			for (int w = 0; w < OUTIMROW; w++) {
				float local_max = C[i][2 * h][2 * w];
				local_max = max(local_max, C[i][2 * h + 1][2 * w]);
				local_max = max(local_max, C[i][2 * h + 1][2 * w + 1]);
				local_max = max(local_max, C[i][2 * h][2 * w + 1]);
				Cout(i,h,w) = local_max;
			}
		}
	}
}
