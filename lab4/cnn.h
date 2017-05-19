#ifndef _CNN_H_
#define _CNN_H_

#include <math.h>

#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5

float rcmp(float a, float b)
{
	return fabs((a - b) / (a + b));
}

void LoadData(float Cin[NUM][INIMROW][INIMROW], float weight[NUM][NUM][KERNEL][KERNEL],
              float bias[NUM])
{
	fprintf(stderr, "start load input& weight\n");
	FILE *fw, *fb, *fi;
	fw = fopen("weight.bin", "rb");
	fb = fopen("bias.bin", "rb");
	float* t_bias = (float *)malloc(sizeof(float) * NUM);
	float* t_wght = (float *)malloc(sizeof(float) * NUM * NUM * KERNEL * KERNEL);
	fread(t_wght, NUM * NUM * KERNEL * KERNEL, sizeof(float), fw);
	fread(t_bias, NUM, sizeof(float), fb);

	for(int i = 0; i < NUM; i++) {
		bias[i] = t_bias[i];
		for(int j = 0; j < NUM; j++) {
			for(int k = 0; k < KERNEL; k++) {
				for(int s = 0; s < KERNEL; s++)
					weight[i][j][k][s] = t_wght[i * NUM * KERNEL * KERNEL + j * KERNEL * KERNEL + k * KERNEL + s];
			}
		}
	}
	fprintf(stderr, "finish load weight\n");
	free(t_bias);
	free(t_wght);
	fclose(fw);
	fclose(fb);

	float* t_in = (float *)malloc(sizeof(float) * NUM * INIMROW * INIMROW);
	fi = fopen("input.bin", "rb");
	fread(t_in, NUM * INIMROW * INIMROW, sizeof(float), fi);
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < INIMROW; j++) {
			for(int k = 0; k < INIMROW; k++)
				Cin[i][j][k] = (float)t_in[i * INIMROW * INIMROW + j * INIMROW + k];
		}
	}
	fprintf(stderr, "finish load Cin\n");
	free(t_in);
	fclose(fi);
}

int Verify(float Cout[NUM][OUTIMROW][OUTIMROW])
{
	int error = 0;
	FILE *fo;
	fo = fopen("output.bin", "rb");
	float* t_out = (float *)malloc(sizeof(float) * NUM * OUTIMROW * OUTIMROW);
	fread(t_out, NUM * OUTIMROW * OUTIMROW, sizeof(float), fo);
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < OUTIMROW; j++) {
			for(int k = 0; k < OUTIMROW; k++) {
            float in = Cout[i][j][k];
            float ans = t_out[i * OUTIMROW * OUTIMROW + j * OUTIMROW + k];
				if(rcmp(in, ans) > 1e-1){
					error++;
            }
			}
		}
	}
	free(t_out);
	fclose(fo);
	return error;
}

#endif
