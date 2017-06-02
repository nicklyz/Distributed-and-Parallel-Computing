#ifndef _CNN_H_
#define _CNN_H_

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <time.h>
#include <sys/time.h>

#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5

#define max(a,b) (a>b)?(a):(b)

int load_file_to_memory(const char *filename, char **result)
{ 
    size_t size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) 
        { 
            *result = NULL;
            return -1; // -1 means file opening fail 
        } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) 
        { 
            free(*result);
            return -2; // -2 means file reading fail 
        } 
    fclose(f);
    (*result)[size] = 0;
    return size;
}

float rcmp(float a, float b)
{
	return fabs((a - b) / (a + b));
}

void LoadData(float Cin[NUM][INIMROW][INIMROW], float weight[NUM][NUM][KERNEL][KERNEL],
              float bias[NUM])
{
	fprintf(stderr, "start load input& weight\n");
	FILE *fw, *fb, *fi;
	fw = fopen("/u/cs/class/cs133/cs133ta/release/big/weight.bin", "rb");
	fb = fopen("/u/cs/class/cs133/cs133ta/release/big/bias.bin", "rb");
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
	fi = fopen("/u/cs/class/cs133/cs133ta/release/big/input.bin", "rb");
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
	fo = fopen("/u/cs/class/cs133/cs133ta/release/big/output.bin", "rb");
	float* t_out = (float *)malloc(sizeof(float) * NUM * OUTIMROW * OUTIMROW);
	fread(t_out, NUM * OUTIMROW * OUTIMROW, sizeof(float), fo);
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < OUTIMROW; j++) {
			for(int k = 0; k < OUTIMROW; k++) {
				if(rcmp(Cout[i][j][k], t_out[i * OUTIMROW * OUTIMROW + j * OUTIMROW + k]) > 1e-3)
					error++;
			}
		}
	}
	free(t_out);
	fclose(fo);
	return error;
}

// Sequential CNN implementation
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
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
							C[i][h][w] += weight[i][j][p][q] * Cin[j][h + p][w + q];
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
				Cout[i][h][w] = local_max;
			}
		}
	}
}

#endif
