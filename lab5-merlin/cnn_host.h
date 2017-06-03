
#include "cnn.h"

float rcmp1(float a, float b)
{
	return fabs((a - b));
}


float rcmp(float a, float b)
{
	return fabs((a - b) / (a + b));
}

void LoadData(float Cin[NUM][INIMROW_A][INIMROW_A], float weight[NUM][NUM][KERNEL][KERNEL],
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
	FILE *fo;
	fo = fopen("/u/cs/class/cs133/cs133ta/release/big/output.bin", "rb");
	float* t_out = (float *)malloc(sizeof(float) * NUM * OUTIMROW * OUTIMROW);
	fread(t_out, NUM * OUTIMROW * OUTIMROW, sizeof(float), fo);

	int error = 0;
  printf("Verifying ... \n", error);
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < OUTIMROW; j++) {
			for(int k = 0; k < OUTIMROW; k++) {
				if(rcmp(Cout[i][j][k], t_out[i * OUTIMROW * OUTIMROW + j * OUTIMROW + k]) > 1e-3
				 && rcmp1(Cout[i][j][k], t_out[i * OUTIMROW * OUTIMROW + j * OUTIMROW + k]) > 0.05)
        {
          if (error < 10) 
            printf("%d,%d,%d out=%lf exp=%lf\n", i, j, k,
              Cout[i][j][k], t_out[i * OUTIMROW * OUTIMROW + j * OUTIMROW + k]);
					error++;
        }
			}
		}
	}
	free(t_out);
	fclose(fo);
	return error;
}



