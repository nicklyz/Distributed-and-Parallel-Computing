seq=1
size=1024
np=16

all:
	mpicc -o mmul mmul_main.c mmul.c -DRUN_SEQ=${seq} -lmpi -lm -O3 -fopenmp

run:
	mpirun -np ${np} mmul ${size}

clean:
	rm mmul
