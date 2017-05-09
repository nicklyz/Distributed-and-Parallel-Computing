CC=gcc

EXEC = cs133_lab3

LINK = -std=c99 -D_BSD_SOURCE -O3 -lm -I$(OPENCL_ROOT)/include -L$(OPENCL_ROOT)/lib64 -lOpenCL

SRC = cnn.c

all: $(EXEC) 

$(EXEC): $(SRC)
	$(CC) -o $(EXEC) $(SRC) $(LINK)

kernel: kernel.cl
	./cl_trans.py kernel.cl

clean:
	rm -rf $(EXEC) kernel_cl.h
