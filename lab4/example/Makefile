CC=gcc

EXEDIR	= .

VECADD_EXEC = vecadd_gpu

LINK	 = -O3 -g -lm -I$(OPENCL_ROOT)/include -L$(OPENCL_ROOT)/lib64 -lOpenCL

all: $(EXEDIR)/$(VECADD_EXEC) 

$(EXEDIR)/$(VECADD_EXEC): main.c vecadd.cl vecadd_cl.h
	$(CC) -o $(EXEDIR)/$(VECADD_EXEC) main.c $(LINK)

vecadd_cl.h: vecadd.cl
	../cl_trans.py vecadd.cl

clean:
	rm -rf $(EXEDIR)/$(VECADD_EXEC) $(EXEDIR)/vecadd_cl.h
