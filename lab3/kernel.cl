#define NUM 256
#define INIMROW 228
#define IMROW 224
#define OUTIMROW 112
#define KERNEL 5

#define PBLOCK 256
#define BLOCK 32

__kernel                                            
void cnn(__global float *Cin,                        
         __global float *weight,                        
         __global float *bias,
	 __global float *Cout)                        
{                                                                                               
   	// Get the work-item's unique ID            
   	int r = get_global_id(0);
	int c = get_global_id(1);
	printf("r:%d, c:%d\n", r, c);
	// Get the global size
	// int global_size = get_global_size(0);
	// printf("global size: %d\n", global_size);
	// Get local size
	int local_size = get_local_size(0);
	// printf("local size: %d\n", local_size);
	// get local id
	int local_id = get_local_id(0);
	// printf("local id: %d\n", local_id);
	
}                                                   

