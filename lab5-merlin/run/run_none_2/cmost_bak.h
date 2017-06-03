/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

//#ifndef _CMOST_H_INCLUDED_
//#define _CMOST_H_INCLUDED_

//#include <stdio.h>
//#include <stdlib.h>
//#include <assert.h>
//#include <math.h>
#include<stddef.h>

//extern void * malloc(size_t size);
//extern void free(void*) __THROW ;
//__BEGIN_NAMESPACE_STD
//extern void free (void *__ptr) __THROW;
//__END_NAMESPACE_STD


//#include "cmost_test_common.h"
//#include "cl_platform.h"

#define CMOST_CC
#define CMOSTCC 


#define FPGA_DDR_USER_BASE (0xc0000000)


int cmost_malloc(void ** a, size_t size );
int cmost_malloc_1d(void ** a, const char * filename, int unit_size, size_t d0 );
int cmost_malloc_2d(void *** a, const char * filename, int unit_size, size_t d0, size_t d1 );
int cmost_malloc_3d(void **** a, const char * filename, int unit_size, size_t d0, size_t d1, size_t d2 );
int altera_malloc(void ** a, const char * filename, int unit_size, size_t d0 );

int cmost_free_1d(void * a);
int cmost_free_2d(void ** a);
int cmost_free_3d( void *** a);

int cmost_dump_1d(void * a, const char * filename);
int cmost_dump_2d(void ** a, const char * filename);
int cmost_dump_3d(void *** a, const char * filename);
void cmost_break_point();

void cmost_write_file(void * buffer, const char * file_name, size_t size);
void cmost_load_file(void * buffer, const char * file_name, size_t size);

void  cmost_start_timer(int i);
void  cmost_read_timer_new(int i, float * sec);   // return the time in seconds

int cmost_free(void * a);
int cmost_dump(void * a, const char * filename);

void __merlin_access_range(
#ifdef __cplusplus
    ...
#endif
    );
//#endif //_CMOST_H_INCLUDED_


