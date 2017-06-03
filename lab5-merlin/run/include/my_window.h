/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     (c) Copyright 2012-2013 Xilinx Inc.
 *     All rights reserved.
 *
 *****************************************************************************/

/*
 * HLS Video Memory Partition Header File
 */

#ifndef ___HLS__VIDEO_MEM__
#define ___HLS__VIDEO_MEM__

//#include "hls_video_core.h"

namespace hls {

/* Template class of Window */
template<int ROWS, int COLS, typename T>
class Window {
public:
    Window() {
#pragma HLS ARRAY_PARTITION variable=val dim=0 complete
    };
    void shift_right(){};
    void shift_left(){};
    void shift_up(){};
    void shift_down(){};
    void insert(T value, int row, int col){};
    void insert_bottom(T value[COLS]){};
    void insert_top(T value[COLS]){};
    void insert_left(T value[ROWS]){};
    void insert_right(T value[ROWS]){};
    T getval(int row, int col){};

    T val[ROWS][COLS];
};


} // namespace hls

#endif
