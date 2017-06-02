#!/bin/bash

source /usr/local/cs133/FCS/Merlin_Compiler_2017.1/settings64.sh 
export LM_LICENSE_FILE=/usr/local/cs133/FCS/license/falcon_cs133.lic

lic_file="xilinx_$( cut -d '.' -f 1 <<< `hostname` ).lic"
export LM_LICENSE_FILE=/usr/local/cs133/Xilinx/SDx/license/${lic_file}:${LM_LICENSE_FILE}

