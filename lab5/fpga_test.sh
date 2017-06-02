#!/bin/bash
echo "Setup the environment..."
TEST_DIR=/u/cs/class/cs133/cs133ta/release/test
source ${TEST_DIR}/setup_oclfpga.sh

hostName=$(hostname)
if [ "$hostName" != "lnxsrv08.seas.ucla.edu" ]; then
	echo "ERROR: You are NOT on the CS133 server and no FPGA card can be used."
	exit 1
fi

echo "Testing FPGA..."
mkdir tmp
cp ${TEST_DIR}/* tmp/
cd tmp
./verify.exe ./verify.xclbin > err.log
rm sdaccel_profile_summary.csv sdaccel_profile_summary.html
res=`grep "Hello World" err.log`
if [ "$res" != "Hello World" ]; then
	echo "ERROR: FPGA board seems not working, or someone else is using it."
	echo "Please find details in err.log."
	exit 1
fi
echo "FPGA is working fine!"
rm err.log
cd ..
rm -rf tmp
