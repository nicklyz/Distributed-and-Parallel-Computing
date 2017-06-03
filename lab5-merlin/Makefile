init: cnn.cpp cnn_kernel.cpp
	merlin_flow run

sim: run/implement
	merlin_flow run sim

hls:
	merlin_flow run hls

gen:
	merlin_flow run impl

clean:
	rm -rf run/pkg
	rm -rf run/implement
	rm -rf run/merlin-sim.log
	rm -rf run/report
	rm -rf run/.merlin*
	rm -rf run/directive.xml
