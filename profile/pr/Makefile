CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11 -g
CROSS_CPP_FLAGS = -fopenmp 
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: benchmark_pr.exe

benchmark_pr.exe: benchmark_pr.cu
	$(CXX) benchmark_pr.cu -o benchmark_pr.exe

.HONY: test
test: benchmark_pr.exe
	./benchmark_pr.exe ../../graphs/graph_scale23_degree16.wsg 

.PHONY: move
move: 
	mv *.yaml ../../util/scheduler

.PHONY: clean
clean:
	-rm benchmark_pr.exe
