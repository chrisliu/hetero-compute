CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11 -g
CROSS_CPP_FLAGS = -fopenmp 
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: benchmark_sssp.exe

benchmark_sssp.exe: benchmark_sssp.cu
	$(CXX) benchmark_sssp.cu -o benchmark_sssp.exe

.HONY: test
test: benchmark_sssp.exe
	./benchmark_sssp.exe ../../graphs/graph_scale23_degree16.wsg 

.PHONY: clean
clean:
	-rm benchmark_sssp.exe
