CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11
CROSS_CPP_FLAGS = -fopenmp 
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: sssp_verify.exe

sssp_verify.exe: sssp_verify.cu
	$(CXX) sssp_verify.cu -o sssp_verify.exe

.HONY: test
test: sssp_verify.exe
	./sssp_verify.exe ../../graphs/graph_scale23_degree16.wsg 

.PHONY: clean
clean:
	-rm sssp_verify.exe
