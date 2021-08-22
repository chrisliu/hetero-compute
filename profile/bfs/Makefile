CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11 -g -lineinfo
CROSS_CPP_FLAGS = -fopenmp #-fsanitize=address
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: benchmark_bfs.exe

benchmark_bfs.exe: benchmark_bfs.cu
	$(CXX) benchmark_bfs.cu -o benchmark_bfs.exe

.HONY: test
test: benchmark_bfs.exe
	./benchmark_bfs.exe ../../graphs/graph_scale23_degree16.sg 

.PHONY: clean
clean:
	-rm benchmark_bfs.exe
