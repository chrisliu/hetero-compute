CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11 -arch=sm_60
CROSS_CPP_FLAGS = -fopenmp #-fsanitize=address
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) \
				  	  $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: bfs_verify.exe

bfs_verify.exe: bfs_verify.cu
	$(CXX) bfs_verify.cu -o bfs_verify.exe

.HONY: test
test: bfs_verify.exe
	./bfs_verify.exe ../../graphs/graph_scale23_degree16.sg 

.PHONY: clean
clean:
	-rm bfs_verify.exe
