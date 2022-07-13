CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11
CROSS_CPP_FLAGS = -fopenmp 
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: pr_verify.exe

pr_verify.exe: pr_verify.cu
	$(CXX) pr_verify.cu -o pr_verify.exe

.HONY: test
test: rp_verify.exe
	./pr_verify.exe ../../graphs/graph_scale23_degree16.wsg 

.PHONY: clean
clean:
	-rm pr_verify.exe
