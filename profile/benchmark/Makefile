CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11
CROSS_CPP_FLAGS = -fopenmp
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) -Xcompiler ${CROSS_CPP_FLAGS}

.PHONY: all
all: prof_sssp_pull.exe

prof_sssp_pull.exe: sssp_gpu.cu
	$(CXX) sssp_gpu.cu -o prof_sssp_gpu.exe

.HONY: test
test: prof_sssp_pull.exe
	./prof_sssp_gpu.exe -g 25 -k 16
	#./prof_sssp_gpu.exe -g 20 -k 16
	#./prof_sssp_gpu.exe -g 3 -k 3

.PHONY: clean
clean:
	-rm prof_sssp_pull.exe
