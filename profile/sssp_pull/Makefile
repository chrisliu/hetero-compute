CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11
CROSS_CPP_FLAGS = -fopenmp
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) -Xcompiler ${CROSS_CPP_FLAGS}

.PHONY: all
all: prof_sssp_pull.exe

prof_sssp_pull.exe: sssp_pull.cu
	$(CXX) sssp_pull.cu -o prof_sssp_pull.exe

.HONY: test
test: prof_sssp_pull.exe
	#./prof_sssp_pull.exe -g 25 -k 16
	./prof_sssp_pull.exe -g 17 -k 16
	#./prof_sssp_pull.exe -g 3 -k 3

.PHONY: clean
clean:
	-rm prof_sssp_pull.exe
