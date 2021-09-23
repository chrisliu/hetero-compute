CPP_COMPILER    = nvcc
CPP_FLAGS       = -O2 --std=c++11 -g -lineinfo
CROSS_CPP_FLAGS = -fopenmp #-fsanitize=address
CXX             = $(CPP_COMPILER) $(CPP_FLAGS) $(addprefix -Xcompiler , ${CROSS_CPP_FLAGS})

.PHONY: all
all: graphgen.exe

graphgen.exe: graphgen.cu
	$(CXX) graphgen.cu -o graphgen.exe

.PHONY: clean
clean:
	-rm graphgen.exe
