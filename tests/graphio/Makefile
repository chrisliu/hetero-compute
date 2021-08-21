CPP_COMPILER = nvcc
CPP_FLAGS    = -O2 --std=c++11
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

.PHONY: all
all: graphio.exe

graphio.exe: graphio.cu
	$(CXX) graphio.cu -o graphio.exe

.PHONY: test
test: graphio.exe
	./graphio.exe 

.PHONY: clean
clean:
	-rm graphio.exe
