CPP_COMPILER = g++
CPP_FLAGS    = -O2 --std=c++11 -fopenmp
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

.PHONY: all
all: graphgen.exe

graphgen.exe: graphgen.cpp
	$(CXX) graphgen.cpp -o graphgen.exe

.PHONY: clean
clean:
	-rm graphgen.exe
