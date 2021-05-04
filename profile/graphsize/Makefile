CPP_COMPILER = g++-10
CPP_FLAGS    = --std=c++11 -fopenmp
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

# Name of profiler.
PROF_NAME    = graphsize 

.PHONY: all
all: graphsize.exe

graphsize.exe: graphsize.cpp
	$(CXX) graphsize.cpp -o graphsize.exe

.PHONY: test
test: graphsize.exe
	./graphsize.exe -s -g 7 -k 5 -b graph.wsg

.PHONY: clean
clean:
	-rm graphsize.exe
