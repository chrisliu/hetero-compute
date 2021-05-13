CPP_COMPILER = g++-10
CPP_FLAGS    = -O2 --std=c++11 -fopenmp
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

# Name of profiler.
PROF_NAME    = graphsize 

.PHONY: all
all: prof_sssp_pull.exe

prof_sssp_pull.exe: ../../src/sssp_pull.cpp
	$(CXX) ../../src/sssp_pull.cpp -o prof_sssp_pull.exe

.PHONY: test
test: prof_sssp_pull.exe
	./prof_sssp_pull.exe -g 20 -k 16
	#./prof_sssp_pull.exe -g 3 -k 3

.PHONY: clean
clean:
	-rm prof_sssp_pull.exe
