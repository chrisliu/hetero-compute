CPP_COMPILER = g++
CPP_FLAGS    = -O2 --std=c++11
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

.PHONY: all
all: graphio.exe

graphio.exe: graphio.cpp
	$(CXX) graphio.cpp -o graphio.exe

.PHONY: test
test: graphio.exe
	./graphio.exe 

.PHONY: clean
clean:
	-rm graphio.exe
