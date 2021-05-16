CPP_COMPILER = g++
CPP_FLAGS    = -O2 --std=c++11
CXX          = $(CPP_COMPILER) $(CPP_FLAGS)

.PHONY: all
all: gapbs.exe

gapbs.exe: gapbs.cpp
	$(CXX) gapbs.cpp -o gapbs.exe

.PHONY: test
test: gapbs.exe
	./gapbs.exe

.PHONY: clean
clean:
	-rm gapbs.exe
