BFS_PROFILE_DIR  = profiles/i7_quadro/bfs/scale23
SSSP_PROFILE_DIR = profiles/i7_quadro/sssp/scale23

.PHONY: all
all: error

.PHONY: bfs
bfs:
	python3 sched.py --algorithm bfs $(BFS_PROFILE_DIR)/*.yaml
	mv bfs.cuh ../../src/kernels/heterogeneous/

.PHONY: sssp
sssp:
	python3 sched.py --algorithm sssp $(SSSP_PROFILE_DIR)/*.yaml
	mv sssp.cuh ../../src/kernels/heterogeneous/

.PHONY: error
error:
	@echo "Please indicate which kernel :)"
