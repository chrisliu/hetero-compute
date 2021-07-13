all: generate

generate:
	python3.8 sched.py profiles/scale23/*
	mv sssp_pull.cuh ../../src/kernels/heterogeneous/
