# Documentation

## Toolflow
Getting a kernel running from start to finish.
1. Generate/Convert graph into serial graph `.sg` or weighted serial graph `.wsg` for BFS and SSSP respectively.
	- Expected Input: Kronecker graph parameters ~~or existing graph~~.
	- Expected Output: serial graph `.sg` (all edge weights are `1`) or weighted serial graph `.wsg` (edges have defined weights).
	- More info in [graph_info](https://github.com/chrisliu/hetero-compute/tree/master/docs/graph_info).
2. Get benchmark results for each segment (and each epoch for BFS).
	- Expected Input: serial graph `.sg` or weighted serial graph `.wsg`.
	- Expected Output: profiles for each device for each kernel in `.yaml` format.
	- More info in [benchmarking](https://github.com/chrisliu/hetero-compute/tree/master/docs/benchmarking).
3. Create a heterogeneous kernel based on the benchmark results.
	- Expected Input: profiles for each device for each kernel in `.yaml` format.
	- Expected Output: "compiled" `<your kernel>_hetero.cuh` (heterogeneous kernel).
	- More info in [scheduling](https://github.com/chrisliu/hetero-compute/tree/master/docs/scheduling).
4. Run results based on the official Graph500 specification.
	- Expected Input: serial graph `.sg` or weighted serial graph `.wsg`.
	- Expected Output: official GTEPs numbers.
	- More info in [benchmarking](https://github.com/chrisliu/hetero-compute/tree/master/docs/benchmarking).
