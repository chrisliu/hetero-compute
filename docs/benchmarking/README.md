# Benchmarking

*Updated: September 23, 2021*

The same code drives the micro-benchmarks and the Graph500 official benchmark.

## Micro-Benchmarks
In order to figure out which device+kernel combination works best, we must try everything out.

### Key Terms
1. Segment: Each graph is ordered by descending degree (i.e., highest degree to lowest degree). For the benchmarking purposes, the graph is divided into `n` segments such that each roughly has an equal number of edges.
2. GTEPs: **G**iga **T**raversed **E**dges **P**er **s**econd. The higher this number, the better!

### Getting Benchmark Results
#### BFS
Go to [profile/bfs](https://github.com/chrisliu/hetero-compute/tree/master/profile/bfs).  In [profile/bfs/benchmark_bfs.cu](https://github.com/chrisliu/hetero-compute/blob/master/profile/bfs/benchmark_bfs.cu), you may configure the number of segments `NUM_SEGMENTS` and the number of GPU blocks `NUM_BLOCKS` to use. Disable `RUN_FULL_KERNELS` if you only want to get micro-benchmark results.

Run `make`. `make test` will run a graph stored under `hetero-compute/graphs/graph_scale23_degree16.sg`. To use your own graph, run `./benchmark_bfs.exe <your graph here>`. Optionally, run `make clean` to delete the compiled binary.

The resulting micro-benchmark results will be emitted in the same folder as named `.yaml` files.

#### SSSP
Go to [profile/sssp](https://github.com/chrisliu/hetero-compute/tree/master/profile/sssp).  In [profile/bfs/benchmark_sssp.cu](https://github.com/chrisliu/hetero-compute/blob/master/profile/bfs/benchmark_sssp.cu), you may configure the number of segments `NUM_SEGMENTS` and the number of GPU blocks `NUM_BLOCKS` to use. Enable the `RUN_EPOCH_KERNELS` flag. Disable `RUN_FULL_KERNELS` if you only want to get micro-benchmark results.

Run `make`. `make test` will run a graph stored under `hetero-compute/graphs/graph_scale23_degree16.sg`. To use your own graph, run `./benchmark_sssp.exe <your graph here>`. Optionally, run `make clean` to delete the compiled binary.

The resulting micro-benchmark results will be emitted in the same folder as named `.yaml` files.

## Official Graph500 Benchmarks
Follow the steps listed under [Getting Benchmark Results](#getting-benchmark-results) to get the appropriate `benchmark.cu` file. Enable the `RUN_FULL_KERNELS` flag and disable the `RUN_EPOCH_KERNELS` flag. Make sure the heterogeneous kernel emitted by the "compiler" is copied (with the same file name) over to [src/kernels/heterogeneous](https://github.com/chrisliu/hetero-compute/tree/master/src/kernels/heterogeneous).
