# Scheduler

*Updated: September 23, 2021*

The scheduler could be found under [util/scheduler](https://github.com/chrisliu/hetero-compute/tree/master/util/scheduler).

## Requirements
- Python 3.8+
	- PyYAML

## Generating a Heterogeneous Kernel
**Subject to future change.** I intend to add [subparsers](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers) for each kernel. For the latest instructions, use `python3 sched.py --help`. An example usage could also be found in the Makefile in the directory.

## BFS
**Currently a work in progress.**

Run `python3 sched.py --algorithm bfs <benchmark YAML files>`. The heterogeneous kernel will be emitted as `bfs.cuh` in the same directory.

### Advanced Configuration
For advanced configurations, you must edit [util/scheduler/scheduler/kernelgen/bfs_hetero.py](https://github.com/chrisliu/hetero-compute/blob/master/util/scheduler/scheduler/kernelgen/bfs_hetero.py).

## SSSP
Run `python3 sched.py --algorithm sssp <benchmark YAML files>`. The heterogeneous kernel will be emitted as `sssp.cuh` in the same directory.

### Advanced Configuration
For now, advanced configurations requires [util/scheduler/scheduler/kernelgen/sssp_hetero.py](https://github.com/chrisliu/hetero-compute/blob/master/util/scheduler/scheduler/kernelgen/sssp_hetero.py) to be edited.

Available options:

 - `INTERLEAVE = {True, False}`: Enable or disable compute and memory operation overlap. If `True`,  memory communication if will be `O(n^2)`; else, the butterfly transfer pattern will be performed with a communication cost of `O(n logn)`.
 - `HIGH_DEGREE_FIRST = {True, False}`: Determines if segments should be computed from highest degree to lowest degree or vice versa.
