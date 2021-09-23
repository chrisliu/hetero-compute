# All About Graphs (not really!)

*Updated: September 23, 2021*

A guide to Kronecker graphs and the technical implementation in this project.

## Social Network Graphs & Kronecker Graphs
Kronecker graphs are synthetic graphs designed to resemble social network graphs. 

![Power Law Distribution](https://mathinsight.org/media/image/image/power_law_degree_distribution_scatter.png)

These graphs are characterized by the power law distribution. That means there are a lot of nodes with very low degrees and few nodes with very high degrees. The above image demonstrates this (note it's in log scale). Imagine the number of Twitter followers a celebrity like Obama has compared to Chris (none in this case :) ).

To generate Kronecker graphs, we are piggybacking off of Scott Beamer's [GAP Benchmark Suite](https://github.com/sbeamer/gapbs/) which uses a R-MAT generator with the parameters supplied by the [Graph500 specification](https://graph500.org/?page_id=12). We only care about two things, the scale of the graph `scale` and the average degree `edgefactor`. Our generated graph will have `2^{scale}` number of nodes and `2^{scale} * edgefactor` number of edges. 

## Generating Kronecker Graphs
The graph generator could be found under [util/graphgen](https://github.com/chrisliu/hetero-compute/tree/master/util/graphgen). Modify [util/graphgen/graphgen.cpp](https://github.com/chrisliu/hetero-compute/blob/master/util/graphgen/graphgen.cpp) to generated a serial graph `.sg` and/or a weighted serial graph `.wsg` by toggling the `UNWEIGHTED_ON` and `WEIGHTED_ON` definitions respectively.

To compile the utility, run `make`.  To generate the unweighted and/or weighted graphs, run `./graphgen.exe -g <scale> -k <edgefactor>` (note for Graph500, `edgefactor=16`). Optionally, remove the utility with `make clean`.

## Data Format of Serial Graphs and Weighted Serial Graphs

### CSR Format
A graph consists of nodes and neighbors. The graphs typically represented in [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix), a sparse-matrix format. There's two arrays, the index array and the neighbors array. The index array indicates a particular node's neighbors should be accessed. The neighbors array contains the neighbor node id and other relevant information (e.g., edge weight).

![Example Graph](https://web.cecs.pdx.edu/~sheard/course/Cs163/Graphics/graph6.png)

In this example, the index and neighbors array look like this. Notice how the size of the index array is `# nodes + 1`.

```
index     = [0, 2, 6, 8, 11, 14]
neighbors = [2, 4, 1, 3, 4, 5, 2, 5, 1, 2, 5, 2, 3, 4]
```

To get the neighbors of node `u`, we need to iterate through the neighbors array from `index[u]` to `index[u + 1]` (exclusive). To access  node `u=2`, it would be 

```
for (nid_t v = index[2]; v < index[2 + 1]; v++)
    do something with neighbors[v]
```

For a weighted graph, the index array would be the same but the neighbors array may look something like this (where each tuple is `(neighbor node id, edge weight`).

```
neighbors = [(2, 2), (4, 5), (1, 2), (3, 14), (4, 5), (5, 4), (2, 14), (5, 34), ... , (4, 58)]
```

### Serial Graphs
The index array for serial graphs is represented by `offset_t` and the neighbors array is represented by `nid_t`. To serialize this graph, perform this.

```
write(# of nodes, sizeof(nid_t))
write(len(neighbors array), sizeof(offset_t))
write(index array, (# of nodes + 1) * sizeof(offset_t))
write(neighbors array, len(neighbors array) * sizeof(nid_t))
```

### Weighted Serial Graphs
Weighted serial graphs are very similar to serial graphs. The index array for serial graphs is represented by `offset_t` and the neighbors array is represented by `wnode_t`. To serialize this graph, perform this.

```
write(# of nodes, sizeof(nid_t))
write(len(neighbors array), sizeof(offset_t))
write(index array, (# of nodes + 1) * sizeof(offset_t))
write(neighbors array, len(neighbors array) * sizeof(wnode_t))
```
