###############################################################################
# C++ Heterogeneous BFS "compiler"                                            #
#   Generates a heterogeneous BFS kernel given a particular schedule and a    #
#   particular hardware configuration (e.g., 1 CPU 1 GPU, 1 CPU 6 GPU, etc).  #
#   The schedule (kernels for each device) will be hardcoded but the size of  #
#   the input graph will be dynamic.                                          #
###############################################################################
