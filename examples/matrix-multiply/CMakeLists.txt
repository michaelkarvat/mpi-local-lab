mpilab_add_example(matrix-multiply
  SOURCES main.c
  MPI
  OPENMP
  DESCRIPTION "row-block C = A x B, MPI between nodes and OpenMP inside one")

# 3 ranks does not divide the default size, and the sweep is what proves the
# Gatherv reassembled the rows in input order rather than arrival order.
mpilab_add_rank_test(matrix-multiply RANKS 1 2 3 4 MODE invariant ARGS 96)
