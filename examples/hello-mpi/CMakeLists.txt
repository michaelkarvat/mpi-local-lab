mpilab_add_example(hello-mpi
  SOURCES main.c
  MPI
  DESCRIPTION "every rank reports its id and the node it landed on")

# ranks-present rather than invariant: the lines legitimately arrive in any
# order, so what is asserted is that each rank ran exactly once.
mpilab_add_rank_test(hello-mpi RANKS 1 2 4 MODE ranks-present)
