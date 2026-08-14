mpilab_add_example(distributed-sum
  SOURCES main.c
  MPI
  DESCRIPTION "MPI_Scatterv a block decomposition, combine with MPI_Reduce")

# The headline property: the rank count must not be observable in the answer.
# 3 is in the sweep because it does not divide the default array size, which is
# the case an even split would hide.
mpilab_add_rank_test(distributed-sum RANKS 1 2 3 4 MODE invariant ARGS 100000)
