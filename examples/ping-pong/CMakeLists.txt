mpilab_add_example(ping-pong
  SOURCES main.c
  MPI
  DESCRIPTION "MPI_Send/MPI_Recv round trips between two ranks")

# Small counts: this test checks that the exchange completes and the payload
# survives, not how fast it is. Ranks above 1 idle at the barrier, so the
# stdout of a 2-rank and a 4-rank run must be identical.
mpilab_add_rank_test(ping-pong RANKS 2 4 MODE invariant ARGS 50 1024)
