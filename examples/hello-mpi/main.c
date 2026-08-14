/* Example 1 -- Hello MPI.
 *
 * The smallest program that proves the environment works: every rank starts,
 * every rank knows its own id and the size of the world, and every rank can
 * say which machine it is on.
 *
 * That last part is the whole point of this repository. Run it with four ranks
 * on one host and every line reports the same hostname; run it across the
 * container cluster and the four lines report four different ones. The
 * hostname is the evidence that the simulated nodes are really separate.
 *
 *   mpirun -n 4 hello-mpi
 *   ./scripts/run-mpi.sh -n 4 hello-mpi
 *
 * The output order is deliberately not controlled. Four processes writing to a
 * shared stdout interleave however the runtime buffers them, and rank 0 is
 * under no obligation to arrive first. Programs that need ordered output have
 * to collect it on one rank -- see examples/distributed-sum -- and pretending
 * otherwise here would teach the wrong lesson on the very first example.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* MPI_Get_processor_name rather than gethostname: it is the portable MPI
     * spelling, and on every launcher worth using it returns exactly the node
     * name the process was placed on. */
    char host[MPI_MAX_PROCESSOR_NAME] = "";
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);

    printf("Hello from rank %d of %d on %s\n", rank, world_size, host);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
