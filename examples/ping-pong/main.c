/* Example 2 -- Ping pong.
 *
 * Point-to-point communication: rank 0 sends a buffer to rank 1, rank 1 sends
 * it straight back, repeat. This is the canonical way to measure MPI latency
 * and the smallest program in which MPI_Send and MPI_Recv have to agree about
 * a tag, a count and a datatype.
 *
 *   mpirun -n 2 ping-pong
 *   mpirun -n 2 ping-pong 1000 65536      # iterations, payload bytes
 *
 * Only ranks 0 and 1 do the exchange. Extra ranks wait at the barrier instead
 * of being an error, so the same binary can be launched at whatever rank count
 * the cluster happens to have and still produce the same answer -- which is
 * what makes the rank sweep in CMakeLists.txt meaningful.
 *
 * Timing goes to stderr, results to stdout. Latency is the one number here
 * that legitimately changes run to run, so keeping it off stdout is what lets
 * the test assert that stdout does not.
 *
 * A caution specific to this repository: the round-trip time you measure
 * between two containers is not a network measurement. Both processes are on
 * one machine talking over a virtual bridge, so the figure reflects the host's
 * memory and loopback path, not any real interconnect. See docs/ARCHITECTURE.md.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PING_TAG 1
#define PONG_TAG 2

static long parse_positive(const char *text, long fallback)
{
    if (text == NULL) {
        return fallback;
    }
    char *end = NULL;
    const long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        return fallback;
    }
    return value;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (rank == 0) {
            fprintf(stderr, "ping-pong needs at least 2 ranks (got %d)\n", world_size);
        }
        MPI_Finalize();
        return 1;
    }

    const long iterations = parse_positive(argc > 1 ? argv[1] : NULL, 1000);
    const long payload = parse_positive(argc > 2 ? argv[2] : NULL, 4096);

    char *buffer = malloc((size_t)payload);
    if (buffer == NULL) {
        fprintf(stderr, "rank %d: cannot allocate %ld bytes\n", rank, payload);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    /* A known pattern, so a corrupted round trip is detectable rather than
     * silently passing on whatever happened to be in the heap. */
    memset(buffer, 0xA5, (size_t)payload);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int corrupted = 0;

    if (rank == 0) {
        for (long i = 0; i < iterations; ++i) {
            MPI_Send(buffer, (int)payload, MPI_BYTE, 1, PING_TAG, MPI_COMM_WORLD);
            MPI_Recv(buffer, (int)payload, MPI_BYTE, 1, PONG_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        for (long b = 0; b < payload; ++b) {
            if ((unsigned char)buffer[b] != 0xA5u) {
                corrupted = 1;
                break;
            }
        }
    } else if (rank == 1) {
        for (long i = 0; i < iterations; ++i) {
            MPI_Recv(buffer, (int)payload, MPI_BYTE, 0, PING_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buffer, (int)payload, MPI_BYTE, 0, PONG_TAG, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        /* Results: deterministic, so the rank sweep can compare them. */
        printf("ping-pong %ld round trips of %ld bytes\n", iterations, payload);
        printf("payload intact: %s\n", corrupted ? "no" : "yes");

        /* Diagnostics: timing varies run to run, so it stays off stdout. */
        const double per_trip_us = (elapsed / (double)iterations) * 1e6;
        fprintf(stderr, "elapsed=%.6fs round-trip=%.2fus one-way=%.2fus\n", elapsed, per_trip_us,
                per_trip_us / 2.0);
    }

    free(buffer);
    MPI_Finalize();
    return corrupted ? 1 : 0;
}
