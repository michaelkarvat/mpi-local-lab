/* Example 3 -- Distributed sum.
 *
 * Split an array across the ranks, have each one sum its own slice, and
 * combine the slices with a single collective. This is the shape of most real
 * distributed computations: scatter, compute locally, reduce.
 *
 *   mpirun -n 4 distributed-sum
 *   mpirun -n 4 distributed-sum 5000000
 *
 * Two decisions here are worth more than the arithmetic.
 *
 * WHY INTEGERS. The obvious version of this example sums doubles, and it is
 * subtly wrong as a teaching device: floating-point addition is not
 * associative, so MPI_Reduce over 2 ranks and over 4 ranks group the additions
 * differently and the last bits of the answer disagree. The program would look
 * correct and the test asserting "same answer at any rank count" would fail
 * intermittently. Summing integers makes the reduction exactly associative, so
 * the claim that the rank count is not observable in the result is true rather
 * than nearly true. If you want to see the other behaviour, change the buffer
 * to double and run the rank sweep -- the failure is the lesson.
 *
 * WHY EVERY RANK COMPUTES THE DECOMPOSITION. counts[] and displs[] are derived
 * from n and the world size, which every rank already knows, so there is
 * nothing to communicate. MPI_Scatterv only reads them on the root, but each
 * rank needs its own count to size its receive buffer, and deriving it beats
 * sending it.
 *
 * The array is not divisible by the rank count on purpose. Even division is
 * the case that hides the bug; MPI_Scatterv with explicit counts and
 * displacements is what handles the real one.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

    const long n = parse_positive(argc > 1 ? argv[1] : NULL, 1000000);

    /* Block decomposition with the remainder spread over the leading ranks,
     * so the largest and smallest slice never differ by more than one. */
    int *counts = malloc((size_t)world_size * sizeof(int));
    int *displs = malloc((size_t)world_size * sizeof(int));
    if (counts == NULL || displs == NULL) {
        fprintf(stderr, "rank %d: cannot allocate decomposition tables\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    const long base = n / world_size;
    const long remainder = n % world_size;
    long offset = 0;
    for (int r = 0; r < world_size; ++r) {
        const long share = base + (r < remainder ? 1 : 0);
        counts[r] = (int)share;
        displs[r] = (int)offset;
        offset += share;
    }

    /* Only the root holds the whole array; every rank holds one slice. That is
     * the memory property that makes distributed-memory programming worth the
     * trouble, and the opposite of the trade the object-matching example
     * makes deliberately -- see its docs/ARCHITECTURE.md. */
    int *whole = NULL;
    if (rank == 0) {
        whole = malloc((size_t)n * sizeof(int));
        if (whole == NULL) {
            fprintf(stderr, "rank 0: cannot allocate %ld elements\n", n);
            MPI_Abort(MPI_COMM_WORLD, 2);
            return 2;
        }
        for (long i = 0; i < n; ++i) {
            whole[i] = (int)(i + 1);
        }
    }

    const int my_count = counts[rank];
    int *slice = malloc((size_t)(my_count > 0 ? my_count : 1) * sizeof(int));
    if (slice == NULL) {
        fprintf(stderr, "rank %d: cannot allocate slice of %d\n", rank, my_count);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    MPI_Scatterv(whole, counts, displs, MPI_INT, slice, my_count, MPI_INT, 0, MPI_COMM_WORLD);

    long long local = 0;
    for (int i = 0; i < my_count; ++i) {
        local += slice[i];
    }

    long long total = 0;
    MPI_Reduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        /* The closed form is the point of using 1..n: the answer is known
         * independently of the program, so this is a correctness check and not
         * a comparison of the program against itself. */
        const long long expected = (long long)n * (n + 1) / 2;

        /* Results only. The rank count and the timing are deliberately absent
         * from stdout: they are exactly the things that differ between runs
         * the test requires to be identical. */
        printf("distributed-sum n=%ld\n", n);
        printf("sum      = %lld\n", total);
        printf("expected = %lld\n", expected);
        printf("match    = %s\n", total == expected ? "yes" : "no");

        fprintf(stderr, "ranks=%d elapsed=%.6fs slice=%d..%d\n", world_size, elapsed,
                counts[world_size - 1], counts[0]);

        free(whole);
        free(slice);
        free(counts);
        free(displs);
        MPI_Finalize();
        return total == expected ? 0 : 1;
    }

    free(slice);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
