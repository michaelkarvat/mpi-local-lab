/* Example 4 -- Distributed matrix multiply, MPI over OpenMP.
 *
 * C = A x B, with the rows of A and C spread across ranks and B replicated.
 * This is the first example where both levels of parallelism are in play at
 * once, which is the configuration the PRD calls MPI + OpenMP: ranks divide
 * the rows between nodes, threads divide one node's rows between its cores.
 *
 *   mpirun -n 4 matrix-multiply
 *   mpirun -n 4 matrix-multiply 512
 *   OMP_NUM_THREADS=2 mpirun -n 4 matrix-multiply
 *
 * WHY B IS BROADCAST AND A IS SCATTERED. Every rank needs every column of B to
 * compute any row of C, but only its own rows of A. Broadcasting the small
 * operand and scattering the large one is the standard row-block layout, and
 * it is why this scales in memory: each rank holds n*n/ranks of A rather than
 * all of it.
 *
 * WHY INTEGERS AGAIN. Same reason as distributed-sum: the test asserts that
 * the answer does not depend on the rank count, and that is only exactly true
 * when the arithmetic is associative. Here it matters twice over, because
 * OpenMP would reassociate the inner product as well. Note that the threading
 * is over rows, never over k -- each element of C is accumulated by one thread
 * in one fixed order, so adding threads cannot change a single bit either.
 *
 * This is a work-distribution demonstration, not a fast GEMM. The inner loop
 * is the naive triple nest; a real implementation would block for cache and
 * call a tuned library. Optimising it would obscure the communication pattern,
 * which is the part worth reading.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

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

/* Deterministic fill: the operands are a pure function of their indices, so
 * every rank count and every machine starts from the same matrices without
 * anyone shipping a data file around. Values stay small so the accumulated
 * products cannot overflow int for any size this example will be run at. */
static int a_at(long row, long col)
{
    return (int)((row + 2 * col) % 7);
}

static int b_at(long row, long col)
{
    return (int)((3 * row + col) % 5);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const long n = parse_positive(argc > 1 ? argv[1] : NULL, 256);

    /* Row decomposition, remainder over the leading ranks -- the same shape as
     * distributed-sum, but the unit is a row of n elements rather than one. */
    int *row_counts = malloc((size_t)world_size * sizeof(int));
    int *elem_counts = malloc((size_t)world_size * sizeof(int));
    int *elem_displs = malloc((size_t)world_size * sizeof(int));
    if (row_counts == NULL || elem_counts == NULL || elem_displs == NULL) {
        fprintf(stderr, "rank %d: cannot allocate decomposition tables\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    const long base = n / world_size;
    const long remainder = n % world_size;
    long row_offset = 0;
    for (int r = 0; r < world_size; ++r) {
        const long rows = base + (r < remainder ? 1 : 0);
        row_counts[r] = (int)rows;
        elem_counts[r] = (int)(rows * n);
        elem_displs[r] = (int)(row_offset * n);
        row_offset += rows;
    }

    const int my_rows = row_counts[rank];

    int *a_whole = NULL;
    int *c_whole = NULL;
    int *b = malloc((size_t)(n * n) * sizeof(int));
    int *a_rows = malloc((size_t)(my_rows > 0 ? (long)my_rows * n : 1) * sizeof(int));
    int *c_rows = malloc((size_t)(my_rows > 0 ? (long)my_rows * n : 1) * sizeof(int));
    if (b == NULL || a_rows == NULL || c_rows == NULL) {
        fprintf(stderr, "rank %d: cannot allocate operands for n=%ld\n", rank, n);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    if (rank == 0) {
        a_whole = malloc((size_t)(n * n) * sizeof(int));
        c_whole = malloc((size_t)(n * n) * sizeof(int));
        if (a_whole == NULL || c_whole == NULL) {
            fprintf(stderr, "rank 0: cannot allocate the full matrices for n=%ld\n", n);
            MPI_Abort(MPI_COMM_WORLD, 2);
            return 2;
        }
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < n; ++j) {
                a_whole[i * n + j] = a_at(i, j);
                b[i * n + j] = b_at(i, j);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    /* B in full to everyone; A one row block each. */
    MPI_Bcast(b, (int)(n * n), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a_whole, elem_counts, elem_displs, MPI_INT, a_rows, elem_counts[rank], MPI_INT, 0,
                 MPI_COMM_WORLD);

    /* Threads split this rank's rows. Nothing is shared for writing, so no
     * reduction, no critical section, and no reassociation of the sum. */
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < my_rows; ++i) {
        for (long j = 0; j < n; ++j) {
            int sum = 0;
            for (long k = 0; k < n; ++k) {
                sum += a_rows[(long)i * n + k] * b[k * n + j];
            }
            c_rows[(long)i * n + j] = sum;
        }
    }

    MPI_Gatherv(c_rows, elem_counts[rank], MPI_INT, c_whole, elem_counts, elem_displs, MPI_INT, 0,
                MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;

    int failed = 0;
    if (rank == 0) {
        /* A checksum stands in for printing n*n numbers, and an independently
         * recomputed corner element checks that the gather put the rows back
         * where they belong rather than merely returning something plausible. */
        long long checksum = 0;
        for (long i = 0; i < n * n; ++i) {
            checksum += c_whole[i];
        }

        long long corner = 0;
        for (long k = 0; k < n; ++k) {
            corner += (long long)a_at(n - 1, k) * b_at(k, n - 1);
        }
        const int got_corner = c_whole[(n - 1) * n + (n - 1)];
        failed = ((long long)got_corner != corner);

        printf("matrix-multiply n=%ld\n", n);
        printf("checksum      = %lld\n", checksum);
        printf("C[n-1][n-1]   = %d\n", got_corner);
        printf("corner check  = %s\n", failed ? "no" : "yes");

        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        fprintf(stderr, "ranks=%d threads/rank=%d rows/rank=%d..%d elapsed=%.6fs\n", world_size,
                threads, row_counts[world_size - 1], row_counts[0], elapsed);
    }

    free(a_whole);
    free(c_whole);
    free(b);
    free(a_rows);
    free(c_rows);
    free(row_counts);
    free(elem_counts);
    free(elem_displs);

    MPI_Finalize();
    return failed ? 1 : 0;
}
