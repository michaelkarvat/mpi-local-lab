/* Entry point: wire configuration, input, a backend and a runtime together.
 *
 * Everything interesting lives behind an interface, so this file stays a thin
 * composition root: it owns process lifetime (MPI init/finalize), decides who
 * reads the input and who writes the output, and reports timing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MSEARCH_HAVE_MPI
#  include <mpi.h>
#endif

#include "msearch/backend.h"
#include "msearch/config.h"
#include "msearch/io.h"
#include "msearch/log.h"
#include "msearch/problem.h"
#include "msearch/runtime.h"

#define ERR_LEN 512

typedef struct {
    int rank;
    int world_size;
} Topology;

static void barrier(void)
{
#ifdef MSEARCH_HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

static void abort_all(int code)
{
#ifdef MSEARCH_HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, code);
#endif
    exit(code);
}

/* This process's position within its compute node.
 *
 * Two backends need this and neither should know about MPI: the CUDA backend
 * maps `rank` onto a device so co-located ranks do not all grab GPU 0, and the
 * OpenMP backend divides the node's cores by `count` so they do not each spawn
 * a full set of threads. */
static void compute_node_topology(int *rank, int *count)
{
    *rank = 0;
    *count = 1;
#ifdef MSEARCH_HAVE_MPI
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, rank);
    MPI_Comm_size(node_comm, count);
    MPI_Comm_free(&node_comm);
#endif
}

static Status run_search(const Problem *problem, const MatchBackend *backend, const Config *config,
                         Match *matches, double *elapsed, char *err, size_t err_len)
{
    barrier();
    const double start = msearch_wtime();
#ifdef MSEARCH_HAVE_MPI
    const Status status = msearch_run_mpi(problem, backend, config, matches, err, err_len);
#else
    const Status status = msearch_run_local(problem, backend, config, matches, err, err_len);
#endif
    barrier();
    *elapsed = msearch_wtime() - start;
    return status;
}

/* --verify: run every available backend over the same problem and require
 * byte-identical results. The serial backend is the reference.
 *
 * This is the payoff of the backend interface and the determinism contract:
 * three independent implementations -- scalar C, OpenMP over placements, and
 * two CUDA kernels -- are checked against each other on real input, without
 * anyone having to hand-maintain expected output. */
static Status run_verify(const Problem *problem, const Config *config, char *err, size_t err_len)
{
    const int count = problem->num_pictures;
    Match *reference = calloc((size_t)(count > 0 ? count : 1), sizeof(Match));
    Match *candidate = calloc((size_t)(count > 0 ? count : 1), sizeof(Match));
    Status status = MSEARCH_OK;
    int failures = 0;

    if (reference == NULL || candidate == NULL) {
        msearch_set_err(err, err_len, "cannot allocate verification buffers");
        status = MSEARCH_ERR_NOMEM;
        goto done;
    }

    status = msearch_run_local(problem, msearch_backend_find("serial"), config, reference, err,
                               err_len);
    if (status != MSEARCH_OK) {
        goto done;
    }
    printf("reference: serial (%d pictures)\n", count);

    int backend_count = 0;
    const MatchBackend *const *backends = msearch_backend_all(&backend_count);
    for (int b = 0; b < backend_count; ++b) {
        const MatchBackend *backend = backends[b];
        char reason[128] = "";
        if (strcmp(backend->name, "serial") == 0) {
            continue;
        }
        if (!backend->available(reason, sizeof(reason))) {
            printf("  %-8s skipped (%s)\n", backend->name, reason);
            continue;
        }

        const Status run_status =
            msearch_run_local(problem, backend, config, candidate, err, err_len);
        if (run_status != MSEARCH_OK) {
            printf("  %-8s FAILED (%s: %s)\n", backend->name, msearch_status_str(run_status), err);
            failures++;
            continue;
        }

        int mismatches = 0;
        for (int i = 0; i < count; ++i) {
            if (memcmp(&reference[i], &candidate[i], sizeof(Match)) != 0) {
                if (mismatches < 5) {
                    char want[128];
                    char got[128];
                    msearch_format_match(&reference[i], want, sizeof(want));
                    msearch_format_match(&candidate[i], got, sizeof(got));
                    printf("    mismatch: expected \"%s\", got \"%s\"\n", want, got);
                }
                mismatches++;
            }
        }
        if (mismatches == 0) {
            printf("  %-8s OK (identical to serial)\n", backend->name);
        } else {
            printf("  %-8s MISMATCH on %d/%d pictures\n", backend->name, mismatches, count);
            failures++;
        }
    }

    if (failures > 0) {
        msearch_set_err(err, err_len, "%d backend(s) disagreed with the serial reference",
                        failures);
        status = MSEARCH_ERR_BACKEND;
    }

done:
    free(reference);
    free(candidate);
    return status;
}

static int compare_double(const void *a, const void *b)
{
    const double lhs = *(const double *)a;
    const double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static void report_bench(const double *samples, int count, const char *backend_name, int world_size)
{
    double *sorted = malloc((size_t)count * sizeof(double));
    if (sorted == NULL) {
        return;
    }
    memcpy(sorted, samples, (size_t)count * sizeof(double));
    qsort(sorted, (size_t)count, sizeof(double), compare_double);

    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        total += sorted[i];
    }

    /* The minimum is the headline figure: it is the run least polluted by
     * scheduler noise, and it is what a speedup ratio should be built from. */
    fprintf(stderr, "backend=%s ranks=%d reps=%d min=%.6fs median=%.6fs mean=%.6fs max=%.6fs\n",
            backend_name, world_size, count, sorted[0], sorted[count / 2], total / count,
            sorted[count - 1]);
    free(sorted);
}

int main(int argc, char **argv)
{
    Config config;
    char err[ERR_LEN] = "";
    Topology topo = {0, 1};
    EarlyAction action = EARLY_NONE;
    int exit_code = 0;

#ifdef MSEARCH_HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &topo.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &topo.world_size);
#endif

    msearch_config_defaults(&config);
    Status status = msearch_config_parse(&config, argc, argv, &action, err, ERR_LEN);
    msearch_log_init(config.log_level, topo.world_size > 1 ? topo.rank : -1);

    if (status != MSEARCH_OK) {
        if (topo.rank == 0) {
            MSEARCH_LOG_ERROR("%s", err);
        }
        exit_code = 2;
        goto finalize;
    }
    if (action != EARLY_NONE) {
        if (topo.rank == 0) {
            switch (action) {
            case EARLY_HELP:           msearch_print_usage(argc > 0 ? argv[0] : "msearch"); break;
            case EARLY_VERSION:        printf("msearch %s\n", msearch_version()); break;
            case EARLY_LIST_BACKENDS:  msearch_backend_print_list(); break;
            case EARLY_NONE:           break;
            }
        }
        goto finalize;
    }

    compute_node_topology(&config.node_rank, &config.node_ranks);

    /* Rank 0 owns the file system: one process reads the input and one writes
     * the output, so a shared file system is never hit by N processes at once. */
    Problem problem;
    msearch_problem_init(&problem);
    if (topo.rank == 0) {
        status = msearch_read_problem(config.input_path, &problem, err, ERR_LEN);
        if (status != MSEARCH_OK) {
            MSEARCH_LOG_ERROR("%s: %s", msearch_status_str(status), err);
            abort_all(2);
        }
        MSEARCH_LOG_INFO("loaded %d pictures and %d objects (threshold %g) from %s",
                         problem.num_pictures, problem.num_objects, problem.threshold,
                         config.input_path);
    }

#ifdef MSEARCH_HAVE_MPI
    status = msearch_broadcast_problem(&problem, 0, err, ERR_LEN);
    if (status != MSEARCH_OK) {
        MSEARCH_LOG_ERROR("broadcast failed: %s", err);
        abort_all(3);
    }
#endif

    if (config.verify) {
        if (topo.world_size > 1) {
            if (topo.rank == 0) {
                MSEARCH_LOG_ERROR("--verify compares backends in one process; run without mpirun");
            }
            exit_code = 2;
            goto cleanup;
        }
        status = run_verify(&problem, &config, err, ERR_LEN);
        if (status != MSEARCH_OK) {
            MSEARCH_LOG_ERROR("%s", err);
            exit_code = 1;
        }
        goto cleanup;
    }

    const MatchBackend *backend = (strcmp(config.backend_name, "auto") == 0)
                                      ? msearch_backend_auto()
                                      : msearch_backend_find(config.backend_name);
    char reason[128] = "";
    if (!backend->available(reason, sizeof(reason))) {
        MSEARCH_LOG_ERROR("backend '%s' is not usable: %s", backend->name, reason);
        abort_all(4);
    }
    if (topo.rank == 0) {
        MSEARCH_LOG_INFO("backend=%s ranks=%d", backend->name, topo.world_size);
    }

    Match *matches = calloc((size_t)(problem.num_pictures > 0 ? problem.num_pictures : 1),
                            sizeof(Match));
    if (matches == NULL) {
        MSEARCH_LOG_ERROR("cannot allocate result array");
        abort_all(5);
    }

    const int reps = config.bench_reps > 0 ? config.bench_reps : 1;
    double *samples = malloc((size_t)reps * sizeof(double));
    if (samples == NULL) {
        MSEARCH_LOG_ERROR("cannot allocate timing buffer");
        abort_all(5);
    }

    for (int r = 0; r < reps; ++r) {
        status = run_search(&problem, backend, &config, matches, &samples[r], err, ERR_LEN);
        if (status != MSEARCH_OK) {
            MSEARCH_LOG_ERROR("search failed: %s", err);
            abort_all(6);
        }
    }

    if (topo.rank == 0) {
        /* Timing is diagnostic, so it goes to stderr and results own stdout.
         * Otherwise `--output -` interleaves the two and `msearch -o - > file`
         * writes a TotalTime line into the results. */
        if (config.bench_reps > 0) {
            report_bench(samples, reps, backend->name, topo.world_size);
        } else {
            fprintf(stderr, "TotalTime = %.6f seconds\n", samples[0]);
        }
        status = msearch_write_results(config.output_path, matches, problem.num_pictures, err,
                                       ERR_LEN);
        if (status != MSEARCH_OK) {
            MSEARCH_LOG_ERROR("%s", err);
            exit_code = 1;
        } else {
            MSEARCH_LOG_INFO("wrote %d results to %s", problem.num_pictures, config.output_path);
        }
    }

    free(samples);
    free(matches);

cleanup:
    msearch_problem_free(&problem);

finalize:
#ifdef MSEARCH_HAVE_MPI
    MPI_Finalize();
#endif
    return exit_code;
}
