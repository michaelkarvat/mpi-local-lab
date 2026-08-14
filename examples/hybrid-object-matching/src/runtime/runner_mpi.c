/* Distributed runtime.
 *
 * Scheduling: pictures in a realistic input differ in size by an order of
 * magnitude, and early exit makes the cost of a same-sized picture vary again,
 * so a static split leaves ranks idle. Work is therefore claimed dynamically.
 *
 * The claim is an MPI-3 one-sided fetch-and-add on a counter hosted by rank 0,
 * rather than the usual master/worker send-recv. Two things follow: there is
 * no dedicated scheduler process (the old driver refused to run on fewer than
 * two ranks and left rank 0 idle throughout), and claiming costs one RMA
 * operation instead of a round trip through a process that might be busy.
 *
 * Trade-off: this requires every rank to hold the whole problem, so peak
 * memory is O(total input) per rank rather than O(one picture). For inputs
 * that fit comfortably in node memory -- the regime this program targets --
 * that buys the elimination of all data movement from the hot path. Streaming
 * pictures on demand would lift the limit at the cost of reintroducing a
 * master; see "Future improvements" in the README.
 */
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "msearch/log.h"
#include "msearch/problem.h"
#include "msearch/runtime.h"

/* One gathered result: picture index plus the Match payload. Indices travel
 * with the payload so rank 0 can scatter results back into input order
 * regardless of which rank produced them. */
#define RECORD_INTS 4

static Status bcast_or_fail(void *buf, int count, MPI_Datatype type, int root, const char *what,
                            char *err, size_t err_len)
{
    if (MPI_Bcast(buf, count, type, root, MPI_COMM_WORLD) != MPI_SUCCESS) {
        msearch_set_err(err, err_len, "MPI_Bcast(%s) failed", what);
        return MSEARCH_ERR_BACKEND;
    }
    return MSEARCH_OK;
}

Status msearch_broadcast_problem(Problem *problem, int root, char *err, size_t err_len)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int header[2] = {problem->num_pictures, problem->num_objects};
    Status status = bcast_or_fail(header, 2, MPI_INT, root, "header", err, err_len);
    if (status != MSEARCH_OK) {
        return status;
    }
    status = bcast_or_fail(&problem->threshold, 1, MPI_DOUBLE, root, "threshold", err, err_len);
    if (status != MSEARCH_OK) {
        return status;
    }

    if (rank != root) {
        status = msearch_problem_alloc(problem, header[0], header[1], err, err_len);
        if (status != MSEARCH_OK) {
            return status;
        }
    }

    /* Shapes travel as two packed arrays rather than two broadcasts per
     * matrix: 2 collectives instead of 2*(P+O) for the metadata. */
    const int shape_ints = (header[0] + header[1]) * 2;
    int *shapes = (shape_ints > 0) ? malloc((size_t)shape_ints * sizeof(int)) : NULL;
    if (shape_ints > 0 && shapes == NULL) {
        msearch_set_err(err, err_len, "cannot allocate shape buffer");
        return MSEARCH_ERR_NOMEM;
    }
    if (rank == root) {
        for (int i = 0; i < header[0]; ++i) {
            shapes[2 * i] = problem->pictures[i].id;
            shapes[2 * i + 1] = problem->pictures[i].n;
        }
        for (int k = 0; k < header[1]; ++k) {
            const int base = 2 * (header[0] + k);
            shapes[base] = problem->objects[k].id;
            shapes[base + 1] = problem->objects[k].m;
        }
    }
    status = bcast_or_fail(shapes, shape_ints, MPI_INT, root, "shapes", err, err_len);
    if (status != MSEARCH_OK) {
        goto done;
    }

    for (int i = 0; i < header[0]; ++i) {
        Picture *picture = &problem->pictures[i];
        if (rank != root) {
            status = msearch_picture_alloc(picture, shapes[2 * i], shapes[2 * i + 1], err, err_len);
            if (status != MSEARCH_OK) {
                goto done;
            }
        }
        status = bcast_or_fail(picture->data, picture->n * picture->n, MPI_INT, root,
                               "picture data", err, err_len);
        if (status != MSEARCH_OK) {
            goto done;
        }
    }
    for (int k = 0; k < header[1]; ++k) {
        Object *object = &problem->objects[k];
        const int base = 2 * (header[0] + k);
        if (rank != root) {
            status = msearch_object_alloc(object, shapes[base], shapes[base + 1], err, err_len);
            if (status != MSEARCH_OK) {
                goto done;
            }
        }
        status = bcast_or_fail(object->data, object->m * object->m, MPI_INT, root, "object data",
                               err, err_len);
        if (status != MSEARCH_OK) {
            goto done;
        }
    }

    if (rank != root) {
        /* Re-validating on every rank is cheap and re-establishes the object
         * ordering the determinism contract depends on. */
        status = msearch_problem_validate(problem, err, err_len);
    }

done:
    free(shapes);
    return status;
}

/* Claim picture indices from the shared counter until it is exhausted. */
static Status process_claimed_pictures(const Problem *problem, const MatchBackend *backend,
                                       void *ctx, MPI_Win win, int *records, int *record_count,
                                       char *err, size_t err_len)
{
    const int increment = 1;
    Status status = MSEARCH_OK;
    *record_count = 0;

    for (;;) {
        int index = 0;
        MPI_Fetch_and_op(&increment, &index, MPI_INT, 0, 0, MPI_SUM, win);
        MPI_Win_flush(0, win);
        if (index >= problem->num_pictures) {
            break;
        }

        Match match;
        status = backend->search(ctx, &problem->pictures[index], &match, err, err_len);
        if (status != MSEARCH_OK) {
            /* Stop claiming, but do not abort: the surviving ranks drain the
             * remaining work and every rank still reaches the collectives
             * below, where the failure is reduced and reported. Aborting here
             * would hang the others in MPI_Gather. */
            break;
        }

        int *record = records + (size_t)(*record_count) * RECORD_INTS;
        record[0] = index;
        record[1] = match.object_id;
        record[2] = match.row;
        record[3] = match.col;
        (*record_count)++;

        MSEARCH_LOG_DEBUG("claimed picture index %d (id %d)", index,
                          problem->pictures[index].id);
    }
    return status;
}

static Status gather_records(const Problem *problem, const int *records, int record_count, int rank,
                             int world_size, Match *matches, char *err, size_t err_len)
{
    int *counts = NULL;
    int *displs = NULL;
    int *gathered = NULL;
    Status status = MSEARCH_OK;

    if (rank == 0) {
        counts = malloc((size_t)world_size * sizeof(int));
        displs = malloc((size_t)world_size * sizeof(int));
        gathered = malloc((size_t)problem->num_pictures * RECORD_INTS * sizeof(int));
        if (counts == NULL || displs == NULL ||
            (problem->num_pictures > 0 && gathered == NULL)) {
            msearch_set_err(err, err_len, "cannot allocate gather buffers");
            status = MSEARCH_ERR_NOMEM;
            goto done;
        }
    }

    int send_ints = record_count * RECORD_INTS;
    MPI_Gather(&send_ints, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < world_size; ++r) {
            displs[r] = total;
            total += counts[r];
        }
    }
    MPI_Gatherv(records, send_ints, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < problem->num_pictures; ++i) {
            matches[i] = msearch_match_none(problem->pictures[i].id);
        }
        int total = 0;
        for (int r = 0; r < world_size; ++r) {
            total += counts[r];
        }
        for (int i = 0; i < total; i += RECORD_INTS) {
            const int index = gathered[i];
            if (index < 0 || index >= problem->num_pictures) {
                msearch_set_err(err, err_len, "gathered out-of-range picture index %d", index);
                status = MSEARCH_ERR_BACKEND;
                goto done;
            }
            matches[index].picture_id = problem->pictures[index].id;
            matches[index].object_id = gathered[i + 1];
            matches[index].row = gathered[i + 2];
            matches[index].col = gathered[i + 3];
        }
    }

done:
    free(counts);
    free(displs);
    free(gathered);
    return status;
}

Status msearch_run_mpi(const Problem *problem, const MatchBackend *backend, const Config *config,
                       Match *matches, char *err, size_t err_len)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* A single rank has nothing to coordinate; skip the RMA machinery. */
    if (world_size == 1) {
        return msearch_run_local(problem, backend, config, matches, err, err_len);
    }

    void *ctx = NULL;
    Status status = backend->create(problem, config, &ctx, err, err_len);

    int *records = NULL;
    int record_count = 0;
    if (status == MSEARCH_OK && problem->num_pictures > 0) {
        /* Worst case a single rank claims everything. */
        records = malloc((size_t)problem->num_pictures * RECORD_INTS * sizeof(int));
        if (records == NULL) {
            msearch_set_err(err, err_len, "cannot allocate result records");
            status = MSEARCH_ERR_NOMEM;
        }
    }

    /* The window is collective, so it is created whether or not this rank is
     * healthy; an unhealthy rank simply claims no work. */
    int *counter = NULL;
    MPI_Win win;
    MPI_Win_allocate(rank == 0 ? (MPI_Aint)sizeof(int) : 0, sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &counter, &win);
    if (rank == 0) {
        *counter = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock_all(0, win);

    /* A rank that failed to start simply claims nothing: the counter is shared,
     * so healthy ranks still drain every index. No compensation is needed. */
    if (status == MSEARCH_OK) {
        status = process_claimed_pictures(problem, backend, ctx, win, records, &record_count, err,
                                          err_len);
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);

    const Status gather_status =
        gather_records(problem, records, record_count, rank, world_size, matches, err, err_len);
    if (status == MSEARCH_OK) {
        status = gather_status;
    }

    /* Every rank must agree on success or the caller's control flow diverges. */
    int local_failed = (status != MSEARCH_OK) ? 1 : 0;
    int any_failed = 0;
    MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed && status == MSEARCH_OK) {
        msearch_set_err(err, err_len, "another rank reported a failure");
        status = MSEARCH_ERR_BACKEND;
    }

    free(records);
    if (ctx != NULL) {
        backend->destroy(ctx);
    }
    return status;
}
