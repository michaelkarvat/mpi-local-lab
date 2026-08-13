/* Multicore CPU backend.
 *
 * Decomposition note: the original implementation parallelised over *objects*
 * (a dozen of them) and guarded a shared flag with a critical section on every
 * iteration, so the loop body was bracketed by a global lock and could not
 * scale past K threads. This backend parallelises over *placements* -- the
 * loop that actually holds the work, typically 10^4..10^7 iterations -- and
 * takes a lock only on the rare event of an actual match.
 */
#include <omp.h>
#include <stdlib.h>

#include "backends.h"
#include "msearch/log.h"
#include "msearch/metric.h"
#include "msearch/problem.h"

/* Placements are cheap individually, so hand out chunks rather than single
 * iterations; dynamic (not static) because early exit makes the cost of a
 * placement vary by an order of magnitude across the picture. */
#define MSEARCH_OMP_CHUNK 256

typedef struct {
    const Problem *problem;
    double zero_eps;
    int threads;
} OpenMpContext;

static bool openmp_available(char *reason, size_t reason_len)
{
    (void)reason;
    (void)reason_len;
    return true;
}

static Status openmp_create(const Problem *problem, const Config *config, void **ctx, char *err,
                            size_t err_len)
{
    OpenMpContext *context = calloc(1, sizeof(OpenMpContext));
    if (context == NULL) {
        msearch_set_err(err, err_len, "cannot allocate OpenMP context");
        return MSEARCH_ERR_NOMEM;
    }
    context->problem = problem;
    context->zero_eps = config->zero_eps;
    context->threads = config->threads > 0 ? config->threads : omp_get_max_threads();
    *ctx = context;
    MSEARCH_LOG_DEBUG("openmp backend: %d threads", context->threads);
    return MSEARCH_OK;
}

/* Lowest matching placement key for one object, or MSEARCH_NO_PLACEMENT. */
static int lowest_matching_placement(const Picture *picture, const Object *object, double threshold,
                                     double zero_eps, long long placements, int span, int threads)
{
    int best = MSEARCH_NO_PLACEMENT;

#pragma omp parallel for num_threads(threads) schedule(dynamic, MSEARCH_OMP_CHUNK) \
    shared(best) default(none)                                                      \
    firstprivate(picture, object, threshold, zero_eps, placements, span)
    for (long long key = 0; key < placements; ++key) {
        /* Early exit: a key at or above the best found so far cannot lower the
         * minimum, so it need not be scored. This is what makes the
         * deterministic min-reduction cost about the same as the racy
         * "first thread wins" flag it replaced, while staying reproducible. */
        int snapshot;
#pragma omp atomic read
        snapshot = best;
        if (key >= snapshot) {
            continue;
        }

        const int row = (int)(key / span);
        const int col = (int)(key % span);
        const double score = msearch_score_at(picture->data, picture->n, object->data, object->m,
                                              row, col, threshold, zero_eps);
        if (score < threshold) {
            /* Reached only on a real match, so the lock is uncontended in the
             * overwhelmingly common no-match case. */
#pragma omp critical(msearch_best_placement)
            {
                if ((int)key < best) {
                    best = (int)key;
                }
            }
        }
    }
    return best;
}

static Status openmp_search(void *ctx, const Picture *picture, Match *out, char *err,
                            size_t err_len)
{
    (void)err;
    (void)err_len;
    const OpenMpContext *context = ctx;
    const Problem *problem = context->problem;

    *out = msearch_match_none(picture->id);

    for (int k = 0; k < problem->num_objects; ++k) {
        const Object *object = &problem->objects[k];
        const long long placements = msearch_placement_count(picture->n, object->m);
        if (placements == 0) {
            continue;
        }
        const int span = picture->n - object->m + 1;
        const int best = lowest_matching_placement(picture, object, problem->threshold,
                                                   context->zero_eps, placements, span,
                                                   context->threads);
        if (best != MSEARCH_NO_PLACEMENT) {
            out->object_id = object->id;
            out->row = best / span;
            out->col = best % span;
            return MSEARCH_OK;
        }
    }
    return MSEARCH_OK;
}

static void openmp_destroy(void *ctx)
{
    free(ctx);
}

const MatchBackend msearch_backend_openmp = {
    "openmp", "multicore CPU (OpenMP, parallel over placements)", openmp_available,
    openmp_create, openmp_search, openmp_destroy,
};
