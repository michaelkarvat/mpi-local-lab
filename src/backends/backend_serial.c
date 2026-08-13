/* Reference implementation.
 *
 * This backend is the project's ground truth: it is always compiled in, it has
 * no dependencies, and every other backend is tested for exact equality
 * against it. It is also the baseline every speedup number is measured from,
 * which is why it is written to be *correct and obvious* rather than clever --
 * the only optimisation it carries is the early exit inside msearch_score_at,
 * which is part of the shared metric and therefore applies to all backends
 * equally.
 */
#include <stdlib.h>

#include "backends.h"
#include "msearch/log.h"
#include "msearch/metric.h"
#include "msearch/problem.h"

typedef struct {
    const Problem *problem;
    double zero_eps;
} SerialContext;

static bool serial_available(char *reason, size_t reason_len)
{
    (void)reason;
    (void)reason_len;
    return true;
}

static Status serial_create(const Problem *problem, const Config *config, void **ctx, char *err,
                            size_t err_len)
{
    SerialContext *context = calloc(1, sizeof(SerialContext));
    if (context == NULL) {
        msearch_set_err(err, err_len, "cannot allocate serial context");
        return MSEARCH_ERR_NOMEM;
    }
    context->problem = problem;
    context->zero_eps = config->zero_eps;
    *ctx = context;
    return MSEARCH_OK;
}

static Status serial_search(void *ctx, const Picture *picture, Match *out, char *err,
                            size_t err_len)
{
    (void)err;
    (void)err_len;
    const SerialContext *context = ctx;
    const Problem *problem = context->problem;

    *out = msearch_match_none(picture->id);

    /* Objects are sorted by ascending id, so the first one that matches is the
     * one the determinism contract asks for, and we can stop immediately. */
    for (int k = 0; k < problem->num_objects; ++k) {
        const Object *object = &problem->objects[k];
        const long long placements = msearch_placement_count(picture->n, object->m);
        if (placements == 0) {
            continue; /* object does not fit in this picture */
        }
        const int span = picture->n - object->m + 1;

        /* Scanning placement keys in increasing order means the first hit is
         * the row-major-first one -- no min-reduction needed here. */
        for (long long key = 0; key < placements; ++key) {
            const int row = (int)(key / span);
            const int col = (int)(key % span);
            const double score = msearch_score_at(picture->data, picture->n, object->data,
                                                  object->m, row, col, problem->threshold,
                                                  context->zero_eps);
            if (score < problem->threshold) {
                out->object_id = object->id;
                out->row = row;
                out->col = col;
                return MSEARCH_OK;
            }
        }
    }
    return MSEARCH_OK;
}

static void serial_destroy(void *ctx)
{
    free(ctx);
}

const MatchBackend msearch_backend_serial = {
    "serial", "single-threaded reference implementation", serial_available,
    serial_create, serial_search, serial_destroy,
};
