#include "msearch/runtime.h"

#include "msearch/log.h"

Status msearch_run_local(const Problem *problem, const MatchBackend *backend, const Config *config,
                         Match *matches, char *err, size_t err_len)
{
    void *ctx = NULL;
    Status status = backend->create(problem, config, &ctx, err, err_len);
    if (status != MSEARCH_OK) {
        return status;
    }

    for (int i = 0; i < problem->num_pictures; ++i) {
        status = backend->search(ctx, &problem->pictures[i], &matches[i], err, err_len);
        if (status != MSEARCH_OK) {
            break;
        }
        MSEARCH_LOG_DEBUG("picture %d done (%d/%d)", problem->pictures[i].id, i + 1,
                          problem->num_pictures);
    }

    backend->destroy(ctx);
    return status;
}
