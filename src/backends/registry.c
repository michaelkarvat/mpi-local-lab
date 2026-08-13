#include "backends.h"

#include <stdio.h>
#include <string.h>

/* Ordered best-first: msearch_backend_auto() takes the first available entry. */
static const MatchBackend *const kBackends[] = {
#ifdef MSEARCH_HAVE_CUDA
    &msearch_backend_cuda,
#endif
#ifdef MSEARCH_HAVE_OPENMP
    &msearch_backend_openmp,
#endif
    &msearch_backend_serial,
};

static const int kBackendCount = (int)(sizeof(kBackends) / sizeof(kBackends[0]));

const MatchBackend *const *msearch_backend_all(int *count)
{
    *count = kBackendCount;
    return kBackends;
}

const MatchBackend *msearch_backend_find(const char *name)
{
    for (int i = 0; i < kBackendCount; ++i) {
        if (strcmp(kBackends[i]->name, name) == 0) {
            return kBackends[i];
        }
    }
    return NULL;
}

const MatchBackend *msearch_backend_auto(void)
{
    for (int i = 0; i < kBackendCount; ++i) {
        char reason[128];
        if (kBackends[i]->available(reason, sizeof(reason))) {
            return kBackends[i];
        }
    }
    /* Unreachable: the serial backend is always compiled in and always
     * available. Returning it explicitly keeps the caller free of NULL checks. */
    return &msearch_backend_serial;
}

void msearch_backend_print_list(void)
{
    printf("%-8s  %-9s  %s\n", "BACKEND", "STATUS", "DESCRIPTION");
    for (int i = 0; i < kBackendCount; ++i) {
        char reason[128] = "";
        const bool ok = kBackends[i]->available(reason, sizeof(reason));
        printf("%-8s  %-9s  %s\n", kBackends[i]->name, ok ? "available" : "unavailable",
               kBackends[i]->description);
        if (!ok && reason[0] != '\0') {
            printf("%-8s  %-9s  (%s)\n", "", "", reason);
        }
    }
}
