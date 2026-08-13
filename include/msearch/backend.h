/* The backend interface: the spine of the architecture.
 *
 * A backend knows how to answer exactly one question -- "does any object occur
 * in this picture?" -- and nothing about where the picture came from. The
 * runtimes (local, MPI) know how to distribute pictures and nothing about how
 * matching is performed. Everything else in the project falls out of that
 * split: backends are runtime-selectable, the build degrades gracefully when
 * CUDA or OpenMP is absent, and any two backends can be run over the same
 * input and compared (see --verify).
 *
 * ---------------------------------------------------------------------------
 * DETERMINISM CONTRACT
 * ---------------------------------------------------------------------------
 * Every implementation of `search` MUST return the canonical match, defined as:
 *
 *   1. the object with the lowest id that occurs anywhere in the picture; then
 *   2. the row-major-first placement of that object (lowest row, then col).
 *
 * `Problem.objects` is sorted by ascending id at load time, so (1) reduces to
 * "the first matching entry in the array". (2) is implemented as a minimum
 * over the packed placement key `row * span + col`.
 *
 * This is a deliberate trade against the obvious alternative -- "return
 * whichever match a thread happens to find first, then bail". That alternative
 * is marginally faster in the best case but makes results irreproducible
 * across runs, and therefore impossible to test or to compare between
 * backends. Determinism is what buys the equivalence test suite.
 */
#ifndef MSEARCH_BACKEND_H
#define MSEARCH_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

#include "msearch/config.h"
#include "msearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MatchBackend {
    const char *name;
    const char *description;

    /* Can this backend run *right now*? Compiled-in is not enough: the CUDA
     * backend also needs a visible device. Fills `reason` when false. */
    bool (*available)(char *reason, size_t reason_len);

    /* One-time setup for a whole problem. This is where expensive, reusable
     * work belongs -- the CUDA backend uploads every object to the device
     * exactly once here rather than once per picture. */
    Status (*create)(const Problem *problem, const Config *config, void **ctx, char *err,
                     size_t err_len);

    /* Search one picture. Must honour the determinism contract above. */
    Status (*search)(void *ctx, const Picture *picture, Match *out, char *err, size_t err_len);

    void (*destroy)(void *ctx);
} MatchBackend;

/* Registry -------------------------------------------------------------- */

/* All backends compiled into this binary, available or not. */
const MatchBackend *const *msearch_backend_all(int *count);

/* Look up by name; NULL if unknown. "auto" is handled by the caller. */
const MatchBackend *msearch_backend_find(const char *name);

/* Best available backend, preferring cuda > openmp > serial. Never NULL:
 * the serial backend is unconditionally compiled in and always available. */
const MatchBackend *msearch_backend_auto(void);

void msearch_backend_print_list(void);

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_BACKEND_H */
