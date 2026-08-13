/* Runtimes decide *which process searches which picture*; backends decide
 * *how* a picture is searched. Both runtimes below produce results indexed by
 * input picture order, so the output file is identical regardless of rank
 * count or backend.
 */
#ifndef MSEARCH_RUNTIME_H
#define MSEARCH_RUNTIME_H

#include <stddef.h>

#include "msearch/backend.h"
#include "msearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single process: search every picture in order.
 * `matches` must have room for problem->num_pictures entries. */
Status msearch_run_local(const Problem *problem, const MatchBackend *backend, const Config *config,
                         Match *matches, char *err, size_t err_len);

#ifdef MSEARCH_HAVE_MPI
/* Distributed run over MPI_COMM_WORLD.
 *
 * Work is claimed dynamically from a shared counter (MPI-3 one-sided
 * fetch-and-add) rather than pushed by a master. Every rank computes -- there
 * is no dedicated scheduler process -- and pictures cost wildly different
 * amounts of work, so pull-based claiming keeps ranks busy without the master
 * becoming a bottleneck or sitting idle.
 *
 * Only rank 0's `matches` array is filled. */
Status msearch_run_mpi(const Problem *problem, const MatchBackend *backend, const Config *config,
                       Match *matches, char *err, size_t err_len);

/* Broadcast a problem read on `root` to every rank. */
Status msearch_broadcast_problem(Problem *problem, int root, char *err, size_t err_len);
#endif

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_RUNTIME_H */
