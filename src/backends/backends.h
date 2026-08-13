/* Private registry wiring. Application code goes through msearch/backend.h. */
#ifndef MSEARCH_BACKENDS_INTERNAL_H
#define MSEARCH_BACKENDS_INTERNAL_H

#include "msearch/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const MatchBackend msearch_backend_serial;

#ifdef MSEARCH_HAVE_OPENMP
extern const MatchBackend msearch_backend_openmp;
#endif

#ifdef MSEARCH_HAVE_CUDA
extern const MatchBackend msearch_backend_cuda;
#endif

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_BACKENDS_INTERNAL_H */
