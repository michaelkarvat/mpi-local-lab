#ifndef MSEARCH_CONFIG_H
#define MSEARCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>

#include "msearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { LOG_ERROR = 0, LOG_WARN = 1, LOG_INFO = 2, LOG_DEBUG = 3 } LogLevel;

/* Fully resolved run configuration. Built once from argv, then read-only:
 * no module reaches for a global or a compile-time constant of its own. */
typedef struct {
    const char *input_path;
    const char *output_path;
    const char *backend_name; /* "auto" resolves at startup */
    int threads; /* 0 = runtime default (OMP_NUM_THREADS / cores) */
    int device;  /* CUDA device ordinal; -1 = derive from node_rank */
    /* Rank of this process within its compute node, filled in by main(). The
     * CUDA backend maps it onto a device so that N ranks on a node drive N
     * GPUs -- the standard one-rank-per-GPU idiom. 0 outside MPI runs. */
    int node_rank;
    double zero_eps;
    int bench_reps; /* 0 = disabled */
    bool verify;    /* cross-check every available backend against serial */
    LogLevel log_level;
} Config;

void msearch_config_defaults(Config *config);

/* Flags that ask the program to print something and stop. Parsing reports the
 * request rather than printing it, so that under mpirun only rank 0 speaks --
 * and so that argument parsing stays a pure function that tests can call. */
typedef enum {
    EARLY_NONE = 0,
    EARLY_HELP,
    EARLY_VERSION,
    EARLY_LIST_BACKENDS
} EarlyAction;

/* Parse argv into `config`. Returns MSEARCH_OK to continue, or fills `err`. */
Status msearch_config_parse(Config *config, int argc, char **argv, EarlyAction *action, char *err,
                            size_t err_len);

void msearch_print_usage(const char *program);
const char *msearch_version(void);

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_CONFIG_H */
