#ifndef MSEARCH_LOG_H
#define MSEARCH_LOG_H

#include "msearch/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Leveled logging to stderr. `rank` is stamped on every line so interleaved
 * output from an mpirun is readable; pass -1 for single-process runs.
 * stdout stays reserved for program results. */
void msearch_log_init(LogLevel level, int rank);
void msearch_log(LogLevel level, const char *fmt, ...);

#define MSEARCH_LOG_ERROR(...) msearch_log(LOG_ERROR, __VA_ARGS__)
#define MSEARCH_LOG_WARN(...) msearch_log(LOG_WARN, __VA_ARGS__)
#define MSEARCH_LOG_INFO(...) msearch_log(LOG_INFO, __VA_ARGS__)
#define MSEARCH_LOG_DEBUG(...) msearch_log(LOG_DEBUG, __VA_ARGS__)

/* Monotonic seconds, for timing. */
double msearch_wtime(void);

/* Format a diagnostic into a caller-owned buffer. Every fallible function
 * takes such a buffer instead of printing: the caller decides whether a
 * failure is fatal, and worker ranks can ship the text to rank 0. NULL `err`
 * is a no-op, so callers that do not care may pass NULL. */
void msearch_set_err(char *err, size_t err_len, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_LOG_H */
