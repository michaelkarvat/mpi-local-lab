#ifndef MSEARCH_IO_H
#define MSEARCH_IO_H

#include <stddef.h>
#include <stdio.h>

#include "msearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Read a problem description.
 *
 * The format is a flat stream of whitespace-separated numbers:
 *
 *   <threshold>
 *   <picture count>
 *   for each picture:  <id> <n> <n*n integers>
 *   <object count>
 *   for each object:   <id> <m> <m*m integers>
 *
 * Any whitespace separates tokens and `#` starts a comment to end of line, so
 * a 20x20 matrix can be laid out as 20 readable rows instead of 400 lines.
 * The original one-integer-per-line inputs remain valid.
 *
 * On failure `err` receives a "path:line: message" diagnostic. */
Status msearch_read_problem(const char *path, Problem *out, char *err, size_t err_len);
Status msearch_read_problem_stream(FILE *fp, const char *path_for_errors, Problem *out, char *err,
                                   size_t err_len);

/* Write results in the order given (the runtimes always hand over input
 * picture order, so output is byte-reproducible across backends and rank
 * counts). Pass "-" to write to stdout. */
Status msearch_write_results(const char *path, const Match *matches, int count, char *err,
                             size_t err_len);
void msearch_format_match(const Match *match, char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_IO_H */
