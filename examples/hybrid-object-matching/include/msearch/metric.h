/* The matching metric -- the single source of truth for every backend.
 *
 * This header is compiled by the host C compiler *and* by nvcc. Both the CPU
 * backends and the CUDA thread-per-placement kernel call `msearch_score_at`
 * verbatim, which is what makes their results bit-for-bit identical rather
 * than merely "close enough". See docs/ARCHITECTURE.md.
 */
#ifndef MSEARCH_METRIC_H
#define MSEARCH_METRIC_H

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#  define MSEARCH_HD __host__ __device__ static inline
#else
#  define MSEARCH_HD static inline
#endif

/* Denominator substituted when a picture element is zero.
 *
 * The problem statement defines the per-element difference as |(p - o) / p|,
 * which is undefined at p == 0. The reference input ships matrices that start
 * at 0, so this is not a corner case -- it is the very first placement of
 * every picture. Left unhandled it yields inf (p == 0, o != 0) or NaN
 * (p == o == 0); both compare false against the threshold, which silently
 * discards genuine matches.
 *
 * Substituting a small epsilon keeps the metric total and monotone:
 *   p == 0, o == 0  ->  0            (exact match, as it should be)
 *   p == 0, o != 0  ->  |o| / eps    (huge, so effectively a mismatch)
 * Tunable via --zero-eps so the behaviour is documented and testable rather
 * than accidental. */
#define MSEARCH_DEFAULT_ZERO_EPS 1e-9

/* Per-element contribution to the matching score. Always >= 0, which is the
 * property the early-exit optimisation relies on. */
MSEARCH_HD double msearch_term(int p, int o, double zero_eps)
{
    const double denom = (p != 0) ? (double)p : zero_eps;
    return fabs(((double)p - (double)o) / denom);
}

/* Matching score of `object` placed with its top-left corner at
 * (row, col) inside `picture`.
 *
 * Because every term is non-negative the partial sum is monotonically
 * increasing, so once it reaches `threshold` the placement can never match and
 * we stop. The check is hoisted to once per object row: per-element it would
 * add a branch to the innermost loop and block vectorisation, per-placement it
 * would give up most of the saving. For the common case (no match) this is the
 * single largest constant-factor win in the whole program.
 *
 * The returned value is only guaranteed exact when it is < threshold; on early
 * exit it is a lower bound. Callers only ever compare against threshold. */
MSEARCH_HD double msearch_score_at(const int *picture, int n, const int *object, int m, int row,
                                   int col, double threshold, double zero_eps)
{
    double sum = 0.0;
    for (int y = 0; y < m; ++y) {
        const int *picture_row = picture + (size_t)(row + y) * (size_t)n + (size_t)col;
        const int *object_row = object + (size_t)y * (size_t)m;
        for (int x = 0; x < m; ++x) {
            sum += msearch_term(picture_row[x], object_row[x], zero_eps);
        }
        if (sum >= threshold) {
            return sum;
        }
    }
    return sum;
}

#endif /* MSEARCH_METRIC_H */
