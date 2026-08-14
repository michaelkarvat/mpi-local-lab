/* The metric is shared verbatim by every backend, so its behaviour -- above
 * all at p == 0, which the original implementation left undefined -- is pinned
 * down here. */
#include "msearch/metric.h"
#include "test_util.h"

int main(void)
{
    TEST_CASE("ordinary elements");
    CHECK_NEAR(msearch_term(10, 10, MSEARCH_DEFAULT_ZERO_EPS), 0.0, 1e-12);
    CHECK_NEAR(msearch_term(10, 5, MSEARCH_DEFAULT_ZERO_EPS), 0.5, 1e-12);
    /* The metric is |(p - o) / p|, so it is asymmetric in p and o. */
    CHECK_NEAR(msearch_term(5, 10, MSEARCH_DEFAULT_ZERO_EPS), 1.0, 1e-12);
    CHECK_NEAR(msearch_term(-4, -2, MSEARCH_DEFAULT_ZERO_EPS), 0.5, 1e-12);

    TEST_CASE("zero picture element is finite, not inf or NaN");
    /* This is the regression test for the original division by zero: the
     * reference input starts every picture at 0, so p == 0 occurs at the very
     * first placement of every picture. */
    const double zero_match = msearch_term(0, 0, MSEARCH_DEFAULT_ZERO_EPS);
    CHECK(zero_match == 0.0);
    CHECK(zero_match == zero_match); /* not NaN */

    const double zero_mismatch = msearch_term(0, 7, MSEARCH_DEFAULT_ZERO_EPS);
    CHECK(zero_mismatch > 0.0);
    CHECK(zero_mismatch < 1.0 / 0.0); /* finite, so comparisons behave */
    CHECK_NEAR(zero_mismatch, 7.0 / MSEARCH_DEFAULT_ZERO_EPS, 1.0);

    TEST_CASE("zero-eps is configurable");
    CHECK_NEAR(msearch_term(0, 3, 1.0), 3.0, 1e-12);

    TEST_CASE("exact placement scores zero");
    const int picture[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const int object[4] = {6, 7, 10, 11};
    CHECK_NEAR(msearch_score_at(picture, 4, object, 2, 1, 1, 100.0, MSEARCH_DEFAULT_ZERO_EPS), 0.0,
               1e-12);

    TEST_CASE("known non-zero score");
    /* Placement (0,0) covers {1,2,5,6} against {6,7,10,11}:
     * |1-6|/1 + |2-7|/2 + |5-10|/5 + |6-11|/6 = 5 + 2.5 + 1 + 0.8333... */
    CHECK_NEAR(msearch_score_at(picture, 4, object, 2, 0, 0, 100.0, MSEARCH_DEFAULT_ZERO_EPS),
               5.0 + 2.5 + 1.0 + 5.0 / 6.0, 1e-12);

    TEST_CASE("early exit returns a lower bound, never a false match");
    /* With a threshold of 1.0 the same placement bails after the first row,
     * so the returned value is smaller than the true score -- but it is still
     * >= threshold, which is the only property callers rely on. */
    const double bounded = msearch_score_at(picture, 4, object, 2, 0, 0, 1.0, MSEARCH_DEFAULT_ZERO_EPS);
    CHECK(bounded >= 1.0);
    CHECK(bounded <= 5.0 + 2.5 + 1.0 + 5.0 / 6.0);

    TEST_CASE("early exit never changes a match decision");
    /* Scoring the same placement with a generous and a tight threshold must
     * agree on whether it matches. */
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double loose =
                msearch_score_at(picture, 4, object, 2, row, col, 1e9, MSEARCH_DEFAULT_ZERO_EPS);
            const double tight =
                msearch_score_at(picture, 4, object, 2, row, col, 2.0, MSEARCH_DEFAULT_ZERO_EPS);
            CHECK((loose < 2.0) == (tight < 2.0));
        }
    }

    return TEST_REPORT();
}
