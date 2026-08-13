#include <stdio.h>
#include <stdlib.h>

#include "msearch/io.h"
#include "msearch/problem.h"
#include "test_util.h"

#define TMP_PATH "test_parser_tmp.txt"

static Status parse_text(const char *text, Problem *out, char *err, size_t err_len)
{
    FILE *fp = fopen(TMP_PATH, "w");
    if (fp == NULL) {
        printf("   cannot create %s\n", TMP_PATH);
        return MSEARCH_ERR_IO;
    }
    fputs(text, fp);
    fclose(fp);

    const Status status = msearch_read_problem(TMP_PATH, out, err, err_len);
    remove(TMP_PATH);
    return status;
}

int main(void)
{
    Problem problem;
    char err[512];

    TEST_CASE("legacy one-integer-per-line input still parses");
    /* threshold, 1 picture (id 7, 2x2), 1 object (id 9, 1x1) */
    Status status = parse_text("2.5\n1\n7\n2\n1\n2\n3\n4\n1\n9\n1\n5\n", &problem, err,
                               sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_OK);
    if (status == MSEARCH_OK) {
        CHECK_NEAR(problem.threshold, 2.5, 1e-12);
        CHECK_EQ_INT(problem.num_pictures, 1);
        CHECK_EQ_INT(problem.pictures[0].id, 7);
        CHECK_EQ_INT(problem.pictures[0].n, 2);
        CHECK_EQ_INT(problem.pictures[0].data[3], 4);
        CHECK_EQ_INT(problem.num_objects, 1);
        CHECK_EQ_INT(problem.objects[0].id, 9);
        CHECK_EQ_INT(problem.objects[0].data[0], 5);
        msearch_problem_free(&problem);
    }

    TEST_CASE("free-form layout and comments");
    status = parse_text("# a readable fixture\n"
                        "2.5\n"
                        "1\n"
                        "7 2      # id, size\n"
                        "  1 2\n"
                        "  3 4\n"
                        "1\n"
                        "9 1\n"
                        "  5\n",
                        &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_OK);
    if (status == MSEARCH_OK) {
        CHECK_EQ_INT(problem.pictures[0].data[2], 3);
        CHECK_EQ_INT(problem.objects[0].data[0], 5);
        msearch_problem_free(&problem);
    }

    TEST_CASE("objects are sorted by ascending id");
    /* The determinism contract reduces "lowest matching object id" to "first
     * matching array entry", which only holds if loading sorts. */
    status = parse_text("1.0 1  1 1  0   3  30 1 3  10 1 1  20 1 2\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_OK);
    if (status == MSEARCH_OK) {
        CHECK_EQ_INT(problem.num_objects, 3);
        CHECK_EQ_INT(problem.objects[0].id, 10);
        CHECK_EQ_INT(problem.objects[1].id, 20);
        CHECK_EQ_INT(problem.objects[2].id, 30);
        /* Payloads must travel with their ids, not be left behind. */
        CHECK_EQ_INT(problem.objects[0].data[0], 1);
        CHECK_EQ_INT(problem.objects[1].data[0], 2);
        CHECK_EQ_INT(problem.objects[2].data[0], 3);
        msearch_problem_free(&problem);
    }

    TEST_CASE("trailing tokens are tolerated");
    status = parse_text("1.0 1 7 1 0 0 junk-is-ignored\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_OK);
    if (status == MSEARCH_OK) {
        msearch_problem_free(&problem);
    }

    TEST_CASE("empty input");
    status = parse_text("", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_PARSE);
    CHECK_STR_CONTAINS(err, "threshold");

    TEST_CASE("non-numeric token is reported with its location");
    status = parse_text("1.0\n2\n7\nbanana\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_PARSE);
    CHECK_STR_CONTAINS(err, "banana");
    CHECK_STR_CONTAINS(err, "picture size");

    TEST_CASE("truncated matrix names the missing element");
    status = parse_text("1.0\n1\n7 3\n1 2 3 4 5\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_PARSE);
    CHECK_STR_CONTAINS(err, "element 6 of 9");

    TEST_CASE("non-positive matrix size");
    status = parse_text("1.0\n1\n7 0\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_INVALID);
    CHECK_STR_CONTAINS(err, "size must be > 0");

    TEST_CASE("non-positive threshold");
    status = parse_text("0\n0\n0\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_INVALID);
    CHECK_STR_CONTAINS(err, "threshold");

    TEST_CASE("absurd counts are rejected before allocating");
    status = parse_text("1.0\n-5\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_INVALID);
    status = parse_text("1.0\n999999999\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_INVALID);

    TEST_CASE("missing file reports an I/O error, not a parse error");
    status = msearch_read_problem("definitely-not-here.txt", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_ERR_IO);
    CHECK_STR_CONTAINS(err, "definitely-not-here.txt");

    TEST_CASE("a problem with no objects is legal");
    status = parse_text("1.0 1 7 1 42 0\n", &problem, err, sizeof(err));
    CHECK_EQ_INT(status, MSEARCH_OK);
    if (status == MSEARCH_OK) {
        CHECK_EQ_INT(problem.num_objects, 0);
        msearch_problem_free(&problem);
    }

    TEST_CASE("result formatting");
    {
        char line[128];
        const Match found = {101, 202, 3, 4};
        msearch_format_match(&found, line, sizeof(line));
        CHECK_STR_CONTAINS(line, "Picture 101 found Object 202 in Position(3,4)");

        const Match none = msearch_match_none(101);
        msearch_format_match(&none, line, sizeof(line));
        CHECK_STR_CONTAINS(line, "Picture 101 No Objects were found");
    }

    return TEST_REPORT();
}
