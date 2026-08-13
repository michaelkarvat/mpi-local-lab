/* Argument parsing is a pure function -- it reports what the user asked for
 * instead of printing and exiting -- which is exactly what makes it testable. */
#include <string.h>

#include "msearch/config.h"
#include "test_util.h"

static Status parse(Config *config, EarlyAction *action, char *err, size_t err_len, int argc,
                    const char **argv)
{
    msearch_config_defaults(config);
    *action = EARLY_NONE;
    err[0] = '\0';
    /* argv is char** by convention; the strings are never modified. */
    return msearch_config_parse(config, argc, (char **)(void *)argv, action, err, err_len);
}

int main(void)
{
    Config config;
    EarlyAction action;
    char err[256];

    TEST_CASE("defaults");
    {
        const char *argv[] = {"msearch"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 1, argv), MSEARCH_OK);
        CHECK_EQ_INT(action, EARLY_NONE);
        CHECK(strcmp(config.backend_name, "auto") == 0);
        CHECK(strcmp(config.input_path, "input.txt") == 0);
        CHECK(strcmp(config.output_path, "output.txt") == 0);
        CHECK_EQ_INT(config.threads, 0);
        CHECK_EQ_INT(config.device, -1);
        CHECK_EQ_INT(config.bench_reps, 0);
        CHECK(!config.verify);
    }

    TEST_CASE("long and short flags");
    {
        const char *argv[] = {"msearch", "--input", "a.txt", "-o",    "b.txt",
                              "-b",      "serial",  "-t",    "4",     "--bench",
                              "3",       "--verify"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 12, argv), MSEARCH_OK);
        CHECK(strcmp(config.input_path, "a.txt") == 0);
        CHECK(strcmp(config.output_path, "b.txt") == 0);
        CHECK(strcmp(config.backend_name, "serial") == 0);
        CHECK_EQ_INT(config.threads, 4);
        CHECK_EQ_INT(config.bench_reps, 3);
        CHECK(config.verify);
    }

    TEST_CASE("bare positional argument is the input path");
    {
        const char *argv[] = {"msearch", "problem.txt"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, argv), MSEARCH_OK);
        CHECK(strcmp(config.input_path, "problem.txt") == 0);
    }

    TEST_CASE("early actions are reported, not executed");
    {
        const char *help[] = {"msearch", "--help"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, help), MSEARCH_OK);
        CHECK_EQ_INT(action, EARLY_HELP);

        const char *version[] = {"msearch", "-V"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, version), MSEARCH_OK);
        CHECK_EQ_INT(action, EARLY_VERSION);

        const char *list[] = {"msearch", "--list-backends"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, list), MSEARCH_OK);
        CHECK_EQ_INT(action, EARLY_LIST_BACKENDS);
    }

    TEST_CASE("verbosity");
    {
        const char *quiet[] = {"msearch", "-q"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, quiet), MSEARCH_OK);
        CHECK_EQ_INT(config.log_level, LOG_ERROR);

        const char *debug[] = {"msearch", "-v", "-v"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, debug), MSEARCH_OK);
        CHECK_EQ_INT(config.log_level, LOG_DEBUG);
    }

    TEST_CASE("unknown option");
    {
        const char *argv[] = {"msearch", "--turbo"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, argv), MSEARCH_ERR_INVALID);
        CHECK_STR_CONTAINS(err, "--turbo");
    }

    TEST_CASE("flag missing its value");
    {
        const char *argv[] = {"msearch", "--threads"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 2, argv), MSEARCH_ERR_INVALID);
        CHECK_STR_CONTAINS(err, "requires a value");
    }

    TEST_CASE("unknown backend is rejected at parse time");
    {
        const char *argv[] = {"msearch", "--backend", "quantum"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, argv), MSEARCH_ERR_INVALID);
        CHECK_STR_CONTAINS(err, "quantum");
    }

    TEST_CASE("numeric validation");
    {
        const char *bad_threads[] = {"msearch", "--threads", "0"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, bad_threads),
                     MSEARCH_ERR_INVALID);

        const char *junk[] = {"msearch", "--threads", "8x"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, junk), MSEARCH_ERR_INVALID);

        const char *bad_eps[] = {"msearch", "--zero-eps", "-1"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, bad_eps), MSEARCH_ERR_INVALID);

        const char *good_eps[] = {"msearch", "--zero-eps", "1e-6"};
        CHECK_EQ_INT(parse(&config, &action, err, sizeof(err), 3, good_eps), MSEARCH_OK);
        CHECK_NEAR(config.zero_eps, 1e-6, 1e-18);
    }

    return TEST_REPORT();
}
