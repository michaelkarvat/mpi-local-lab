#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "msearch/backend.h"
#include "msearch/config.h"
#include "msearch/log.h"
#include "msearch/metric.h"

#ifndef MSEARCH_VERSION
#  define MSEARCH_VERSION "0.0.0-dev"
#endif

const char *msearch_version(void)
{
    return MSEARCH_VERSION;
}

void msearch_config_defaults(Config *config)
{
    config->input_path = "input.txt";
    config->output_path = "output.txt";
    config->backend_name = "auto";
    config->threads = 0;
    config->device = -1;
    config->node_rank = 0;
    config->zero_eps = MSEARCH_DEFAULT_ZERO_EPS;
    config->bench_reps = 0;
    config->verify = false;
    config->log_level = LOG_INFO;
}

void msearch_print_usage(const char *program)
{
    printf(
        "Hybrid parallel object matching -- find small matrices inside large ones.\n"
        "\n"
        "Usage: %s [options] [input]\n"
        "\n"
        "Options:\n"
        "  -i, --input PATH     problem description to read (default: input.txt)\n"
        "  -o, --output PATH    results file, '-' for stdout (default: output.txt)\n"
        "  -b, --backend NAME   serial | openmp | cuda | auto (default: auto)\n"
        "  -t, --threads N      CPU threads for the openmp backend (default: all)\n"
        "  -d, --device N       CUDA device ordinal (default: derived from rank)\n"
        "      --zero-eps EPS   denominator used where a picture element is 0\n"
        "                       (default: %g; see docs/ARCHITECTURE.md)\n"
        "      --verify         run every available backend and compare results\n"
        "      --bench N        repeat the search N times and report timings\n"
        "      --list-backends  show which backends this binary can use\n"
        "  -q, --quiet          errors only\n"
        "  -v, --verbose        per-picture progress (repeat for debug output)\n"
        "  -h, --help           this message\n"
        "  -V, --version        version string\n"
        "\n"
        "Examples:\n"
        "  %s -i tests/data/planted.txt -o -\n"
        "  %s --backend openmp --threads 8 --bench 5\n"
        "  mpirun -n 4 %s --backend cuda -i big.txt\n",
        program, MSEARCH_DEFAULT_ZERO_EPS, program, program, program);
}

/* Fetch the value that follows a flag, or report the flag as incomplete. */
static const char *take_value(int argc, char **argv, int *i, const char *flag, char *err,
                              size_t err_len)
{
    if (*i + 1 >= argc) {
        msearch_set_err(err, err_len, "%s requires a value", flag);
        return NULL;
    }
    return argv[++(*i)];
}

static Status parse_int_arg(const char *text, const char *flag, int min_value, int *out, char *err,
                            size_t err_len)
{
    char *end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || value < min_value || value > 1000000) {
        msearch_set_err(err, err_len, "%s: expected an integer >= %d, got '%s'", flag, min_value,
                        text);
        return MSEARCH_ERR_INVALID;
    }
    *out = (int)value;
    return MSEARCH_OK;
}

Status msearch_config_parse(Config *config, int argc, char **argv, EarlyAction *action, char *err,
                            size_t err_len)
{
    int verbosity = 0;
    *action = EARLY_NONE;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *value = NULL;
        Status status = MSEARCH_OK;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            *action = EARLY_HELP;
            return MSEARCH_OK;
        }
        if (strcmp(arg, "-V") == 0 || strcmp(arg, "--version") == 0) {
            *action = EARLY_VERSION;
            return MSEARCH_OK;
        }
        if (strcmp(arg, "--list-backends") == 0) {
            *action = EARLY_LIST_BACKENDS;
            return MSEARCH_OK;
        }

        if (strcmp(arg, "-i") == 0 || strcmp(arg, "--input") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            config->input_path = value;
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            config->output_path = value;
        } else if (strcmp(arg, "-b") == 0 || strcmp(arg, "--backend") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            if (strcmp(value, "auto") != 0 && msearch_backend_find(value) == NULL) {
                msearch_set_err(err, err_len,
                                "unknown backend '%s' (try --list-backends)", value);
                return MSEARCH_ERR_INVALID;
            }
            config->backend_name = value;
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            status = parse_int_arg(value, arg, 1, &config->threads, err, err_len);
        } else if (strcmp(arg, "-d") == 0 || strcmp(arg, "--device") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            status = parse_int_arg(value, arg, 0, &config->device, err, err_len);
        } else if (strcmp(arg, "--bench") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            status = parse_int_arg(value, arg, 1, &config->bench_reps, err, err_len);
        } else if (strcmp(arg, "--zero-eps") == 0) {
            value = take_value(argc, argv, &i, arg, err, err_len);
            if (value == NULL) {
                return MSEARCH_ERR_INVALID;
            }
            char *end = NULL;
            config->zero_eps = strtod(value, &end);
            if (end == value || *end != '\0' || !(config->zero_eps > 0.0)) {
                msearch_set_err(err, err_len, "--zero-eps: expected a positive number, got '%s'",
                                value);
                return MSEARCH_ERR_INVALID;
            }
        } else if (strcmp(arg, "--verify") == 0) {
            config->verify = true;
        } else if (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0) {
            verbosity = -1;
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            if (verbosity < 1) {
                verbosity = 1;
            } else {
                verbosity = 2;
            }
        } else if (arg[0] == '-' && arg[1] != '\0') {
            msearch_set_err(err, err_len, "unknown option '%s' (try --help)", arg);
            return MSEARCH_ERR_INVALID;
        } else {
            config->input_path = arg; /* bare positional = input path */
        }

        if (status != MSEARCH_OK) {
            return status;
        }
    }

    if (verbosity < 0) {
        config->log_level = LOG_ERROR;
    } else if (verbosity == 1) {
        config->log_level = LOG_INFO;
    } else if (verbosity >= 2) {
        config->log_level = LOG_DEBUG;
    }
    return MSEARCH_OK;
}
