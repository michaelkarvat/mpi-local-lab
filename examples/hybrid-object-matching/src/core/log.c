#include "msearch/log.h"

#include <stdarg.h>
#include <stdio.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#endif

static LogLevel g_level = LOG_INFO;
static int g_rank = -1;

void msearch_log_init(LogLevel level, int rank)
{
    g_level = level;
    g_rank = rank;
}

void msearch_log(LogLevel level, const char *fmt, ...)
{
    static const char *const kLabels[] = {"error", "warn", "info", "debug"};

    if (level > g_level) {
        return;
    }
    if (g_rank >= 0) {
        fprintf(stderr, "[%-5s rank %d] ", kLabels[level], g_rank);
    } else {
        fprintf(stderr, "[%-5s] ", kLabels[level]);
    }

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}

void msearch_set_err(char *err, size_t err_len, const char *fmt, ...)
{
    if (err == NULL || err_len == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, err_len, fmt, args);
    va_end(args);
}

double msearch_wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}
