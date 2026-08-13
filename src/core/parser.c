/* Tokenising reader for the problem-description format.
 *
 * The original reader required exactly one integer per line, which turned a
 * 20x20 picture into 400 lines and made fixtures unreadable. Treating the file
 * as a whitespace-separated token stream is both less code and strictly more
 * permissive: every previously valid file still parses, and matrices may now
 * be laid out as rows with `#` comments.
 */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "msearch/io.h"
#include "msearch/log.h"
#include "msearch/problem.h"

#define TOKEN_MAX 64

typedef struct {
    FILE *fp;
    const char *path;
    long line;
    char token[TOKEN_MAX];
} Lexer;

static void lexer_init(Lexer *lexer, FILE *fp, const char *path)
{
    lexer->fp = fp;
    lexer->path = path;
    lexer->line = 1;
    lexer->token[0] = '\0';
}

static int lexer_getc(Lexer *lexer)
{
    const int c = fgetc(lexer->fp);
    if (c == '\n') {
        lexer->line++;
    }
    return c;
}

/* Read the next whitespace-delimited token into lexer->token.
 * Returns false at end of input or on an over-long token. */
static bool lexer_next(Lexer *lexer, char *err, size_t err_len)
{
    int c;
    for (;;) {
        do {
            c = lexer_getc(lexer);
        } while (c != EOF && isspace((unsigned char)c));

        if (c == EOF) {
            return false;
        }
        if (c != '#') {
            break;
        }
        while (c != EOF && c != '\n') { /* comment runs to end of line */
            c = lexer_getc(lexer);
        }
        if (c == EOF) {
            return false;
        }
    }

    size_t len = 0;
    while (c != EOF && !isspace((unsigned char)c)) {
        if (len + 1 >= TOKEN_MAX) {
            msearch_set_err(err, err_len, "%s:%ld: token longer than %d characters", lexer->path,
                            lexer->line, TOKEN_MAX - 1);
            return false;
        }
        lexer->token[len++] = (char)c;
        c = lexer_getc(lexer);
    }
    if (c == '\n') { /* the delimiter belonged to the *next* line */
        lexer->line--;
    }
    lexer->token[len] = '\0';
    return len > 0;
}

static Status expect_int(Lexer *lexer, const char *what, int *out, char *err, size_t err_len)
{
    if (!lexer_next(lexer, err, err_len)) {
        if (err == NULL || err[0] == '\0') {
            msearch_set_err(err, err_len, "%s:%ld: unexpected end of input, expected %s",
                            lexer->path, lexer->line, what);
        }
        return MSEARCH_ERR_PARSE;
    }
    char *end = NULL;
    errno = 0;
    const long value = strtol(lexer->token, &end, 10);
    if (end == lexer->token || *end != '\0' || errno == ERANGE || value < INT_MIN ||
        value > INT_MAX) {
        msearch_set_err(err, err_len, "%s:%ld: expected %s, got '%s'", lexer->path, lexer->line,
                        what, lexer->token);
        return MSEARCH_ERR_PARSE;
    }
    *out = (int)value;
    return MSEARCH_OK;
}

static Status expect_double(Lexer *lexer, const char *what, double *out, char *err, size_t err_len)
{
    if (!lexer_next(lexer, err, err_len)) {
        if (err == NULL || err[0] == '\0') {
            msearch_set_err(err, err_len, "%s:%ld: unexpected end of input, expected %s",
                            lexer->path, lexer->line, what);
        }
        return MSEARCH_ERR_PARSE;
    }
    char *end = NULL;
    errno = 0;
    const double value = strtod(lexer->token, &end);
    if (end == lexer->token || *end != '\0' || errno == ERANGE) {
        msearch_set_err(err, err_len, "%s:%ld: expected %s, got '%s'", lexer->path, lexer->line,
                        what, lexer->token);
        return MSEARCH_ERR_PARSE;
    }
    *out = value;
    return MSEARCH_OK;
}

static Status read_matrix(Lexer *lexer, int *data, int side, const char *what, int id, char *err,
                          size_t err_len)
{
    const size_t elems = (size_t)side * (size_t)side;
    for (size_t i = 0; i < elems; ++i) {
        char label[96];
        snprintf(label, sizeof(label), "element %zu of %zu for %s %d", i + 1, elems, what, id);
        const Status status = expect_int(lexer, label, &data[i], err, err_len);
        if (status != MSEARCH_OK) {
            return status;
        }
    }
    return MSEARCH_OK;
}

/* Counts are read before the corresponding data, so a corrupt count would ask
 * us to allocate an absurd array before the first bad token is seen. Bound it. */
#define MSEARCH_MAX_COUNT 1000000

static Status read_count(Lexer *lexer, const char *what, int *out, char *err, size_t err_len)
{
    const Status status = expect_int(lexer, what, out, err, err_len);
    if (status != MSEARCH_OK) {
        return status;
    }
    if (*out < 0 || *out > MSEARCH_MAX_COUNT) {
        msearch_set_err(err, err_len, "%s:%ld: %s must be in [0, %d], got %d", lexer->path,
                        lexer->line, what, MSEARCH_MAX_COUNT, *out);
        return MSEARCH_ERR_INVALID;
    }
    return MSEARCH_OK;
}

Status msearch_read_problem_stream(FILE *fp, const char *path_for_errors, Problem *out, char *err,
                                   size_t err_len)
{
    msearch_problem_init(out);
    if (err != NULL && err_len > 0) {
        err[0] = '\0';
    }

    Lexer lexer;
    lexer_init(&lexer, fp, path_for_errors);

    Status status = expect_double(&lexer, "matching threshold", &out->threshold, err, err_len);
    if (status != MSEARCH_OK) {
        goto fail;
    }

    int num_pictures = 0;
    status = read_count(&lexer, "picture count", &num_pictures, err, err_len);
    if (status != MSEARCH_OK) {
        goto fail;
    }

    /* Pictures and objects are allocated in two passes because the object
     * count only appears after all picture data. Allocating the picture array
     * first keeps ownership with `out` from the very first element, so any
     * failure below is cleaned up by a single msearch_problem_free. */
    status = msearch_problem_alloc(out, num_pictures, 0, err, err_len);
    if (status != MSEARCH_OK) {
        goto fail;
    }

    for (int i = 0; i < num_pictures; ++i) {
        int id = 0;
        int n = 0;
        status = expect_int(&lexer, "picture id", &id, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = expect_int(&lexer, "picture size", &n, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = msearch_picture_alloc(&out->pictures[i], id, n, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = read_matrix(&lexer, out->pictures[i].data, n, "picture", id, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
    }

    int num_objects = 0;
    status = read_count(&lexer, "object count", &num_objects, err, err_len);
    if (status != MSEARCH_OK) {
        goto fail;
    }
    if (num_objects > 0) {
        out->objects = calloc((size_t)num_objects, sizeof(Object));
        if (out->objects == NULL) {
            msearch_set_err(err, err_len, "cannot allocate %d objects", num_objects);
            status = MSEARCH_ERR_NOMEM;
            goto fail;
        }
    }
    out->num_objects = num_objects;

    for (int k = 0; k < num_objects; ++k) {
        int id = 0;
        int m = 0;
        status = expect_int(&lexer, "object id", &id, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = expect_int(&lexer, "object size", &m, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = msearch_object_alloc(&out->objects[k], id, m, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
        status = read_matrix(&lexer, out->objects[k].data, m, "object", id, err, err_len);
        if (status != MSEARCH_OK) {
            goto fail;
        }
    }

    if (lexer_next(&lexer, err, err_len)) {
        MSEARCH_LOG_WARN("%s:%ld: ignoring trailing token '%s'", path_for_errors, lexer.line,
                         lexer.token);
    }

    status = msearch_problem_validate(out, err, err_len);
    if (status != MSEARCH_OK) {
        goto fail;
    }
    return MSEARCH_OK;

fail:
    msearch_problem_free(out);
    return status;
}

Status msearch_read_problem(const char *path, Problem *out, char *err, size_t err_len)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        msearch_problem_init(out);
        msearch_set_err(err, err_len, "cannot open '%s': %s", path, strerror(errno));
        return MSEARCH_ERR_IO;
    }
    const Status status = msearch_read_problem_stream(fp, path, out, err, err_len);
    fclose(fp);
    return status;
}
