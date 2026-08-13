#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "msearch/io.h"
#include "msearch/log.h"

void msearch_format_match(const Match *match, char *buf, size_t buf_len)
{
    if (msearch_match_found(match)) {
        snprintf(buf, buf_len, "Picture %d found Object %d in Position(%d,%d)", match->picture_id,
                 match->object_id, match->row, match->col);
    } else {
        snprintf(buf, buf_len, "Picture %d No Objects were found", match->picture_id);
    }
}

Status msearch_write_results(const char *path, const Match *matches, int count, char *err,
                             size_t err_len)
{
    const bool to_stdout = (strcmp(path, "-") == 0);
    /* Binary mode, deliberately: in text mode Windows expands '\n' to CRLF, so
     * the same input would produce different bytes on different platforms.
     * Byte-identical output across backends, rank counts *and* operating
     * systems is the property the golden-file tests check. */
    FILE *fp = to_stdout ? stdout : fopen(path, "wb");
    if (fp == NULL) {
        msearch_set_err(err, err_len, "cannot open '%s' for writing: %s", path, strerror(errno));
        return MSEARCH_ERR_IO;
    }

    for (int i = 0; i < count; ++i) {
        char line[128];
        msearch_format_match(&matches[i], line, sizeof(line));
        if (fprintf(fp, "%s\n", line) < 0) {
            msearch_set_err(err, err_len, "write to '%s' failed: %s", path, strerror(errno));
            if (!to_stdout) {
                fclose(fp);
            }
            return MSEARCH_ERR_IO;
        }
    }

    /* fclose can fail on a full disk after every fprintf succeeded, so the
     * return value is checked rather than discarded. */
    if (!to_stdout && fclose(fp) != 0) {
        msearch_set_err(err, err_len, "closing '%s' failed: %s", path, strerror(errno));
        return MSEARCH_ERR_IO;
    }
    return MSEARCH_OK;
}
