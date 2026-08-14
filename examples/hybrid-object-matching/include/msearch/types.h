/* Core data types shared by every layer of the program.
 *
 * This header deliberately knows nothing about MPI, OpenMP or CUDA: the core
 * data model is what allows the same problem description to be handed to any
 * backend and any runtime.
 */
#ifndef MSEARCH_TYPES_H
#define MSEARCH_TYPES_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned across module boundaries. Callers pair these with a
 * caller-supplied message buffer, so the code says *what* went wrong and the
 * buffer says *where* -- the old design encoded both in ~14 distinct integers. */
typedef enum {
    MSEARCH_OK = 0,
    MSEARCH_ERR_IO,          /* file could not be opened / read / written  */
    MSEARCH_ERR_PARSE,       /* malformed input                           */
    MSEARCH_ERR_INVALID,     /* structurally valid but semantically wrong */
    MSEARCH_ERR_NOMEM,       /* allocation failure                        */
    MSEARCH_ERR_BACKEND,     /* backend (CUDA/OpenMP) runtime failure     */
    MSEARCH_ERR_UNAVAILABLE  /* requested backend not built or no device  */
} Status;

const char *msearch_status_str(Status status);

/* A Picture is the haystack: an n x n row-major integer matrix. */
typedef struct {
    int id;
    int n;
    int *data; /* n*n elements, owned by the enclosing Problem */
} Picture;

/* An Object is the needle: an m x m row-major integer matrix. */
typedef struct {
    int id;
    int m;
    int *data; /* m*m elements, owned by the enclosing Problem */
} Object;

typedef struct {
    double threshold;
    int num_pictures;
    Picture *pictures;
    int num_objects;
    Object *objects; /* kept sorted by ascending id: see msearch_problem_validate */
} Problem;

/* Result for a single picture.
 *
 * There is no separate `found` flag: `object_id >= 0` is the single source of
 * truth. Carrying both invites the two to disagree. */
typedef struct {
    int picture_id;
    int object_id; /* -1 when no object matched */
    int row;       /* -1 when no object matched */
    int col;       /* -1 when no object matched */
} Match;

static inline bool msearch_match_found(const Match *match)
{
    return match->object_id >= 0;
}

static inline Match msearch_match_none(int picture_id)
{
    Match match = {picture_id, -1, -1, -1};
    return match;
}

/* Sentinel for "no placement matched", used as the identity element of the
 * atomicMin / min-reduction that implements the determinism contract. Every
 * real placement key is in [0, placements), so INT_MAX is unambiguous. */
#define MSEARCH_NO_PLACEMENT 0x7fffffff

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_TYPES_H */
