#include "msearch/problem.h"

#include <stdlib.h>
#include <string.h>

#include "msearch/log.h"

const char *msearch_status_str(Status status)
{
    switch (status) {
    case MSEARCH_OK:              return "ok";
    case MSEARCH_ERR_IO:          return "I/O error";
    case MSEARCH_ERR_PARSE:       return "parse error";
    case MSEARCH_ERR_INVALID:     return "invalid input";
    case MSEARCH_ERR_NOMEM:       return "out of memory";
    case MSEARCH_ERR_BACKEND:     return "backend error";
    case MSEARCH_ERR_UNAVAILABLE: return "backend unavailable";
    }
    return "unknown error";
}

void msearch_problem_init(Problem *problem)
{
    memset(problem, 0, sizeof(*problem));
}

void msearch_problem_free(Problem *problem)
{
    if (problem == NULL) {
        return;
    }
    for (int i = 0; i < problem->num_pictures; ++i) {
        free(problem->pictures[i].data);
    }
    free(problem->pictures);
    for (int k = 0; k < problem->num_objects; ++k) {
        free(problem->objects[k].data);
    }
    free(problem->objects);
    memset(problem, 0, sizeof(*problem));
}

Status msearch_problem_alloc(Problem *problem, int num_pictures, int num_objects, char *err,
                             size_t err_len)
{
    /* calloc(0, ..) may return NULL legitimately, so only treat a NULL result
     * as failure when we actually asked for elements. */
    problem->pictures = NULL;
    problem->objects = NULL;
    problem->num_pictures = num_pictures;
    problem->num_objects = num_objects;

    if (num_pictures > 0) {
        problem->pictures = calloc((size_t)num_pictures, sizeof(Picture));
        if (problem->pictures == NULL) {
            msearch_set_err(err, err_len, "cannot allocate %d pictures", num_pictures);
            return MSEARCH_ERR_NOMEM;
        }
    }
    if (num_objects > 0) {
        problem->objects = calloc((size_t)num_objects, sizeof(Object));
        if (problem->objects == NULL) {
            msearch_set_err(err, err_len, "cannot allocate %d objects", num_objects);
            return MSEARCH_ERR_NOMEM;
        }
    }
    return MSEARCH_OK;
}

/* Guard against `n * n` overflowing before it is used as an allocation size.
 * 46340^2 is the largest square that fits in a signed 32-bit int, and the
 * placement key packing in the backends relies on the same bound. */
#define MSEARCH_MAX_SIDE 46340

static Status alloc_matrix(int **data, int side, const char *what, int id, char *err,
                           size_t err_len)
{
    if (side <= 0) {
        msearch_set_err(err, err_len, "%s %d: size must be > 0, got %d", what, id, side);
        return MSEARCH_ERR_INVALID;
    }
    if (side > MSEARCH_MAX_SIDE) {
        msearch_set_err(err, err_len, "%s %d: size %d exceeds the supported maximum %d", what, id,
                        side, MSEARCH_MAX_SIDE);
        return MSEARCH_ERR_INVALID;
    }
    const size_t elems = (size_t)side * (size_t)side;
    *data = malloc(elems * sizeof(int));
    if (*data == NULL) {
        msearch_set_err(err, err_len, "%s %d: cannot allocate %zu elements", what, id, elems);
        return MSEARCH_ERR_NOMEM;
    }
    return MSEARCH_OK;
}

Status msearch_picture_alloc(Picture *picture, int id, int n, char *err, size_t err_len)
{
    picture->id = id;
    picture->n = n;
    return alloc_matrix(&picture->data, n, "picture", id, err, err_len);
}

Status msearch_object_alloc(Object *object, int id, int m, char *err, size_t err_len)
{
    object->id = id;
    object->m = m;
    return alloc_matrix(&object->data, m, "object", id, err, err_len);
}

long long msearch_placement_count(int n, int m)
{
    const long long span = (long long)n - (long long)m + 1;
    return span > 0 ? span * span : 0;
}

size_t msearch_max_picture_elems(const Problem *problem)
{
    size_t max_elems = 0;
    for (int i = 0; i < problem->num_pictures; ++i) {
        const size_t elems = (size_t)problem->pictures[i].n * (size_t)problem->pictures[i].n;
        if (elems > max_elems) {
            max_elems = elems;
        }
    }
    return max_elems;
}

/* Stable insertion sort by ascending object id.
 *
 * Stability matters: it fixes the tie-break for duplicate ids to input order,
 * so the determinism contract still holds on inputs that repeat an id. K is
 * small (tens), so an O(K^2) sort is not worth replacing. */
static void sort_objects_by_id(Object *objects, int count)
{
    for (int i = 1; i < count; ++i) {
        const Object key = objects[i];
        int j = i - 1;
        while (j >= 0 && objects[j].id > key.id) {
            objects[j + 1] = objects[j];
            --j;
        }
        objects[j + 1] = key;
    }
}

Status msearch_problem_validate(Problem *problem, char *err, size_t err_len)
{
    if (!(problem->threshold > 0.0)) {
        msearch_set_err(err, err_len, "threshold must be > 0, got %g", problem->threshold);
        return MSEARCH_ERR_INVALID;
    }
    if (problem->num_pictures < 0 || problem->num_objects < 0) {
        msearch_set_err(err, err_len, "picture/object counts must be >= 0");
        return MSEARCH_ERR_INVALID;
    }

    for (int i = 0; i < problem->num_pictures; ++i) {
        const Picture *picture = &problem->pictures[i];
        if (picture->n <= 0 || picture->data == NULL) {
            msearch_set_err(err, err_len, "picture %d: incomplete", picture->id);
            return MSEARCH_ERR_INVALID;
        }
        if (msearch_placement_count(picture->n, 1) > MSEARCH_NO_PLACEMENT) {
            msearch_set_err(err, err_len, "picture %d: %dx%d has too many placements to index",
                            picture->id, picture->n, picture->n);
            return MSEARCH_ERR_INVALID;
        }
    }
    for (int k = 0; k < problem->num_objects; ++k) {
        const Object *object = &problem->objects[k];
        if (object->m <= 0 || object->data == NULL) {
            msearch_set_err(err, err_len, "object %d: incomplete", object->id);
            return MSEARCH_ERR_INVALID;
        }
    }

    /* An object larger than every picture can never match. That is legal, but
     * it is nearly always a mistake in the input, so say so once. */
    int max_n = 0;
    for (int i = 0; i < problem->num_pictures; ++i) {
        if (problem->pictures[i].n > max_n) {
            max_n = problem->pictures[i].n;
        }
    }
    for (int k = 0; k < problem->num_objects; ++k) {
        if (problem->objects[k].m > max_n) {
            MSEARCH_LOG_WARN("object %d (%dx%d) is larger than every picture (max %dx%d)"
                             " and can never match",
                             problem->objects[k].id, problem->objects[k].m, problem->objects[k].m,
                             max_n, max_n);
        }
    }

    sort_objects_by_id(problem->objects, problem->num_objects);
    return MSEARCH_OK;
}
