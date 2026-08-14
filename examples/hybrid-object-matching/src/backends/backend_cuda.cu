/* GPU backend.
 *
 * Design summary (see docs/ARCHITECTURE.md for the reasoning):
 *
 *  - Objects are uploaded to the device exactly once, in create(), packed into
 *    a single allocation. The previous implementation re-uploaded the picture
 *    *and* an object for every (picture, object) pair from every OpenMP
 *    thread, so an N x N picture crossed PCIe K times per picture.
 *  - A picture is uploaded once per search(), into a buffer preallocated for
 *    the largest picture in the problem. No allocation happens on the hot path.
 *  - One kernel is launched per object, all on a single stream, all async.
 *    A kernel first checks whether a lower-id object already matched and exits
 *    immediately if so, which gives cross-object early exit *without* a
 *    host-device round trip between objects: a whole picture costs one
 *    synchronisation and one K-int copy back.
 *  - One kernel: a thread per placement, reusing the shared metric so results
 *    are bit-identical to the CPU backends and the early exit applies here too.
 *  - Determinism comes from atomicMin over the packed placement key, not from
 *    a race on a found-flag.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "msearch/backend.h"
#include "msearch/log.h"
#include "msearch/metric.h"
#include "msearch/problem.h"

/* Threads per block. 256 balances occupancy against the register pressure of
 * the double-precision accumulator on all architectures we target. */
#define MSEARCH_CUDA_BLOCK 256

/* Cap on resident blocks; the kernel is grid-stride, so any cap is correct. */
#define MSEARCH_CUDA_BLOCKS_PER_SM 32

/* ------------------------------------------------------------------ kernels */

/* True if any lower-id object already matched this picture. Kernels for
 * different objects run on the same stream, so those writes are visible here. */
__device__ __forceinline__ bool earlier_object_matched(const int *results, int object_index)
{
    for (int j = 0; j < object_index; ++j) {
        if (((const volatile int *)results)[j] != MSEARCH_NO_PLACEMENT) {
            return true;
        }
    }
    return false;
}

/* One thread per placement, calling the shared host/device msearch_score_at.
 *
 * Two properties follow from reusing that function rather than hand-writing a
 * device version: the summation order matches the CPU backends exactly, so
 * results are bit-identical rather than merely close; and the per-placement
 * early exit applies on the GPU too.
 *
 * An earlier revision also carried a warp-per-placement kernel, on the
 * reasoning that one thread walking m*m elements would serialise too much for
 * large objects. Measurement killed it -- see the "one kernel, not two" note
 * in docs/ARCHITECTURE.md. The short version: spreading a placement across a
 * warp requires summing all m*m terms before the total can be tested, which
 * forfeits the early exit, and the early exit is worth far more than the extra
 * lanes. It lost at every object size and threshold tried, by up to 16x. */
__global__ void k_search_thread_per_placement(const int *__restrict__ picture, int n,
                                              const int *__restrict__ object, int m, int span,
                                              long long placements, double threshold,
                                              double zero_eps, int *__restrict__ results,
                                              int object_index)
{
    if (earlier_object_matched(results, object_index)) {
        return;
    }
    int *best = results + object_index;

    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long key = (long long)blockIdx.x * blockDim.x + threadIdx.x; key < placements;
         key += stride) {
        /* Keys only increase along this loop, so once the running minimum is
         * at or below our key we can never improve it. */
        if (key >= *(const volatile int *)best) {
            return;
        }
        const int row = (int)(key / span);
        const int col = (int)(key % span);
        if (msearch_score_at(picture, n, object, m, row, col, threshold, zero_eps) < threshold) {
            atomicMin(best, (int)key);
        }
    }
}

/* ------------------------------------------------------------------ context */

struct CudaContext {
    const Problem *problem;
    double zero_eps;
    int device;
    int sm_count;
    cudaStream_t stream;

    int *d_objects;     /* every object, packed back to back */
    size_t *obj_offset; /* host-side offset of each object   */
    int *d_picture;     /* sized for the largest picture     */
    int *d_results;     /* one placement key per object      */
    int *h_results;     /* pinned, for the single copy back  */
    int *h_picture;     /* pinned staging for the upload     */
    size_t picture_capacity;
};

/* Error plumbing: every CUDA call is checked. The macro assumes `err`,
 * `err_len`, `status` and a `fail:` label are in scope -- an explicit contract
 * that keeps call sites readable in a file where every second line is a
 * fallible runtime call. */
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t cuda_err_ = (call);                                                \
        if (cuda_err_ != cudaSuccess) {                                                      \
            msearch_set_err(err, err_len, "CUDA error in %s (%s:%d): %s", #call, __FILE__,   \
                            __LINE__, cudaGetErrorString(cuda_err_));                        \
            status = MSEARCH_ERR_BACKEND;                                                    \
            goto fail;                                                                       \
        }                                                                                    \
    } while (0)

static void cuda_destroy(void *ctx)
{
    CudaContext *c = static_cast<CudaContext *>(ctx);
    if (c == NULL) {
        return;
    }
    /* Best-effort teardown: a failure here cannot be reported to anyone and
     * the process is about to release the context anyway. */
    cudaSetDevice(c->device);
    if (c->stream != NULL) {
        cudaStreamDestroy(c->stream);
    }
    cudaFree(c->d_objects);
    cudaFree(c->d_picture);
    cudaFree(c->d_results);
    cudaFreeHost(c->h_results);
    cudaFreeHost(c->h_picture);
    free(c->obj_offset);
    free(c);
}

static bool cuda_available(char *reason, size_t reason_len)
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        msearch_set_err(reason, reason_len, "cudaGetDeviceCount: %s", cudaGetErrorString(err));
        return false;
    }
    if (count <= 0) {
        msearch_set_err(reason, reason_len, "no CUDA device visible");
        return false;
    }
    return true;
}

static Status cuda_create(const Problem *problem, const Config *config, void **ctx, char *err,
                          size_t err_len)
{
    /* Declared up front because CUDA_CHECK jumps to `fail`, and C++ forbids a
     * goto that skips over an initialisation in the same scope. */
    Status status = MSEARCH_OK;
    CudaContext *c = NULL;
    int device_count = 0;
    size_t total_object_elems = 0;
    size_t offset = 0;
    cudaDeviceProp prop;

    c = static_cast<CudaContext *>(calloc(1, sizeof(CudaContext)));
    if (c == NULL) {
        msearch_set_err(err, err_len, "cannot allocate CUDA context");
        return MSEARCH_ERR_NOMEM;
    }
    c->problem = problem;
    c->zero_eps = config->zero_eps;

    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        msearch_set_err(err, err_len, "no CUDA device visible");
        status = MSEARCH_ERR_UNAVAILABLE;
        goto fail;
    }
    /* One rank per GPU: ranks sharing a node fan out across its devices. */
    c->device = (config->device >= 0) ? config->device : (config->node_rank % device_count);
    if (c->device >= device_count) {
        msearch_set_err(err, err_len, "device %d requested but only %d visible", c->device,
                        device_count);
        status = MSEARCH_ERR_INVALID;
        goto fail;
    }
    CUDA_CHECK(cudaSetDevice(c->device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, c->device));
    c->sm_count = prop.multiProcessorCount;
    MSEARCH_LOG_INFO("cuda backend: device %d (%s, %d SMs)", c->device, prop.name, c->sm_count);

    CUDA_CHECK(cudaStreamCreate(&c->stream));

    /* --- upload every object once, packed into one allocation --- */
    for (int k = 0; k < problem->num_objects; ++k) {
        total_object_elems += (size_t)problem->objects[k].m * (size_t)problem->objects[k].m;
    }
    if (problem->num_objects > 0) {
        c->obj_offset = static_cast<size_t *>(calloc((size_t)problem->num_objects, sizeof(size_t)));
        if (c->obj_offset == NULL) {
            msearch_set_err(err, err_len, "cannot allocate object offset table");
            status = MSEARCH_ERR_NOMEM;
            goto fail;
        }
        CUDA_CHECK(cudaMalloc(&c->d_objects, total_object_elems * sizeof(int)));
        for (int k = 0; k < problem->num_objects; ++k) {
            const size_t elems = (size_t)problem->objects[k].m * (size_t)problem->objects[k].m;
            c->obj_offset[k] = offset;
            CUDA_CHECK(cudaMemcpy(c->d_objects + offset, problem->objects[k].data,
                                  elems * sizeof(int), cudaMemcpyHostToDevice));
            offset += elems;
        }
        CUDA_CHECK(cudaMalloc(&c->d_results, (size_t)problem->num_objects * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&c->h_results, (size_t)problem->num_objects * sizeof(int)));
    }

    /* --- one picture buffer, sized for the largest picture --- */
    c->picture_capacity = msearch_max_picture_elems(problem);
    if (c->picture_capacity > 0) {
        CUDA_CHECK(cudaMalloc(&c->d_picture, c->picture_capacity * sizeof(int)));
        /* Pinned staging turns the per-picture upload into a DMA transfer
         * instead of a staged copy through a driver bounce buffer. */
        CUDA_CHECK(cudaMallocHost(&c->h_picture, c->picture_capacity * sizeof(int)));
    }

    *ctx = c;
    return MSEARCH_OK;

fail:
    cuda_destroy(c);
    return status;
}

static Status cuda_search(void *ctx, const Picture *picture, Match *out, char *err, size_t err_len)
{
    Status status = MSEARCH_OK;
    CudaContext *c = static_cast<CudaContext *>(ctx);
    const Problem *problem = c->problem;
    const size_t picture_elems = (size_t)picture->n * (size_t)picture->n;
    const int max_blocks = c->sm_count * MSEARCH_CUDA_BLOCKS_PER_SM;

    *out = msearch_match_none(picture->id);
    if (problem->num_objects == 0) {
        return MSEARCH_OK;
    }

    CUDA_CHECK(cudaSetDevice(c->device));

    /* One picture upload for the whole search, not one per object. */
    memcpy(c->h_picture, picture->data, picture_elems * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(c->d_picture, c->h_picture, picture_elems * sizeof(int),
                               cudaMemcpyHostToDevice, c->stream));

    /* MSEARCH_NO_PLACEMENT is 0x7fffffff, so a byte-wise memset cannot produce
     * it; write the sentinel explicitly from pinned memory instead. */
    for (int k = 0; k < problem->num_objects; ++k) {
        c->h_results[k] = MSEARCH_NO_PLACEMENT;
    }
    CUDA_CHECK(cudaMemcpyAsync(c->d_results, c->h_results,
                               (size_t)problem->num_objects * sizeof(int), cudaMemcpyHostToDevice,
                               c->stream));

    /* Queue one kernel per object. They are ordered on the stream, so kernel k
     * sees the results of kernels 0..k-1 and exits immediately if a lower-id
     * object already matched. Nothing blocks until the single sync below. */
    for (int k = 0; k < problem->num_objects; ++k) {
        const Object *object = &problem->objects[k];
        const long long placements = msearch_placement_count(picture->n, object->m);
        if (placements == 0) {
            continue;
        }
        const int span = picture->n - object->m + 1;
        const int *d_object = c->d_objects + c->obj_offset[k];

        long long want = (placements + MSEARCH_CUDA_BLOCK - 1) / MSEARCH_CUDA_BLOCK;
        if (want > max_blocks) {
            want = max_blocks;
        }
        const int blocks = (int)(want > 0 ? want : 1);

        k_search_thread_per_placement<<<blocks, MSEARCH_CUDA_BLOCK, 0, c->stream>>>(
            c->d_picture, picture->n, d_object, object->m, span, placements, problem->threshold,
            c->zero_eps, c->d_results, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(c->h_results, c->d_results,
                               (size_t)problem->num_objects * sizeof(int), cudaMemcpyDeviceToHost,
                               c->stream));
    CUDA_CHECK(cudaStreamSynchronize(c->stream));

    /* Objects are id-sorted, so the first hit is the canonical one. */
    for (int k = 0; k < problem->num_objects; ++k) {
        if (c->h_results[k] != MSEARCH_NO_PLACEMENT) {
            const int span = picture->n - problem->objects[k].m + 1;
            out->object_id = problem->objects[k].id;
            out->row = c->h_results[k] / span;
            out->col = c->h_results[k] % span;
            break;
        }
    }
    return MSEARCH_OK;

fail:
    return status;
}

extern "C" const MatchBackend msearch_backend_cuda = {
    "cuda",
    "NVIDIA GPU (CUDA, persistent buffers, one thread per placement)",
    cuda_available,
    cuda_create,
    cuda_search,
    cuda_destroy,
};
