#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUBCR_DMAX 4
#define CUBCR_AMAX 8
#define CUBCR_E5_RMAX 512

typedef struct {
    int32_t  n_agents;
    int64_t  n_states;
    int64_t  init_state;
    int64_t  n_rows;
    int64_t  nnz;
    const uint8_t*  state_kind;
    const int64_t*  state_row_start;
    const int16_t*  state_agent;
    const uint8_t*  state_radix;
    const double*   weights;
    const int64_t*  row_ptr;
    const int32_t*  col;
    const double*   val;
} cubcr_model;

typedef struct cubcr_gpu_ctx cubcr_gpu_ctx;


#define CUBCR_E5_MODE_SMEM 0
#define CUBCR_E5_MODE_ACC 1

cubcr_gpu_ctx* cubcr_e5_create(const cubcr_model* m, int device, int mode);
void cubcr_e5_destroy(cubcr_gpu_ctx* ctx);
int cubcr_e5_solve(cubcr_gpu_ctx* ctx, int k, const uint64_t* masks, int mask_step_stride,
                   int64_t B, double* out, int dtype, int64_t max_batch, int use_graph, int tpb,
                   int64_t l2_tile);

typedef struct {
    double  total_ms;
    double  kernel_ms;
    int64_t batch;
    int     waves;
    int     rmax_class;
    int     tpb;
    size_t  bytes_per_wave;
    int64_t l2_tile;
    size_t  table_bytes;
    int     graph_captures;
} cubcr_gpu_stats;

typedef struct {
    int     rmax_class;
    int     tpb;
    int64_t shm_bytes;
    int64_t n_states;
    int     fmax;
    int     blocks_per_sm;
    int     num_regs;
    int64_t local_bytes;
    int     max_threads_per_block;
} cubcr_e5_class_info;

int cubcr_e5_last_stats(cubcr_gpu_ctx* ctx, cubcr_gpu_stats* st);
int cubcr_e5_class_info_get(cubcr_gpu_ctx* ctx, int max_n, cubcr_e5_class_info* out);
int cubcr_e5_mem_info(int device, size_t* free_b, size_t* total_b);
int cubcr_e5_device_attr(int device, int* smem_optin, int* smem_per_sm, int* regs_per_sm,
                         int* max_threads_per_sm, int* sm_count);
int cubcr_e5_dmax(void);
int cubcr_e5_rmax(void);
const char* cubcr_e5_last_error(void);

#ifdef __cplusplus
}
#endif
