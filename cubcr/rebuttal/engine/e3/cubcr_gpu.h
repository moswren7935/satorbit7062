#pragma once
#include "cubcr.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct cubcr_gpu_ctx cubcr_gpu_ctx;


cubcr_gpu_ctx* cubcr_gpu_create(const cubcr_model* m, int device);
void cubcr_gpu_destroy(cubcr_gpu_ctx* ctx);


int cubcr_gpu_solve(cubcr_gpu_ctx* ctx, int k, const uint64_t* masks, int mask_step_stride,
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
int cubcr_gpu_last_stats(cubcr_gpu_ctx* ctx, cubcr_gpu_stats* st);
int cubcr_gpu_mem_info(int device, size_t* free_b, size_t* total_b);
int cubcr_gpu_device_name(int device, char* buf, int len);
int cubcr_gpu_device_count(void);
const char* cubcr_gpu_last_error(void);
#ifdef __cplusplus
}
#endif
