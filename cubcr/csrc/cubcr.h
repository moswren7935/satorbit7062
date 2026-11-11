#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUBCR_DMAX 3
#define CUBCR_AMAX 8

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


int cubcr_cpu_solve_f64(const cubcr_model* m, int k, const uint64_t* masks, int mask_step_stride,
                        int64_t B, double* out, int n_threads, int block);
int cubcr_cpu_solve_f32(const cubcr_model* m, int k, const uint64_t* masks, int mask_step_stride,
                        int64_t B, double* out, int n_threads, int block);
const char* cubcr_cpu_last_error(void);
int cubcr_cpu_max_threads(void);

#ifdef __cplusplus
}
#endif
