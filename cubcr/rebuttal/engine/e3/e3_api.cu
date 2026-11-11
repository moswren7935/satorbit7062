#include "cubcr_gpu.cu"
#include <omp.h>

namespace {
using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
}

struct e3_host {
    int64_t S = 0, R = 0, nnz = 0;
    int rmax = 1;
    std::vector<int64_t> tab_ofs;
    std::vector<Entry> tab;
    std::vector<uint8_t> deff;
    std::vector<uint8_t> nrow;
    std::vector<float> val32;
    std::vector<int32_t> slist;
    std::vector<int64_t> class_ofs;
    std::vector<int> class_rmax;
};

extern "C" {


int e3_context_init(int device, double* ms) {
    auto t0 = clk::now();
    CK(cudaSetDevice(device));
    CK(cudaFree(0));
    CK(cudaDeviceSynchronize());
    if (ms) *ms = ms_since(t0);
    return 0;
}


e3_host* e3_host_build(const cubcr_model* m, double* ms) {
    if (!m) { g_err = "null model"; return nullptr; }
    e3_host* h = new e3_host();
    h->S = m->n_states; h->R = m->n_rows; h->nnz = m->nnz;
    const int64_t S = m->n_states;
    auto t0 = clk::now();
    build_tables(m, h->tab_ofs, h->tab, h->deff, h->rmax);
    if (ms) ms[0] = ms_since(t0);
    t0 = clk::now();
    h->nrow.assign(m->nnz, 0);
    for (int64_t s = 0; s < S; ++s) {
        const int64_t ra = m->state_row_start[s], rb = m->state_row_start[s + 1];
        for (int64_t r = ra; r < rb; ++r)
            for (int64_t p = m->row_ptr[r]; p < m->row_ptr[r + 1]; ++p) h->nrow[p] = (uint8_t)(r - ra);
    }
    if (ms) ms[1] = ms_since(t0);
    t0 = clk::now();
    h->val32.resize(m->nnz);
    for (int64_t i = 0; i < m->nnz; ++i) h->val32[i] = (float)m->val[i];
    if (ms) ms[2] = ms_since(t0);
    t0 = clk::now();
    {
        const int bounds[5] = {4, 8, 16, 32, 64};
        h->slist.reserve(S);
        h->class_ofs.push_back(0);
        for (int cidx = 0; cidx < 5; ++cidx) {
            const int lo = cidx ? bounds[cidx - 1] : 0, hi = bounds[cidx];
            size_t before = h->slist.size();
            for (int64_t s = 0; s < S; ++s) {
                const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
                if (R > lo && R <= hi) h->slist.push_back((int32_t)s);
            }
            if (h->slist.size() > before) { h->class_ofs.push_back((int64_t)h->slist.size()); h->class_rmax.push_back(hi); }
        }
    }
    if (ms) ms[3] = ms_since(t0);
    return h;
}

void e3_host_free(e3_host* h) { delete h; }


void e3_host_info(const e3_host* h, int64_t* info) {
    info[0] = (int64_t)h->tab.size(); info[1] = (int64_t)(h->tab.size() * sizeof(Entry)); info[2] = h->rmax;
    info[3] = (int64_t)h->class_rmax.size(); info[4] = (int64_t)h->slist.size();
}


cubcr_gpu_ctx* e3_alloc(const e3_host* h, const cubcr_model* m, int device, double* ms) {
    if (!h || !m) { g_err = "null pointer"; return nullptr; }
    auto t0 = clk::now();
    CKP(cudaSetDevice(device));
    if (h->rmax > 64) { g_err = "more than 64 joint-action rows in a state is unsupported"; return nullptr; }
    cubcr_gpu_ctx* ctx = new cubcr_gpu_ctx();
    ctx->device = device;
    DevModel& d = ctx->dm;
    d.S = m->n_states; d.init_state = m->init_state; d.R = m->n_rows; d.nnz = m->nnz;
    ctx->rmax = h->rmax;
    const int64_t S = m->n_states, D = CUBCR_DMAX;
    int rc = 0;
    auto A = [&](void** p, size_t bytes) -> int { CK(cudaMalloc(p, bytes ? bytes : 1)); return 0; };
    rc |= A((void**)&d.kind, S);
    rc |= A((void**)&d.row_start, (S + 1) * 8);
    rc |= A((void**)&d.agent, S * D * 2);
    rc |= A((void**)&d.deff, S);
    rc |= A((void**)&d.row_ptr, (m->n_rows + 1) * 8);
    rc |= A((void**)&d.col, m->nnz * 4);
    rc |= A((void**)&d.nrow, m->nnz);
    rc |= A((void**)&d.val64, m->nnz * 8);
    rc |= A((void**)&d.val32, m->nnz * 4);
    rc |= A((void**)&d.tab_ofs, (S + 1) * 8);
    rc |= A((void**)&d.tab, h->tab.size() * sizeof(Entry));
    rc |= A((void**)&d.slist, h->slist.size() * 4);
    ctx->table_bytes = h->tab.size() * sizeof(Entry);
    ctx->class_ofs = h->class_ofs;
    ctx->class_rmax = h->class_rmax;
    if (rc) { cubcr_gpu_destroy(ctx); return nullptr; }
    if (cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking) != cudaSuccess) {
        g_err = "cudaStreamCreateWithFlags failed"; cubcr_gpu_destroy(ctx); return nullptr;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { g_err = "cudaDeviceSynchronize failed"; cubcr_gpu_destroy(ctx); return nullptr; }
    if (ms) ms[0] = ms_since(t0);
    return ctx;
}


int e3_upload(cubcr_gpu_ctx* ctx, const e3_host* h, const cubcr_model* m, double* ms, int64_t* bytes) {
    if (!ctx || !h || !m) { g_err = "null pointer"; return 1; }
    auto t0 = clk::now();
    CK(cudaSetDevice(ctx->device));
    DevModel& d = ctx->dm;
    const int64_t S = m->n_states, D = CUBCR_DMAX;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, 0));
    int64_t tot = 0;
    auto C = [&](void* dst, const void* src, size_t n) -> int {
        if (n) { CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); tot += (int64_t)n; }
        return 0;
    };
    int rc = 0;
    rc |= C((void*)d.kind, m->state_kind, S);
    rc |= C((void*)d.row_start, m->state_row_start, (S + 1) * 8);
    rc |= C((void*)d.agent, m->state_agent, S * D * 2);
    rc |= C((void*)d.deff, h->deff.data(), S);
    rc |= C((void*)d.row_ptr, m->row_ptr, (m->n_rows + 1) * 8);
    rc |= C((void*)d.col, m->col, m->nnz * 4);
    rc |= C((void*)d.nrow, h->nrow.data(), m->nnz);
    rc |= C((void*)d.val64, m->val, m->nnz * 8);
    rc |= C((void*)d.val32, h->val32.data(), m->nnz * 4);
    rc |= C((void*)d.tab_ofs, h->tab_ofs.data(), (S + 1) * 8);
    rc |= C((void*)d.tab, h->tab.data(), h->tab.size() * sizeof(Entry));
    rc |= C((void*)d.slist, h->slist.data(), h->slist.size() * 4);
    if (rc) return rc;
    CK(cudaEventRecord(e1, 0));
    CK(cudaEventSynchronize(e1));
    CK(cudaDeviceSynchronize());
    float ev = 0.f; CK(cudaEventElapsedTime(&ev, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (ms) { ms[0] = ms_since(t0); ms[1] = ev; }
    if (bytes) *bytes = tot;
    return 0;
}


int e3_update_policy(cubcr_gpu_ctx* ctx, e3_host* h, const cubcr_model* m, double* ms) {
    if (!ctx || !h || !m) { g_err = "null pointer"; return 1; }
    if (m->n_states != h->S || m->n_rows != h->R) { g_err = "structure changed (S / rows): use a full rebuild"; return 30; }
    CK(cudaSetDevice(ctx->device));
    CK(cudaStreamSynchronize(ctx->stream));
    auto t0 = clk::now();
    const int D = CUBCR_DMAX, A = CUBCR_AMAX;
    const int64_t S = m->n_states;
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t s = 0; s < S; ++s) {
        const int d = h->deff[s];
        const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
        int rad[3]; for (int l = 0; l < D; ++l) rad[l] = m->state_radix[s * D + l];
        const double* W = m->weights + (int64_t)s * D * A;
        for (int p = 0; p < (1 << d); ++p) {
            Entry* e = h->tab.data() + h->tab_ofs[s] + (int64_t)p * R;
            for (int i = 0; i < R; ++i) {
                int rr = e[i].perm, dg[3]; dg[2] = rr % rad[2]; rr /= rad[2]; dg[1] = rr % rad[1]; rr /= rad[1]; dg[0] = rr;
                double ww = 1.0;
                for (int l = 0; l < D; ++l) {
                    const bool pinned = (l >= d) || ((p >> l) & 1);
                    if (pinned) ww *= W[l * A + dg[l]];
                }
                e[i].w = ww;
            }
        }
    }
    if (ms) ms[0] = ms_since(t0);
    t0 = clk::now();
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, 0));
    const size_t nb = h->tab.size() * sizeof(Entry);
    if (nb) CK(cudaMemcpy((void*)ctx->dm.tab, h->tab.data(), nb, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e1, 0));
    CK(cudaEventSynchronize(e1));
    CK(cudaDeviceSynchronize());
    float ev = 0.f; CK(cudaEventElapsedTime(&ev, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (ms) { ms[1] = ms_since(t0); ms[2] = ev; }
    return 0;
}


int e3_update_values(cubcr_gpu_ctx* ctx, e3_host* h, const cubcr_model* m, double* ms) {
    if (!ctx || !h || !m) { g_err = "null pointer"; return 1; }
    if (m->nnz != h->nnz || m->n_rows != h->R) { g_err = "sparsity changed: use a full rebuild"; return 31; }
    CK(cudaSetDevice(ctx->device));
    CK(cudaStreamSynchronize(ctx->stream));
    auto t0 = clk::now();
    for (int64_t i = 0; i < m->nnz; ++i) h->val32[i] = (float)m->val[i];
    if (ms) ms[0] = ms_since(t0);
    t0 = clk::now();
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, 0));
    if (m->nnz) {
        CK(cudaMemcpy((void*)ctx->dm.val64, m->val, m->nnz * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy((void*)ctx->dm.val32, h->val32.data(), m->nnz * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaEventRecord(e1, 0));
    CK(cudaEventSynchronize(e1));
    CK(cudaDeviceSynchronize());
    float ev = 0.f; CK(cudaEventElapsedTime(&ev, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (ms) { ms[1] = ms_since(t0); ms[2] = ev; }
    return 0;
}


int e3_omp_max_threads(void) { return omp_get_max_threads(); }

}
