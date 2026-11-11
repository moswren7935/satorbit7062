#include "cubcr_e5.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static std::string g_err;
extern "C" const char* cubcr_e5_last_error(void) { return g_err.c_str(); }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { g_err = std::string(#x) + ": " + cudaGetErrorString(e_); return 10; } } while (0)
#define CKP(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { g_err = std::string(#x) + ": " + cudaGetErrorString(e_); return nullptr; } } while (0)

struct Entry { double w; uint16_t perm; uint8_t grp_end; uint8_t pad; uint32_t pad2; };


static const int N_CLASSES = 8;
static const int CLASS_BOUNDS[N_CLASSES] = {4, 8, 16, 32, 64, 128, 256, 512};

struct DevModel {
    int64_t S, init_state, R, nnz;
    const uint8_t* kind;
    const int64_t* row_start;
    const int16_t* agent;
    const uint8_t* radix;
    const uint8_t* deff;
    const int64_t* row_ptr;
    const int32_t* col;
    const uint16_t* nrow;
    const double* val64;  const float* val32;
    const int64_t* tab_ofs;
    const Entry* tab;
    const int32_t* slist;
};

struct cubcr_gpu_ctx {
    int device;
    int mode;
    DevModel dm;
    int rmax;
    size_t table_bytes;
    std::vector<int64_t> class_ofs;
    std::vector<int> class_rmax;
    std::vector<int> class_fmax;
    std::vector<int> class_tpb;
    std::vector<int64_t> class_shm;
    cudaStream_t stream;
    void* V = nullptr; void* Vn = nullptr; size_t vbytes = 0;
    uint64_t* dmask = nullptr; size_t mbytes = 0;
    uint64_t* hmask = nullptr; size_t hmbytes = 0;
    void* hout = nullptr; size_t hobytes = 0;
    cudaGraphExec_t gexec = nullptr;
    struct { int64_t nb, tile; int k, mstride, dtype, tpb; void *V, *Vn, *dmask; } gkey{};
    int graph_captures = 0;
    cubcr_gpu_stats stats{};
    std::vector<cubcr_e5_class_info> cinfo;
};

template <typename T> struct Sel;
template <> struct Sel<double> { static const double* val(const DevModel& m) { return m.val64; } };
template <> struct Sel<float>  { static const float*  val(const DevModel& m) { return m.val32; } };

template <typename T>
__global__ void init_kernel(const uint8_t* __restrict__ kind, T* __restrict__ V, int64_t S, int64_t B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * B) V[i] = (kind[i / B] == 1) ? T(1) : T(0);
}


template <typename T, int RMAX>
__global__ void __launch_bounds__(256)
sweep_kernel(DevModel m, const T* __restrict__ val, const T* __restrict__ V, T* __restrict__ Vn,
             const uint64_t* __restrict__ masks, int64_t B, int64_t c_begin, int64_t c_end, int stile,
             const int32_t* __restrict__ slist, int64_t n_list) {
    extern __shared__ unsigned char smem_raw[];
    T* Q = reinterpret_cast<T*>(smem_raw);
    const int tid = threadIdx.x;
    const int TPB = blockDim.x;
    const int64_t c = c_begin + (int64_t)blockIdx.x * TPB + tid;
    if (c >= c_end) return;
    const uint64_t mk = masks[c];
    const int64_t s_begin = (int64_t)blockIdx.y * stile;
    const int64_t s_end = min(s_begin + (int64_t)stile, n_list);
    const int D = CUBCR_DMAX;
    for (int64_t si = s_begin; si < s_end; ++si) {
        const int64_t s = slist[si];
        const uint8_t kind = m.kind[s];
        if (kind == 1) { Vn[s * B + c] = T(1); continue; }
        if (kind == 2) { Vn[s * B + c] = T(0); continue; }
        const int64_t r0 = m.row_start[s];
        const int R = (int)(m.row_start[s + 1] - r0);
        {
            const int64_t p0 = m.row_ptr[r0], p1 = m.row_ptr[r0 + R];
            int cur = 0; T q = T(0);
            for (int64_t p = p0; p < p1; p += 4) {
                const int64_t pa = p, pb = min(p + 1, p1 - 1), pc = min(p + 2, p1 - 1), pd = min(p + 3, p1 - 1);
                const int ca = m.col[pa], cb = m.col[pb], cc = m.col[pc], cd = m.col[pd];
                const T va = val[pa], vb = val[pb], vc = val[pc], vd = val[pd];
                const int ra = m.nrow[pa], rb = m.nrow[pb], rc = m.nrow[pc], rd = m.nrow[pd];
                const T ga = V[(int64_t)ca * B + c];
                const T gb = V[(int64_t)cb * B + c];
                const T gc = V[(int64_t)cc * B + c];
                const T gd = V[(int64_t)cd * B + c];
                if (ra != cur) { Q[cur * TPB + tid] = q; q = T(0); cur = ra; }
                q += va * ga;
                if (p + 1 < p1) { if (rb != cur) { Q[cur * TPB + tid] = q; q = T(0); cur = rb; } q += vb * gb; }
                if (p + 2 < p1) { if (rc != cur) { Q[cur * TPB + tid] = q; q = T(0); cur = rc; } q += vc * gc; }
                if (p + 3 < p1) { if (rd != cur) { Q[cur * TPB + tid] = q; q = T(0); cur = rd; } q += vd * gd; }
            }
            Q[cur * TPB + tid] = q;
        }
        const int d = m.deff[s];
        int pat = 0;
        for (int l = 0; l < d; ++l) pat |= (int)((mk >> m.agent[s * D + l]) & 1ull) << l;
        const Entry* e = m.tab + m.tab_ofs[s] + (int64_t)pat * R;
        T best = INFINITY, acc = T(0);
        for (int i = 0; i < R; ++i) {
            const double2 ew = __ldg(reinterpret_cast<const double2*>(e + i));
            const unsigned long long bits = __double_as_longlong(ew.y);
            const int perm = (int)(bits & 0xFFFFull);
            const bool gend = ((bits >> 16) & 0xFFull) != 0;
            acc += (T)ew.x * Q[perm * TPB + tid];
            best = gend ? fmin(best, acc) : best;
            acc = gend ? T(0) : acc;
        }
        Vn[s * B + c] = best;
    }
}


template <typename T, int RMAX>
__global__ void __launch_bounds__(256)
sweep_kernel_acc(DevModel m, const T* __restrict__ val, const T* __restrict__ V, T* __restrict__ Vn,
                 const uint64_t* __restrict__ masks, int64_t B, int64_t c_begin, int64_t c_end, int stile,
                 const int32_t* __restrict__ slist, int64_t n_list) {
    extern __shared__ unsigned char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);
    const int tid = threadIdx.x;
    const int TPB = blockDim.x;
    const int64_t c = c_begin + (int64_t)blockIdx.x * TPB + tid;
    if (c >= c_end) return;
    const uint64_t mk = masks[c];
    const int64_t s_begin = (int64_t)blockIdx.y * stile;
    const int64_t s_end = min(s_begin + (int64_t)stile, n_list);
    const int D = CUBCR_DMAX;
    for (int64_t si = s_begin; si < s_end; ++si) {
        const int64_t s = slist[si];
        const uint8_t kind = m.kind[s];
        if (kind == 1) { Vn[s * B + c] = T(1); continue; }
        if (kind == 2) { Vn[s * B + c] = T(0); continue; }
        const int64_t r0 = m.row_start[s];
        const int R = (int)(m.row_start[s + 1] - r0);
        const int d = m.deff[s];
        int pat = 0, F = 1;
        for (int l = 0; l < d; ++l) {
            const int pinned = (int)((mk >> m.agent[s * D + l]) & 1ull);
            pat |= pinned << l;
            F *= pinned ? 1 : (int)m.radix[s * D + l];
        }
        const Entry* e = m.tab + m.tab_ofs[s] + (int64_t)pat * R;
        const bool single = (F == R);
        T best = INFINITY;
        if (!single) for (int g = 0; g < F; ++g) A[g * TPB + tid] = T(0);
        auto fold = [&](int r, T qv) {
            const double2 ew = __ldg(reinterpret_cast<const double2*>(e + r));
            const unsigned long long bits = __double_as_longlong(ew.y);
            const int g = (int)(bits & 0xFFFFull);
            if (single) { best = fmin(best, (T)ew.x * qv); }
            else { A[g * TPB + tid] += (T)ew.x * qv; }
        };
        {
            const int64_t p0 = m.row_ptr[r0], p1 = m.row_ptr[r0 + R];
            int cur = 0; T q = T(0);
            for (int64_t p = p0; p < p1; p += 4) {
                const int64_t pa = p, pb = min(p + 1, p1 - 1), pc = min(p + 2, p1 - 1), pd = min(p + 3, p1 - 1);
                const int ca = m.col[pa], cb = m.col[pb], cc = m.col[pc], cd = m.col[pd];
                const T va = val[pa], vb = val[pb], vc = val[pc], vd = val[pd];
                const int ra = m.nrow[pa], rb = m.nrow[pb], rc = m.nrow[pc], rd = m.nrow[pd];
                const T ga = V[(int64_t)ca * B + c];
                const T gb = V[(int64_t)cb * B + c];
                const T gc = V[(int64_t)cc * B + c];
                const T gd = V[(int64_t)cd * B + c];
                if (ra != cur) { fold(cur, q); q = T(0); cur = ra; }
                q += va * ga;
                if (p + 1 < p1) { if (rb != cur) { fold(cur, q); q = T(0); cur = rb; } q += vb * gb; }
                if (p + 2 < p1) { if (rc != cur) { fold(cur, q); q = T(0); cur = rc; } q += vc * gc; }
                if (p + 3 < p1) { if (rd != cur) { fold(cur, q); q = T(0); cur = rd; } q += vd * gd; }
            }
            fold(cur, q);
        }
        if (!single) for (int g = 0; g < F; ++g) best = fmin(best, A[g * TPB + tid]);
        Vn[s * B + c] = best;
    }
}

template <typename T>
static int upload(const void* src, size_t bytes, T** dst) {
    CK(cudaMalloc((void**)dst, bytes ? bytes : 1));
    if (bytes) CK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
    return 0;
}


static void build_tables(const cubcr_model* m, int mode, std::vector<int64_t>& tab_ofs, std::vector<Entry>& tab,
                         std::vector<uint8_t>& deff, int& rmax) {
    const int D = CUBCR_DMAX, A = CUBCR_AMAX;
    const int64_t S = m->n_states;
    tab_ofs.assign(S + 1, 0); deff.assign(S, 0); rmax = 1;
    for (int64_t s = 0; s < S; ++s) {
        int d = 0;
        for (int l = 0; l < D; ++l) if (m->state_agent[s * D + l] >= 0) d = l + 1;
        deff[s] = (uint8_t)d;
        const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
        rmax = (int)std::max<int64_t>(rmax, R);
        tab_ofs[s + 1] = tab_ofs[s] + (int64_t(1) << d) * R;
    }
    tab.resize(tab_ofs[S]);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t s = 0; s < S; ++s) {
        const int d = deff[s];
        const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
        int rad[D]; for (int l = 0; l < D; ++l) rad[l] = m->state_radix[s * D + l];
        int st[D]; st[D - 1] = 1; for (int l = D - 2; l >= 0; --l) st[l] = st[l + 1] * rad[l + 1];
        const double* W = m->weights + (int64_t)s * D * A;
        std::vector<int> key(R), order(R), gid(R); std::vector<double> w(R);
        for (int p = 0; p < (1 << d); ++p) {
            Entry* e = tab.data() + tab_ofs[s] + (int64_t)p * R;
            for (int r = 0; r < R; ++r) {
                int rr = r, dg[D];
                for (int l = D - 1; l >= 1; --l) { dg[l] = rr % rad[l]; rr /= rad[l]; }
                dg[0] = rr;
                int f = 0, g = 0; double ww = 1.0;
                for (int l = 0; l < D; ++l) {
                    const bool pinned = (l >= d) || ((p >> l) & 1);
                    if (pinned) ww *= W[l * A + dg[l]]; else { f += dg[l] * st[l]; g = g * rad[l] + dg[l]; }
                }
                key[r] = f; w[r] = ww; order[r] = r; gid[r] = g;
            }
            if (mode == CUBCR_E5_MODE_SMEM) {
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
                for (int i = 0; i < R; ++i) {
                    e[i].w = w[order[i]]; e[i].perm = (uint16_t)order[i];
                    e[i].grp_end = (i + 1 == R || key[order[i + 1]] != key[order[i]]) ? 1 : 0; e[i].pad = 0; e[i].pad2 = 0;
                }
            } else {
                for (int r = 0; r < R; ++r) { e[r].w = w[r]; e[r].perm = (uint16_t)gid[r]; e[r].grp_end = 0; e[r].pad = 0; e[r].pad2 = 0; }
            }
        }
    }
}

extern "C" int cubcr_e5_dmax(void) { return CUBCR_DMAX; }
extern "C" int cubcr_e5_rmax(void) { return CUBCR_E5_RMAX; }

extern "C" cubcr_gpu_ctx* cubcr_e5_create(const cubcr_model* m, int device, int mode) {
    if (!m) { g_err = "null model"; return nullptr; }
    if (mode != CUBCR_E5_MODE_SMEM && mode != CUBCR_E5_MODE_ACC) { g_err = "bad mode"; return nullptr; }
    CKP(cudaSetDevice(device));
    cubcr_gpu_ctx* ctx = new cubcr_gpu_ctx();
    ctx->device = device;
    ctx->mode = mode;
    DevModel& d = ctx->dm;
    d.S = m->n_states; d.init_state = m->init_state; d.R = m->n_rows; d.nnz = m->nnz;
    const int64_t S = m->n_states, D = CUBCR_DMAX;
    std::vector<int64_t> tab_ofs; std::vector<Entry> tab; std::vector<uint8_t> deff; int rmax = 1;
    build_tables(m, mode, tab_ofs, tab, deff, rmax);
    ctx->rmax = rmax;
    if (rmax > CUBCR_E5_RMAX) { g_err = "more than 512 joint-action rows in a state is unsupported"; delete ctx; return nullptr; }
    int rc = 0;
    rc |= upload(m->state_kind, S, (uint8_t**)&d.kind);
    rc |= upload(m->state_row_start, (S + 1) * 8, (int64_t**)&d.row_start);
    rc |= upload(m->state_agent, S * D * 2, (int16_t**)&d.agent);
    rc |= upload(m->state_radix, S * D, (uint8_t**)&d.radix);
    rc |= upload(deff.data(), S, (uint8_t**)&d.deff);
    rc |= upload(m->row_ptr, (m->n_rows + 1) * 8, (int64_t**)&d.row_ptr);
    rc |= upload(m->col, m->nnz * 4, (int32_t**)&d.col);
    {
        std::vector<uint16_t> nrow(m->nnz);
        for (int64_t s = 0; s < S; ++s) {
            const int64_t ra = m->state_row_start[s], rb = m->state_row_start[s + 1];
            for (int64_t r = ra; r < rb; ++r)
                for (int64_t p = m->row_ptr[r]; p < m->row_ptr[r + 1]; ++p) nrow[p] = (uint16_t)(r - ra);
        }
        rc |= upload(nrow.data(), m->nnz * 2, (uint16_t**)&d.nrow);
    }
    rc |= upload(m->val, m->nnz * 8, (double**)&d.val64);
    {
        std::vector<float> tmp(m->nnz);
        for (int64_t i = 0; i < m->nnz; ++i) tmp[i] = (float)m->val[i];
        rc |= upload(tmp.data(), m->nnz * 4, (float**)&d.val32);
    }
    rc |= upload(tab_ofs.data(), (S + 1) * 8, (int64_t**)&d.tab_ofs);
    rc |= upload(tab.data(), tab.size() * sizeof(Entry), (Entry**)&d.tab);
    ctx->table_bytes = tab.size() * sizeof(Entry);
    if (rc) { cubcr_e5_destroy(ctx); return nullptr; }
    {
        std::vector<int32_t> slist; slist.reserve(S);
        ctx->class_ofs.push_back(0);
        for (int cidx = 0; cidx < N_CLASSES; ++cidx) {
            const int lo = cidx ? CLASS_BOUNDS[cidx - 1] : 0, hi = CLASS_BOUNDS[cidx];
            size_t before = slist.size();
            int fmax = 0;
            for (int64_t s = 0; s < S; ++s) {
                const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
                if (R > lo && R <= hi) {
                    slist.push_back((int32_t)s);

                    int rmin = 0;
                    for (int l = 0; l < deff[s]; ++l) { const int r_ = m->state_radix[s * D + l]; if (r_ >= 2 && (rmin == 0 || r_ < rmin)) rmin = r_; }
                    if (rmin) fmax = std::max<int>(fmax, (int)(R / rmin));
                }
            }
            if (slist.size() > before) {
                ctx->class_ofs.push_back((int64_t)slist.size()); ctx->class_rmax.push_back(hi); ctx->class_fmax.push_back(fmax);
            }
        }
        if (upload(slist.data(), slist.size() * 4, (int32_t**)&d.slist)) { cubcr_e5_destroy(ctx); return nullptr; }
    }
    ctx->class_tpb.assign(ctx->class_rmax.size(), 0);
    ctx->class_shm.assign(ctx->class_rmax.size(), 0);
    CKP(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    return ctx;
}

extern "C" void cubcr_e5_destroy(cubcr_gpu_ctx* ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    DevModel& d = ctx->dm;
    cudaFree((void*)d.kind); cudaFree((void*)d.row_start); cudaFree((void*)d.agent); cudaFree((void*)d.radix); cudaFree((void*)d.deff);
    cudaFree((void*)d.row_ptr); cudaFree((void*)d.col); cudaFree((void*)d.nrow); cudaFree((void*)d.val64); cudaFree((void*)d.val32);
    cudaFree((void*)d.tab_ofs); cudaFree((void*)d.tab); cudaFree((void*)d.slist);
    if (ctx->V) cudaFree(ctx->V); if (ctx->Vn) cudaFree(ctx->Vn); if (ctx->dmask) cudaFree(ctx->dmask);
    if (ctx->hmask) cudaFreeHost(ctx->hmask); if (ctx->hout) cudaFreeHost(ctx->hout);
    if (ctx->gexec) cudaGraphExecDestroy(ctx->gexec);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

static int rmax_class(int rmax) {
    for (int i = 0; i < N_CLASSES; ++i) if (rmax <= CLASS_BOUNDS[i]) return CLASS_BOUNDS[i];
    return CLASS_BOUNDS[N_CLASSES - 1];
}

#define E5_DISPATCH(RC, KERNEL, ...) \
    switch (RC) { \
        case 4:   KERNEL<T, 4>__VA_ARGS__; break; \
        case 8:   KERNEL<T, 8>__VA_ARGS__; break; \
        case 16:  KERNEL<T, 16>__VA_ARGS__; break; \
        case 32:  KERNEL<T, 32>__VA_ARGS__; break; \
        case 64:  KERNEL<T, 64>__VA_ARGS__; break; \
        case 128: KERNEL<T, 128>__VA_ARGS__; break; \
        case 256: KERNEL<T, 256>__VA_ARGS__; break; \
        default:  KERNEL<T, 512>__VA_ARGS__; break; \
    }

template <typename T>
static int launch_sweep(cubcr_gpu_ctx* ctx, const T* V, T* Vn, const uint64_t* masks, int64_t B, int64_t c0, int64_t c1, int stile) {
    const DevModel& m = ctx->dm;
    const T* val = Sel<T>::val(m);
    for (size_t ci = 0; ci < ctx->class_rmax.size(); ++ci) {
        const int rcl = ctx->class_rmax[ci];
        const int tpb = ctx->class_tpb[ci];
        const int64_t l0 = ctx->class_ofs[ci], n_list = ctx->class_ofs[ci + 1] - l0;
        const int32_t* sl = m.slist + l0;
        const size_t shm = (size_t)ctx->class_shm[ci];
        dim3 grid((unsigned)((c1 - c0 + tpb - 1) / tpb), (unsigned)((n_list + stile - 1) / stile));
        if (ctx->mode == CUBCR_E5_MODE_SMEM) {
            E5_DISPATCH(rcl, sweep_kernel, <<<grid, tpb, shm, ctx->stream>>>(m, val, V, Vn, masks, B, c0, c1, stile, sl, n_list))
        } else {
            E5_DISPATCH(rcl, sweep_kernel_acc, <<<grid, tpb, shm, ctx->stream>>>(m, val, V, Vn, masks, B, c0, c1, stile, sl, n_list))
        }
        CK(cudaGetLastError());
    }
    return 0;
}

template <typename T, int RC>
static int class_attrs_k(int mode, size_t shm, int tpb, cubcr_e5_class_info* ci) {
    cudaFuncAttributes fa;
    int nb = 0;
    if (mode == CUBCR_E5_MODE_SMEM) {
        CK(cudaFuncSetAttribute(sweep_kernel<T, RC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncGetAttributes(&fa, sweep_kernel<T, RC>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, sweep_kernel<T, RC>, tpb, shm));
    } else {
        CK(cudaFuncSetAttribute(sweep_kernel_acc<T, RC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        CK(cudaFuncGetAttributes(&fa, sweep_kernel_acc<T, RC>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, sweep_kernel_acc<T, RC>, tpb, shm));
    }
    ci->blocks_per_sm = nb; ci->num_regs = fa.numRegs; ci->local_bytes = (int64_t)fa.localSizeBytes;
    ci->max_threads_per_block = fa.maxThreadsPerBlock;
    return 0;
}
template <typename T>
static int class_attrs(int rc_class, int mode, size_t shm, int tpb, cubcr_e5_class_info* ci) {
    switch (rc_class) {
        case 4:   return class_attrs_k<T, 4>(mode, shm, tpb, ci);
        case 8:   return class_attrs_k<T, 8>(mode, shm, tpb, ci);
        case 16:  return class_attrs_k<T, 16>(mode, shm, tpb, ci);
        case 32:  return class_attrs_k<T, 32>(mode, shm, tpb, ci);
        case 64:  return class_attrs_k<T, 64>(mode, shm, tpb, ci);
        case 128: return class_attrs_k<T, 128>(mode, shm, tpb, ci);
        case 256: return class_attrs_k<T, 256>(mode, shm, tpb, ci);
        default:  return class_attrs_k<T, 512>(mode, shm, tpb, ci);
    }
}

template <typename T>
static int solve_t(cubcr_gpu_ctx* ctx, int k, const uint64_t* masks, int mstride, int64_t B, double* out,
                   int64_t max_batch, int use_graph, int tpb, int64_t l2_tile) {
    auto t_start = std::chrono::steady_clock::now();
    CK(cudaSetDevice(ctx->device));
    const DevModel& m = ctx->dm;
    const int64_t S = m.S;
    const int rcl = rmax_class(ctx->rmax);
    if (tpb <= 0) tpb = (rcl >= 32) ? 128 : 256;
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx->device));
    ctx->cinfo.assign(ctx->class_rmax.size(), cubcr_e5_class_info{});
    for (size_t ci = 0; ci < ctx->class_rmax.size(); ++ci) {
        const int cl = ctx->class_rmax[ci];
        const size_t per_thread = (ctx->mode == CUBCR_E5_MODE_SMEM) ? (size_t)cl * sizeof(T)
                                                                     : (size_t)std::max(1, ctx->class_fmax[ci]) * sizeof(T);
        int t = tpb;
        if (per_thread * (size_t)t > (size_t)optin) t = (int)(((size_t)optin / per_thread) / 32 * 32);
        if (t < 32) { g_err = "row class does not fit shared memory even at 32 threads per block"; return 30; }
        ctx->class_tpb[ci] = t;
        ctx->class_shm[ci] = (int64_t)(per_thread * (size_t)t);
        cubcr_e5_class_info& inf = ctx->cinfo[ci];
        inf.rmax_class = cl; inf.tpb = t; inf.shm_bytes = ctx->class_shm[ci];
        inf.n_states = ctx->class_ofs[ci + 1] - ctx->class_ofs[ci]; inf.fmax = ctx->class_fmax[ci];
        int r = class_attrs<T>(cl, ctx->mode, (size_t)ctx->class_shm[ci], t, &inf);
        if (r) return r;
    }
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const size_t per_coal = 2 * (size_t)S * sizeof(T) + (mstride ? (size_t)k : 1) * sizeof(uint64_t) + sizeof(T);
    int64_t Bb = (int64_t)((double)free_b * 0.85 / (double)per_coal);
    if (max_batch > 0) Bb = std::min(Bb, max_batch);
    Bb = std::min(Bb, B);
    if (Bb < 1) { g_err = "not enough GPU memory for a single coalition"; return 20; }
    int64_t tile = Bb;
    if (l2_tile > 0) tile = std::min<int64_t>(l2_tile, Bb);
    const size_t vb = (size_t)S * Bb * sizeof(T);
    if (vb > ctx->vbytes) {
        if (ctx->V) cudaFree(ctx->V); if (ctx->Vn) cudaFree(ctx->Vn);
        CK(cudaMalloc(&ctx->V, vb)); CK(cudaMalloc(&ctx->Vn, vb)); ctx->vbytes = vb;
    }
    const size_t mb = (size_t)(mstride ? k : 1) * Bb * sizeof(uint64_t);
    if (mb > ctx->mbytes) { if (ctx->dmask) cudaFree(ctx->dmask); CK(cudaMalloc((void**)&ctx->dmask, mb)); ctx->mbytes = mb; }
    if (mb > ctx->hmbytes) { if (ctx->hmask) cudaFreeHost(ctx->hmask); CK(cudaMallocHost((void**)&ctx->hmask, mb)); ctx->hmbytes = mb; }
    const size_t ob = (size_t)Bb * sizeof(T);
    if (ob > ctx->hobytes) { if (ctx->hout) cudaFreeHost(ctx->hout); CK(cudaMallocHost(&ctx->hout, ob)); ctx->hobytes = ob; }
    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    float kernel_ms_total = 0.f;
    int waves = 0;
    const int dtype_id = (sizeof(T) == 8) ? 0 : 1;
    int stile_req = 16;
    if (const char* ev = getenv("CUBCR_STILE")) stile_req = atoi(ev);
    if (stile_req <= 0) stile_req = 16;
    const int stile = (int)std::max<int64_t>(1, std::min<int64_t>(S, stile_req));
    for (int64_t c0 = 0; c0 < B; c0 += Bb) {
        const int64_t nb = std::min(Bb, B - c0);
        const int nt = mstride ? k : 1;
        for (int t = 0; t < nt; ++t)
            std::memcpy(ctx->hmask + (size_t)t * nb, masks + (mstride ? (size_t)t * B : 0) + c0, nb * sizeof(uint64_t));
        CK(cudaMemcpyAsync(ctx->dmask, ctx->hmask, (size_t)nt * nb * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx->stream));
        const int64_t total = S * nb;
        const unsigned init_grid = (unsigned)std::min<int64_t>((total + 255) / 256, 0x7fffffff);
        CK(cudaEventRecord(ev0, ctx->stream));
        T* V = (T*)ctx->V; T* Vn = (T*)ctx->Vn;
        auto run_sweeps = [&]() -> int {
            init_kernel<T><<<init_grid, 256, 0, ctx->stream>>>(m.kind, V, S, nb);
            CK(cudaGetLastError());
            for (int64_t ct = 0; ct < nb; ct += tile) {
                const int64_t c1 = std::min(ct + tile, nb);
                T* Va = V; T* Vb = Vn;
                for (int t = k - 1; t >= 0; --t) {
                    const uint64_t* mk = ctx->dmask + (mstride ? (size_t)t * nb : 0);
                    int r = launch_sweep<T>(ctx, Va, Vb, mk, nb, ct, c1, stile);
                    if (r) return r;
                    std::swap(Va, Vb);
                }
            }
            return 0;
        };
        if (use_graph) {
            const bool same = ctx->gexec && ctx->gkey.nb == nb && ctx->gkey.tile == tile && ctx->gkey.k == k &&
                              ctx->gkey.mstride == mstride && ctx->gkey.dtype == dtype_id && ctx->gkey.tpb == tpb &&
                              ctx->gkey.V == ctx->V && ctx->gkey.Vn == ctx->Vn && ctx->gkey.dmask == (void*)ctx->dmask;
            if (!same) {
                if (ctx->gexec) { cudaGraphExecDestroy(ctx->gexec); ctx->gexec = nullptr; }
                cudaGraph_t graph;
                CK(cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal));
                int r = run_sweeps();
                if (r) { cudaStreamEndCapture(ctx->stream, &graph); return r; }
                CK(cudaStreamEndCapture(ctx->stream, &graph));
                CK(cudaGraphInstantiate(&ctx->gexec, graph, nullptr, nullptr, 0));
                cudaGraphDestroy(graph);
                ctx->gkey = {nb, tile, k, mstride, dtype_id, tpb, ctx->V, ctx->Vn, (void*)ctx->dmask};
                ctx->graph_captures++;
            }
            CK(cudaGraphLaunch(ctx->gexec, ctx->stream));
        } else {
            int r = run_sweeps();
            if (r) return r;
        }
        T* Vres = (k & 1) ? (T*)ctx->Vn : (T*)ctx->V;
        CK(cudaEventRecord(ev1, ctx->stream));
        CK(cudaMemcpyAsync(ctx->hout, Vres + m.init_state * nb, nb * sizeof(T), cudaMemcpyDeviceToHost, ctx->stream));
        CK(cudaStreamSynchronize(ctx->stream));
        float ms = 0.f; CK(cudaEventElapsedTime(&ms, ev0, ev1)); kernel_ms_total += ms;
        const T* ho = (const T*)ctx->hout;
        for (int64_t j = 0; j < nb; ++j) out[c0 + j] = (double)ho[j];
        ++waves;
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    ctx->stats.kernel_ms = kernel_ms_total;
    ctx->stats.batch = Bb; ctx->stats.waves = waves; ctx->stats.rmax_class = rcl; ctx->stats.tpb = tpb;
    ctx->stats.bytes_per_wave = 2 * vb + mb; ctx->stats.l2_tile = tile; ctx->stats.table_bytes = ctx->table_bytes;
    ctx->stats.graph_captures = ctx->graph_captures;
    ctx->stats.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return 0;
}

extern "C" int cubcr_e5_solve(cubcr_gpu_ctx* ctx, int k, const uint64_t* masks, int mstride, int64_t B,
                              double* out, int dtype, int64_t max_batch, int use_graph, int tpb, int64_t l2_tile) {
    if (!ctx || !masks || !out) { g_err = "null pointer"; return 1; }
    if (k < 0) { g_err = "k must be >= 0"; return 2; }
    if (B <= 0) return 0;
    if (dtype == 0) return solve_t<double>(ctx, k, masks, mstride, B, out, max_batch, use_graph, tpb, l2_tile);
    return solve_t<float>(ctx, k, masks, mstride, B, out, max_batch, use_graph, tpb, l2_tile);
}

extern "C" int cubcr_e5_last_stats(cubcr_gpu_ctx* ctx, cubcr_gpu_stats* st) { if (!ctx || !st) return 1; *st = ctx->stats; return 0; }
extern "C" int cubcr_e5_class_info_get(cubcr_gpu_ctx* ctx, int max_n, cubcr_e5_class_info* out) {
    if (!ctx) return -1;
    const int n = (int)ctx->cinfo.size();
    for (int i = 0; i < n && i < max_n; ++i) out[i] = ctx->cinfo[i];
    return n;
}
extern "C" int cubcr_e5_mem_info(int device, size_t* f, size_t* t) { CK(cudaSetDevice(device)); CK(cudaMemGetInfo(f, t)); return 0; }
extern "C" int cubcr_e5_device_attr(int device, int* smem_optin, int* smem_per_sm, int* regs_per_sm,
                                    int* max_threads_per_sm, int* sm_count) {
    CK(cudaDeviceGetAttribute(smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CK(cudaDeviceGetAttribute(smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CK(cudaDeviceGetAttribute(regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CK(cudaDeviceGetAttribute(max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    CK(cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, device));
    return 0;
}
