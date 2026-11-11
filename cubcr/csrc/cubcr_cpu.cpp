#include "cubcr.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <immintrin.h>

static std::string g_err;


template <typename T> struct Simd;
template <> struct Simd<double> {
    static constexpr int L = 8; using V = __m512d; using M = __mmask8;
    static inline V zero() { return _mm512_setzero_pd(); }
    static inline V set1(double x) { return _mm512_set1_pd(x); }
    static inline V load(const double* p) { return _mm512_loadu_pd(p); }
    static inline void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static inline V fma(V a, V b, V c) { return _mm512_fmadd_pd(a, b, c); }
    static inline V mask_min(V src, M m, V a, V b) { return _mm512_mask_min_pd(src, m, a, b); }
    static inline V maskz_keep(M m, V a) { return _mm512_maskz_mov_pd(m, a); }
    static inline M all() { return (M)0xFF; }
};
template <> struct Simd<float> {
    static constexpr int L = 16; using V = __m512; using M = __mmask16;
    static inline V zero() { return _mm512_setzero_ps(); }
    static inline V set1(float x) { return _mm512_set1_ps(x); }
    static inline V load(const float* p) { return _mm512_loadu_ps(p); }
    static inline void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static inline V fma(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
    static inline V mask_min(V src, M m, V a, V b) { return _mm512_mask_min_ps(src, m, a, b); }
    static inline V maskz_keep(M m, V a) { return _mm512_maskz_mov_ps(m, a); }
    static inline M all() { return (M)0xFFFF; }
};
extern "C" const char* cubcr_cpu_last_error(void) { return g_err.c_str(); }
extern "C" int cubcr_cpu_max_threads(void) { return omp_get_max_threads(); }

#ifndef PF_DIST
#define PF_DIST 32
#endif


struct Plan {
    cubcr_model m;
    std::vector<int64_t> dig_ofs;
    std::vector<int64_t> tab_ofs;
    std::vector<uint16_t> rdig;
    std::vector<uint8_t> pe;
    std::vector<uint8_t> deff;
    int rmax = 1;
};

static Plan* build_plan(const cubcr_model* m) {
    const int D = CUBCR_DMAX;
    Plan* pl = new Plan();
    pl->m = *m;
    const int64_t S = m->n_states;
    pl->dig_ofs.assign(S + 1, 0); pl->tab_ofs.assign(S + 1, 0); pl->deff.assign(S, 0);
    for (int64_t s = 0; s < S; ++s) {
        int d = 0;
        for (int l = 0; l < D; ++l) if (m->state_agent[s * D + l] >= 0) d = l + 1;
        pl->deff[s] = (uint8_t)d;
        const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
        pl->rmax = std::max<int64_t>(pl->rmax, R);
        pl->dig_ofs[s + 1] = pl->dig_ofs[s] + R;
        pl->tab_ofs[s + 1] = pl->tab_ofs[s] + (int64_t(1) << d) * R;
    }
    pl->rdig.resize(pl->dig_ofs[S]); pl->pe.resize(pl->tab_ofs[S]);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t s = 0; s < S; ++s) {
        const int d = pl->deff[s];
        const int64_t R = m->state_row_start[s + 1] - m->state_row_start[s];
        int rad[3]; for (int l = 0; l < D; ++l) rad[l] = m->state_radix[s * D + l];
        const int st[3] = {rad[1] * rad[2], rad[2], 1};
        std::vector<int> key(R), order(R);
        for (int r = 0; r < R; ++r) {
            int rr = r, dg[3]; dg[2] = rr % rad[2]; rr /= rad[2]; dg[1] = rr % rad[1]; rr /= rad[1]; dg[0] = rr;
            pl->rdig[pl->dig_ofs[s] + r] = (uint16_t)(dg[0] | (dg[1] << 3) | (dg[2] << 6));
        }
        for (int p = 0; p < (1 << d); ++p) {
            uint8_t* e = pl->pe.data() + pl->tab_ofs[s] + (int64_t)p * R;
            for (int r = 0; r < R; ++r) {
                const int dgv = pl->rdig[pl->dig_ofs[s] + r];
                const int dg[3] = {dgv & 7, (dgv >> 3) & 7, (dgv >> 6) & 7};
                int f = 0;
                for (int l = 0; l < D; ++l) {
                    const bool pinned = (l >= d) || ((p >> l) & 1);
                    if (!pinned) f += dg[l] * st[l];
                }
                key[r] = f; order[r] = r;
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
            for (int i = 0; i < R; ++i) {
                const bool gend = (i + 1 == R || key[order[i + 1]] != key[order[i]]);
                e[i] = (uint8_t)(order[i] | (gend ? 0x80 : 0));
            }
        }
    }
    return pl;
}


static inline double row_weight(const double* W, int dgv, int p, int d) {
    const int A = CUBCR_AMAX;
    double w = 1.0;
    if (d > 0 && (p & 1)) w *= W[0 * A + (dgv & 7)];
    if (d > 1 && (p & 2)) w *= W[1 * A + ((dgv >> 3) & 7)];
    if (d > 2 && (p & 4)) w *= W[2 * A + ((dgv >> 6) & 7)];
    return w;
}


template <typename T, int RMAX>
static void solve_block_simd(const Plan* pl, int k, const uint64_t* masks, int mstride, int64_t B, int64_t c0, int nc,
                             T* __restrict__ V, T* __restrict__ Vn, double* out) {
    using S_ = Simd<T>; using Vec = typename S_::V; using Msk = typename S_::M;
    constexpr int L = S_::L;
    const cubcr_model* m = &pl->m;
    const int64_t S = m->n_states;
    const int D = CUBCR_DMAX;
    for (int64_t s = 0; s < S; ++s) {
        const T v0 = (m->state_kind[s] == 1) ? T(1) : T(0);
        for (int j = 0; j < L; ++j) V[s * L + j] = v0;
    }
    uint64_t mk[L];
    alignas(64) T Q[RMAX][L];
    const Vec vinf = S_::set1(std::numeric_limits<T>::infinity());
    const Vec vone = S_::set1(T(1)), vzero = S_::zero();
    for (int t = k - 1; t >= 0; --t) {
        for (int j = 0; j < L; ++j) mk[j] = masks[(mstride ? (int64_t)t * B : 0) + c0 + (j < nc ? j : nc - 1)];
        for (int64_t s = 0; s < S; ++s) {
            const uint8_t kind = m->state_kind[s];
            T* __restrict__ vn = Vn + s * L;
            if (kind == 1) { S_::store(vn, vone); continue; }
            if (kind == 2) { S_::store(vn, vzero); continue; }
            const int64_t r0 = m->state_row_start[s];
            const int R = (int)(m->state_row_start[s + 1] - r0);
            const int64_t pend = m->row_ptr[r0 + R];
            for (int r = 0; r < R; ++r) {
                Vec q = S_::zero();
                const int64_t p0 = m->row_ptr[r0 + r], p1 = m->row_ptr[r0 + r + 1];
                for (int64_t p = p0; p < p1; ++p) {
                    const int64_t pf = p + PF_DIST;
                    if (pf < pend) _mm_prefetch((const char*)(V + (int64_t)m->col[pf] * L), _MM_HINT_T0);
                    q = S_::fma(S_::set1((T)m->val[p]), S_::load(V + (int64_t)m->col[p] * L), q);
                }
                S_::store(Q[r], q);
            }
            const int d = pl->deff[s];
            Vec best = vinf;
            if (d == 0) {
                best = S_::load(Q[0]);
            } else {
                const uint8_t* tab = pl->pe.data() + pl->tab_ofs[s];
                const uint16_t* dig = pl->rdig.data() + pl->dig_ofs[s];
                const double* W = m->weights + (int64_t)s * D * CUBCR_AMAX;
                int pat[L]; unsigned present = 0;
                for (int j = 0; j < L; ++j) {
                    int p = 0;
                    for (int l = 0; l < d; ++l) p |= (int)((mk[j] >> m->state_agent[s * D + l]) & 1ull) << l;
                    pat[j] = p; present |= 1u << p;
                }
                for (int p = 0; p < (1 << d); ++p) {
                    if (!((present >> p) & 1u)) continue;
                    Msk lm = 0;
                    for (int j = 0; j < L; ++j) lm |= (Msk)((pat[j] == p) ? 1u : 0u) << j;
                    const uint8_t* e = tab + (int64_t)p * R;
                    Vec acc = S_::zero();
                    for (int i = 0; i < R; ++i) {
                        const int perm = e[i] & 0x7F; const bool gend = (e[i] & 0x80) != 0;
                        const T w = (T)row_weight(W, dig[perm], p, d);
                        acc = S_::fma(S_::set1(w), S_::load(Q[perm]), acc);
                        const Msk g = gend ? lm : (Msk)0;
                        best = S_::mask_min(best, g, best, acc);
                        acc = S_::maskz_keep(gend ? (Msk)0 : S_::all(), acc);
                    }
                }
            }
            S_::store(vn, best);
        }
        std::swap(V, Vn);
    }
    for (int j = 0; j < nc; ++j) out[c0 + j] = (double)V[m->init_state * L + j];
}

template <typename T, int CB, int RMAX>
static void solve_block(const Plan* pl, int k, const uint64_t* masks, int mstride, int64_t B, int64_t c0, int nc,
                        T* __restrict__ V, T* __restrict__ Vn, double* out) {
    const cubcr_model* m = &pl->m;
    const int64_t S = m->n_states;
    const int D = CUBCR_DMAX;
    for (int64_t s = 0; s < S; ++s) {
        const T v0 = (m->state_kind[s] == 1) ? T(1) : T(0);
        for (int j = 0; j < CB; ++j) V[s * CB + j] = v0;
    }
    uint64_t mk[CB];
    alignas(64) T Q[RMAX][CB];
    for (int t = k - 1; t >= 0; --t) {
        for (int j = 0; j < CB; ++j) mk[j] = masks[(mstride ? (int64_t)t * B : 0) + c0 + (j < nc ? j : nc - 1)];
        for (int64_t s = 0; s < S; ++s) {
            const uint8_t kind = m->state_kind[s];
            T* __restrict__ vn = Vn + s * CB;
            if (kind == 1) { for (int j = 0; j < CB; ++j) vn[j] = T(1); continue; }
            if (kind == 2) { for (int j = 0; j < CB; ++j) vn[j] = T(0); continue; }
            const int64_t r0 = m->state_row_start[s];
            const int R = (int)(m->state_row_start[s + 1] - r0);

            for (int r = 0; r < R; ++r) {
                T* __restrict__ q = Q[r];
                for (int j = 0; j < CB; ++j) q[j] = T(0);
                const int64_t p0 = m->row_ptr[r0 + r], p1 = m->row_ptr[r0 + r + 1];
                for (int64_t p = p0; p < p1; ++p) {
                    const T pv = (T)m->val[p];
                    const T* __restrict__ vc = V + (int64_t)m->col[p] * CB;
                    for (int j = 0; j < CB; ++j) q[j] += pv * vc[j];
                }
            }

            const int d = pl->deff[s];
            int pat[CB]; unsigned present = 0;
            for (int j = 0; j < CB; ++j) {
                int p = 0;
                for (int l = 0; l < d; ++l) p |= (int)((mk[j] >> m->state_agent[s * D + l]) & 1ull) << l;
                pat[j] = p; present |= 1u << p;
            }
            T best[CB];
            for (int j = 0; j < CB; ++j) best[j] = std::numeric_limits<T>::infinity();
            const uint8_t* tab = pl->pe.data() + pl->tab_ofs[s];
            const uint16_t* dig = pl->rdig.data() + pl->dig_ofs[s];
            const double* W = m->weights + (int64_t)s * D * CUBCR_AMAX;
            for (int p = 0; p < (1 << d); ++p) {
                if (!((present >> p) & 1u)) continue;
                const uint8_t* e = tab + (int64_t)p * R;
                T acc[CB];
                for (int j = 0; j < CB; ++j) acc[j] = T(0);
                for (int i = 0; i < R; ++i) {
                    const int perm = e[i] & 0x7F; const bool gend = (e[i] & 0x80) != 0;
                    const T w = (T)row_weight(W, dig[perm], p, d);
                    const T* __restrict__ q = Q[perm];
                    for (int j = 0; j < CB; ++j) acc[j] += w * q[j];
                    if (gend) {
                        for (int j = 0; j < CB; ++j) best[j] = (pat[j] == p) ? std::min(best[j], acc[j]) : best[j];
                        for (int j = 0; j < CB; ++j) acc[j] = T(0);
                    }
                }
            }
            for (int j = 0; j < CB; ++j) vn[j] = best[j];
        }
        std::swap(V, Vn);
    }
    for (int j = 0; j < nc; ++j) out[c0 + j] = (double)V[m->init_state * CB + j];
}

template <typename T, int CB>
static void dispatch_rmax(const Plan* pl, int k, const uint64_t* masks, int mstride, int64_t B, int64_t c0, int nc,
                          T* V, T* Vn, double* out) {
    if constexpr (CB == Simd<T>::L) {
        if (pl->rmax <= 4) solve_block_simd<T, 4>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
        else if (pl->rmax <= 8) solve_block_simd<T, 8>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
        else if (pl->rmax <= 16) solve_block_simd<T, 16>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
        else solve_block_simd<T, 64>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
        return;
    }
    if (pl->rmax <= 4) solve_block<T, CB, 4>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
    else if (pl->rmax <= 8) solve_block<T, CB, 8>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
    else if (pl->rmax <= 16) solve_block<T, CB, 16>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
    else solve_block<T, CB, 64>(pl, k, masks, mstride, B, c0, nc, V, Vn, out);
}

template <typename T>
static int solve_impl(const Plan* pl, int k, const uint64_t* masks, int mstride, int64_t B, double* out,
                      int n_threads, int block) {
    if (!pl || !masks || !out) { g_err = "null pointer"; return 1; }
    if (k < 0) { g_err = "k must be >= 0"; return 2; }
    if (block != 1 && block != 4 && block != 8 && block != 16) { g_err = "block must be 1,4,8,16"; return 3; }
    if (pl->rmax > 64) { g_err = "more than 64 rows per state unsupported"; return 4; }
    const int64_t S = pl->m.n_states;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    const int64_t nblocks = (B + block - 1) / block;
#pragma omp parallel num_threads(n_threads)
    {
        std::vector<T> V((size_t)S * block), Vn((size_t)S * block);
#pragma omp for schedule(dynamic, 1)
        for (int64_t b = 0; b < nblocks; ++b) {
            const int64_t c0 = b * block;
            const int nc = (int)std::min<int64_t>(block, B - c0);
            switch (block) {
                case 1: dispatch_rmax<T, 1>(pl, k, masks, mstride, B, c0, nc, V.data(), Vn.data(), out); break;
                case 4: dispatch_rmax<T, 4>(pl, k, masks, mstride, B, c0, nc, V.data(), Vn.data(), out); break;
                case 8: dispatch_rmax<T, 8>(pl, k, masks, mstride, B, c0, nc, V.data(), Vn.data(), out); break;
                default: dispatch_rmax<T, 16>(pl, k, masks, mstride, B, c0, nc, V.data(), Vn.data(), out); break;
            }
        }
    }
    return 0;
}


extern "C" void* cubcr_cpu_create(const cubcr_model* m) {
    if (!m) { g_err = "null model"; return nullptr; }
    return (void*)build_plan(m);
}
extern "C" void cubcr_cpu_destroy(void* ctx) { delete (Plan*)ctx; }
extern "C" int cubcr_cpu_solve_ctx(void* ctx, int k, const uint64_t* masks, int mstride, int64_t B, double* out,
                                   int dtype, int n_threads, int block) {
    if (dtype == 0) return solve_impl<double>((Plan*)ctx, k, masks, mstride, B, out, n_threads, block);
    return solve_impl<float>((Plan*)ctx, k, masks, mstride, B, out, n_threads, block);
}
extern "C" int64_t cubcr_cpu_plan_bytes(void* ctx) { const Plan* p = (Plan*)ctx; return (int64_t)(p->pe.size() + 2 * p->rdig.size()); }


extern "C" int cubcr_cpu_solve_f64(const cubcr_model* m, int k, const uint64_t* masks, int mstride,
                                   int64_t B, double* out, int n_threads, int block) {
    Plan* pl = build_plan(m); int r = solve_impl<double>(pl, k, masks, mstride, B, out, n_threads, block); delete pl; return r;
}
extern "C" int cubcr_cpu_solve_f32(const cubcr_model* m, int k, const uint64_t* masks, int mstride,
                                   int64_t B, double* out, int n_threads, int block) {
    Plan* pl = build_plan(m); int r = solve_impl<float>(pl, k, masks, mstride, B, out, n_threads, block); delete pl; return r;
}
