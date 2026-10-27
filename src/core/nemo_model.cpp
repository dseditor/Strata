// src/core/nemo_model.cpp - see include/strata/core/nemo_model.hpp.
#include "strata/core/nemo_model.hpp"

#include "strata/kernels/mamba.hpp"
#include "strata/kernels/nemo_attn.hpp"
#include "strata/kernels/nemo_gemv.hpp"
#include "strata/kernels/nemo_fp4.hpp"
#include "strata/kernels/nemo_q8.hpp"
#include "strata/kernels/nemo_moe.hpp"
#include "strata/kernels/nemo_ops.hpp"
#include "strata/kernels/nemo_tc.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <map>
#include <thread>
#ifdef _WIN32
// windows headers (via cuda_runtime.h) define IN / OUT as empty macros; this file uses IN as a local name
#undef IN
#undef OUT
#endif

namespace strata::core {

using strata::kernels::NemoWType;

namespace {

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {   // subnormal
            float v = (float) m / 1024.0f / 16384.0f;
            return s ? -v : v;
        }
    } else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112u) << 23) | (m << 13);
    float v;
    std::memcpy(&v, &f, 4);
    return v;
}
uint16_t float_to_bf16(float v) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

#define NCK(x)                                                                                    \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            err = std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x;                     \
            return false;                                                                         \
        }                                                                                         \
    } while (0)

/// A matrix on the device in its GGUF format.
struct Mat {
    NemoWType t = NemoWType::BF16;
    void* d = nullptr;
    float s2 = 1.0f;
    int64_t rows = 0, cols = 0;
    bool tiled = false;   ///< in nemo_gemv's tiled layout (the tensor-core kernels'), not the GGUF rows
};

/// An attention block's K/V pages (strata::kernels::NemoKV: int8 K, its scales, FP16 V).
struct KvPages {
    std::vector<int8_t*> k;
    std::vector<float*> ks;
    std::vector<uint16_t*> v;
};

struct Block {
    NemotronBlock kind = NemotronBlock::Mamba;
    float* norm = nullptr;
    // mamba
    Mat in, out;
    float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    float *state = nullptr, *conv = nullptr;
    // attention
    Mat q, k, v, o;
    KvPages kv;   ///< its K/V pages (Impl::kv_shift cells each)
    // moe
    Mat gate, fc1, fc2, sh_up, sh_dn;
    float* bias = nullptr;
    int moe = -1;   ///< index among the MoE blocks
    int mamba = -1; ///< index among the Mamba blocks
};

/// One MoE layer's experts: their bytes in the (page-locked) mapping, and the layer's VRAM slots.
struct ExpertLayer {
    NemoWType tu = NemoWType::NVFP4, td = NemoWType::NVFP4;   ///< the up / down formats (per layer in a mixed file)
    uint64_t ub = 0, db = 0;       ///< bytes of one expert's up / down matrix; a slot holds up, then down
    uint64_t stride = 0;           ///< a slot's bytes (ub + db, 16-byte aligned)
    const uint8_t* up = nullptr;   ///< host: expert e's up rows at up + e * ub
    const uint8_t* dn = nullptr;
    std::vector<float> s_up, s_dn; ///< per expert scale_2
    std::vector<uint8_t*> bands;   ///< device: slot s in band s / Impl::BS (Impl::slot_ptr), BS slots a band
    std::vector<uint8_t*> loan;    ///< bands lent by the chunk scratch (Impl::lend), after the own ones
    std::vector<int32_t> slot_of;  ///< expert -> slot or -1
    std::vector<int32_t> expert_in; ///< slot -> expert or -1
    std::vector<uint64_t> used;    ///< slot -> last use (LRU clock)
    std::vector<float> score;      ///< expert -> decayed use count (STRATA_NEMO_EVICT=lfu), as of score_at
    std::vector<uint64_t> score_at;
    std::vector<float> dsc;        ///< STRATA_NEMO_LFU_FAST: every slot's decayed score as of dsc_ticks (-1: empty slot)
    uint64_t dsc_ticks = UINT64_MAX;
    uint64_t ticks = 0;            ///< tokens this layer has routed (decode / verify rows)
};

}  // namespace

/// A prompt chunk's routing by expert: expert e's entries (t * K + j, increasing) at rows[off[e] .. off[e] + cnt[e]),
/// built by one counting pass (cnt[e] = 0 drops the expert).
struct ExpertRows {
    struct Span {
        const int32_t* b;
        const int32_t* e;
        const int32_t* begin() const { return b; }
        const int32_t* end() const { return e; }
        size_t size() const { return (size_t) (e - b); }
        bool empty() const { return b == e; }
    };
    std::vector<int32_t> off, cnt, rows;
    void build(const int32_t* ids, int64_t n, int64_t ne) {
        cnt.assign((size_t) ne, 0);
        off.resize((size_t) ne);
        rows.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) ++cnt[(size_t) ids[i]];
        int32_t at = 0;
        for (int64_t e = 0; e < ne; ++e) {
            off[(size_t) e] = at;
            at += cnt[(size_t) e];
        }
        std::vector<int32_t> fill(off);
        for (int64_t i = 0; i < n; ++i) rows[(size_t) fill[(size_t) ids[i]]++] = (int32_t) i;
    }
    Span operator[](size_t e) const { return Span{rows.data() + off[e], rows.data() + off[e] + cnt[e]}; }
    void clear(size_t e) { cnt[e] = 0; }
};

struct NemoModel::Impl {
    std::unique_ptr<strata::GgufModel> gguf;
    std::unique_ptr<strata::GgufModel> xgguf;   ///< --experts-gguf: the routed experts' tensors from another file
    std::unique_ptr<strata::GgufModel> mgguf;   ///< --mtp-gguf
    std::vector<Block> blocks;
    std::vector<ExpertLayer> ex;
    uint64_t blob = 0;   ///< a slot: the largest layer's up + down
    int64_t slots = 0;
    uint64_t clock = 0;
    // ---- the K/V cache in pages, and the expert cache in bands: the two share what the card has left ----
    // A sequence starts with kv_base_pages pages of 2^kv_shift cells (every attention block and the MTP); past them a
    // page is added as the context reaches it, and the VRAM for it comes from the expert cache: whole bands of BS
    // slots of every MoE layer are dropped (their experts evicted) until the page fits.  reset() gives it back.
    static constexpr int64_t BS = 4;
    int kv_shift = 16;
    int64_t kv_pages = 0, kv_base_pages = 0, slots_max = 0, kd = 0;
    int64_t reserve_b = 0;
    uint8_t* slot_ptr(const ExpertLayer& X, int64_t s) const {
        if (s >= real_slots) {
            s -= real_slots;
            return X.loan[(size_t) (s / BS)] + (size_t) (s % BS) * X.stride;
        }
        return X.bands[(size_t) (s / BS)] + (size_t) (s % BS) * X.stride;
    }
    // ---- slots lent by the chunk scratch ----
    // The scratch holds C rows, and a chunk of at most LC rows uses the first LC of them: the rest of the big buffers
    // (~1 GB at C = 16,384) holds loan bands of expert slots meanwhile (slots real_slots .. slots).  A longer chunk
    // takes them back first (reclaim: their experts leave the cache); lend() gives them again after it.
    static constexpr int64_t LC = 8192;
    int64_t real_slots = 0;
    bool lent = false;
    std::vector<std::pair<uint8_t*, uint64_t>> loan_free;   ///< the scratch's spare tails (computed at load)
    void reclaim() {
        if (!lent) return;
        cudaStreamSynchronize(st);
        cudaStreamSynchronize(cst);
        for (size_t l = 0; l < ex.size(); ++l) {
            ExpertLayer& X = ex[l];
            for (int64_t s = real_slots; s < slots; ++s)
                if (X.expert_in[(size_t) s] >= 0) X.slot_of[(size_t) X.expert_in[(size_t) s]] = -1;
            X.expert_in.resize((size_t) real_slots);
            X.used.resize((size_t) real_slots);
            pf[l].resize((size_t) real_slots);
            fill_tok[l].resize((size_t) real_slots);
        }
        slots = real_slots;
        lent = false;
    }
    void lend() {
        if (lent || ex.empty() || ex[0].loan.empty()) return;
        const int64_t n = (int64_t) ex[0].loan.size() * BS;
        for (size_t l = 0; l < ex.size(); ++l) {
            ExpertLayer& X = ex[l];
            X.expert_in.resize((size_t) (real_slots + n), -1);
            X.used.resize((size_t) (real_slots + n), 0);
            pf[l].resize((size_t) (real_slots + n), 0);
            fill_tok[l].resize((size_t) (real_slots + n), UINT64_MAX);
        }
        slots = real_slots + n;
        lent = true;
    }
    /// Carve the loan bands out of loan_free: as many whole sets (a band of every layer) as fit.
    void carve_loans() {
        std::vector<std::pair<uint8_t*, uint64_t>> fr = loan_free;
        for (;;) {
            std::vector<std::pair<size_t, uint8_t*>> got;   // (region, piece) per layer
            std::vector<std::pair<uint8_t*, uint64_t>> tmp = fr;
            for (const ExpertLayer& X : ex) {
                const uint64_t need = (uint64_t) BS * X.stride;
                bool ok = false;
                for (size_t r = 0; r < tmp.size(); ++r) {
                    uint8_t* a = (uint8_t*) (((uintptr_t) tmp[r].first + 255) & ~(uintptr_t) 255);
                    const uint64_t lost = (uint64_t) (a - tmp[r].first);
                    if (tmp[r].second >= lost + need) {
                        got.emplace_back(r, a);
                        tmp[r].first = a + need;
                        tmp[r].second -= lost + need;
                        ok = true;
                        break;
                    }
                }
                if (!ok) return;
            }
            for (size_t l = 0; l < ex.size(); ++l) ex[l].loan.push_back(got[l].second);
            fr = tmp;
        }
    }
    strata::kernels::NemoKV kv_of(const KvPages& p) const {
        strata::kernels::NemoKV kv;
        kv.shift = kv_shift;
        for (size_t i = 0; i < p.k.size(); ++i) {
            kv.k[i] = p.k[i];
            kv.ks[i] = p.ks[i];
            kv.v[i] = p.v[i];
        }
        return kv;
    }
    std::vector<KvPages*> kv_lists;   ///< every K/V owner
    int64_t n_kv_heads = 0;
    /// a page of every owner: per cell and KV head, head_dim int8 + a float + head_dim FP16
    uint64_t kv_page_bytes() const {
        return (uint64_t) kv_lists.size() * ((uint64_t) n_kv_heads << kv_shift) * (3 * (uint64_t) (kd / n_kv_heads) + 4);
    }
    bool kv_add_page() {
        const size_t cells = (size_t) 1 << kv_shift;
        for (KvPages* l : kv_lists) {
            int8_t* k = nullptr;
            float* ks = nullptr;
            uint16_t* v = nullptr;
            if (cudaMalloc(&k, cells * kd) != cudaSuccess || cudaMalloc(&ks, cells * n_kv_heads * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&v, cells * kd * sizeof(uint16_t)) != cudaSuccess) {
                cudaGetLastError();
                if (k) cudaFree(k);
                if (ks) cudaFree(ks);
                return false;
            }
            l->k.push_back(k);
            l->ks.push_back(ks);
            l->v.push_back(v);
        }
        ++kv_pages;
        return true;
    }
    void kv_drop_page() {
        for (KvPages* l : kv_lists) {
            cudaFree(l->k.back());
            cudaFree(l->ks.back());
            cudaFree(l->v.back());
            l->k.pop_back();
            l->ks.pop_back();
            l->v.pop_back();
        }
        --kv_pages;
    }
    /// Drop the last band of every MoE layer (its experts leave the cache).
    void band_drop() {   // (the loans reclaimed)
        const int64_t s0 = (int64_t) (ex[0].bands.size() - 1) * BS;
        for (size_t l = 0; l < ex.size(); ++l) {
            ExpertLayer& X = ex[l];
            for (int64_t s = s0; s < real_slots; ++s)
                if (X.expert_in[(size_t) s] >= 0) X.slot_of[(size_t) X.expert_in[(size_t) s]] = -1;
            cudaFree(X.bands.back());
            X.bands.pop_back();
            X.expert_in.resize((size_t) s0);
            X.used.resize((size_t) s0);
            pf[l].resize((size_t) s0);
            fill_tok[l].resize((size_t) s0);
        }
        slots = real_slots = s0;
    }
    /// A band more for every MoE layer (empty slots), if the card has room for it and the reserve.
    bool band_add() {   // (the loans reclaimed)
        if (real_slots + BS > slots_max) return false;
        uint64_t need = 0;
        for (const ExpertLayer& X : ex) need += BS * X.stride;
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if ((int64_t) fr < (int64_t) need + reserve_b) return false;
        std::vector<uint8_t*> got;
        for (ExpertLayer& X : ex) {
            uint8_t* p = nullptr;
            if (cudaMalloc(&p, (size_t) BS * X.stride) != cudaSuccess) {
                cudaGetLastError();
                for (size_t i = 0; i < got.size(); ++i) cudaFree(got[i]);
                return false;
            }
            got.push_back(p);
        }
        for (size_t l = 0; l < ex.size(); ++l) {
            ExpertLayer& X = ex[l];
            X.bands.push_back(got[l]);
            X.expert_in.resize((size_t) (real_slots + BS), -1);
            X.used.resize((size_t) (real_slots + BS), 0);
            pf[l].resize((size_t) (real_slots + BS), 0);
            fill_tok[l].resize((size_t) (real_slots + BS), UINT64_MAX);
        }
        slots = real_slots = real_slots + BS;
        return true;
    }
    /// Pages for `cells` cells, the expert cache giving up bands for them.
    bool kv_grow(int64_t cells, std::string& err) {
        const int64_t want = (cells + (1ll << kv_shift) - 1) >> kv_shift;
        if (want <= kv_pages) return true;
        if (want > strata::kernels::NEMO_KV_MAX_PAGES) {
            err = "the K/V cache would need more than " + std::to_string(strata::kernels::NEMO_KV_MAX_PAGES) + " pages";
            return false;
        }
        const bool was_lent = lent;
        reclaim();
        cudaStreamSynchronize(st);
        cudaStreamSynchronize(cst);
        while (kv_pages < want) {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            while ((int64_t) fr < (int64_t) kv_page_bytes() + reserve_b) {
                if (real_slots - BS < 2 * 22 + BS) {   // keep a working cache (a verify window's experts and more)
                    err = "no VRAM left for the K/V cache of " + std::to_string(cells) + " cells";
                    return false;
                }
                band_drop();
                cudaMemGetInfo(&fr, &tot);
            }
            if (!kv_add_page()) {
                err = "cudaMalloc of a K/V page failed";
                return false;
            }
        }
        if (was_lent) lend();
        return true;
    }
    /// Back to the base pages, the bands they had returned to the expert cache.
    void kv_shrink() {
        if (kv_pages <= kv_base_pages) return;
        cudaStreamSynchronize(st);
        cudaStreamSynchronize(cst);
        const bool was_lent = lent;
        reclaim();
        while (kv_pages > kv_base_pages) kv_drop_page();
        while (band_add()) {
        }
        if (was_lent) lend();
    }
    std::vector<void*> allocs;
    const uint16_t* embd = nullptr;   ///< host, in the mapping: (n_vocab, n_embd) BF16
    float* out_norm = nullptr;
    Mat head;
    cudaStream_t st = nullptr;
    int64_t C = 0;                    ///< rows the scratch holds (the chunk)
    int64_t SC = 0;                   ///< rows of the Mamba and shared-expert sub-chunks

    // scratch
    float *x = nullptr, *xn = nullptr, *zx = nullptr, *xbc = nullptr, *y = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    void* attn_scratch = nullptr;
    float *logits = nullptr, *w = nullptr, *lat = nullptr, *parts = nullptr, *routed = nullptr;
    float *g_in = nullptr, *g_h = nullptr, *g_out = nullptr, *sh_h = nullptr, *eh = nullptr;
    int32_t *ids = nullptr, *idx = nullptr;
    const uint8_t** eptr = nullptr;
    float* es2 = nullptr;
    float* head_out = nullptr;
    uint16_t* embd_stage = nullptr;   ///< pinned host staging for the token rows
    uint16_t* embd_pinned = nullptr;  ///< the embedding table, page-locked (else rows come from the mapping)
    std::vector<uint16_t*> host_pinned;
    // images (set_image_rows): runs of (absolute position, count, first row), their rows page-locked
    struct ImgRun {
        int64_t pos, n, row;
    };
    std::vector<ImgRun> img_runs;
    float* img_host = nullptr;
    size_t img_cap = 0;   ///< floats
    bool img_pinned = false;   ///< false: a pageable fallback (page-locked RAM ran out, e.g. the Windows pinned limit)
    /// The image rows among T rows that start at absolute position p0, copied over `x` (device, E floats a row).
    void img_overlay(float* x, int64_t p0, int64_t T, int64_t E, cudaStream_t s) {
        for (const ImgRun& r : img_runs) {
            const int64_t a = std::max(r.pos, p0), b = std::min(r.pos + r.n, p0 + T);
            if (a < b)
                cudaMemcpyAsync(x + (a - p0) * E, img_host + (size_t) (r.row + a - r.pos) * E,
                                (size_t) (b - a) * E * sizeof(float), cudaMemcpyHostToDevice, s);
        }
    }
    // snapshot
    std::vector<float*> snap_state, snap_conv;
    bool pinned = false;
    uint8_t* arena = nullptr;      ///< page-locked host copy of every expert
    float* all_logits = nullptr;   ///< forward_all's host output, else null
    // the prompt's tensor-core path (nemo_tc): a BF16 copy of the dense matrix in use, the chunk's BF16 activations,
    // and for the experts a group's BF16 weights and its token rows (at most TC_ROWS rows, TC_GROUP experts a call)
    static constexpr int64_t TC_MIN = 16, TC_GROUP = 32, TC_ROWS = 8192;
    int64_t tc_rows = TC_ROWS;   ///< a group's rows: at least the chunk (one expert can take every row of it)
    /// wb's two N4 weights (the W4A4 path converts a weight once for all the sub-chunks of a chunk: the Mamba and
    /// shared-expert loops alternate two matrices); the BF16 path's use of wb forgets them
    const void* n4_of[2] = {nullptr, nullptr};
    int n4_next = 0;
    uint64_t wb_elems = 0;
    uint16_t *wb = nullptr, *xb = nullptr, *eg_up = nullptr, *eg_dn = nullptr, *eg_xb = nullptr, *eg_hb = nullptr;
    float *eg_h = nullptr, *eg_out = nullptr, *eg_in = nullptr;
    int32_t* eg_idx = nullptr;
    const uint8_t** eg_ptr = nullptr;
    double* routed_d = nullptr;           ///< the prompt MoE's routed sum, in double (nemo_accum_rows_f64)
    uint16_t* parts_bf = nullptr;         ///< the FP4 path: every (t, j) expert output, BF16 (C K rows of L)
    int32_t* h_eg_idx = nullptr;          ///< pinned: a prompt MoE pass's group lists, uploaded in one copy
    const uint8_t** h_eg_ptr = nullptr;
    size_t eg_idx_cap = 0, eg_ptr_cap = 0;
    // ---- the prompt's expert stream ----
    // A chunk of the prompt needs nearly every expert of every MoE layer, and those not in a layer's slots cross PCIe
    // (~25 ms a layer at 8K rows).  The copies go on `cst` in one sequence (`pq`, the misses layer after layer) into
    // a ring of `ring_n` blobs of their own (or an empty cache slot, which then keeps the expert), so they run under
    // the compute: a layer's hits are computed first, its misses group by group as their copies land, and once a
    // layer is planned the next one's non-resident experts are queued blindly (a chunk of >= blind_min rows uses
    // almost all of them).  A ring blob is free again once the group that read it has expanded it to BF16 (an event
    // on `st`); the copies wait for that, the compute waits for its group's copies (`pcopy`).
    struct PItem {
        int32_t layer = 0, e = 0, ring = -1;   ///< ring = -1: an empty cache slot of the layer
        uint8_t* dst = nullptr;
        bool skip = false;                     ///< no row needs it: not copied
    };
    std::vector<PItem> pq;
    size_t pq_copied = 0;                      ///< pq[0, pq_copied) have their copies enqueued (or are skipped)
    std::vector<size_t> pq_first, pq_end;      ///< per MoE layer: its items, pq_first = SIZE_MAX when not planned
    std::vector<std::vector<int32_t>> pitem;   ///< per MoE layer, per expert: its item, or -1
    uint8_t* ring = nullptr;
    int64_t ring_n = std::getenv("STRATA_NEMO_RING") ? std::atoll(std::getenv("STRATA_NEMO_RING")) : 384;
    int64_t ring_next = 0, blind_min = 512;
    std::vector<int32_t> ring_rel;             ///< ring blob -> the event (pev) after its last reader, or -1
    std::vector<uint8_t> ring_busy;            ///< ring blob holds an item no group has read yet
    std::vector<cudaEvent_t> pev;
    size_t pev_next = 0;
    cudaEvent_t pcopy = nullptr;
    uint64_t pq_copies = 0;

    /// Plan MoE layer lm's copies: its non-resident experts (only those with rows, when `need` is given).
    std::vector<int32_t> tok_cnt, tok_pos;   ///< the prompt MoE's per-token scratch (counting sorts), zero between groups
    // a short prompt chunk with the skip (pskip): the next MoE layer's router on the residual as it is, so its
    // likely misses (predicted weight >= pskip_w) are queued while this layer computes
    float *pp_logits = nullptr, *pp_w = nullptr;
    int32_t* pp_ids = nullptr;
    std::vector<int32_t> h_pp_ids;
    std::vector<float> h_pp_w;
    int64_t gg_max = std::getenv("STRATA_NEMO_GG_MAX") ? std::atoll(std::getenv("STRATA_NEMO_GG_MAX")) : INT64_MAX;
    int64_t mtp_window = std::getenv("STRATA_NEMO_MTP_WINDOW") ? std::atoll(std::getenv("STRATA_NEMO_MTP_WINDOW")) : 0;
    bool pskip_pred = !std::getenv("STRATA_NEMO_PROMPT_PRED") || std::atoi(std::getenv("STRATA_NEMO_PROMPT_PRED")) != 0;
    void pq_gen(int lm, const ExpertRows* need, const std::vector<uint8_t>* want = nullptr) {
        ExpertLayer& X = ex[(size_t) lm];
        if (pq_first[(size_t) lm] == SIZE_MAX) pq_first[(size_t) lm] = pq.size();   // (else: extending, the layer's
        int32_t empty = 0;                                                           //  items are the tail of pq)
        for (int32_t e = 0; e < (int32_t) X.slot_of.size(); ++e) {
            if (X.slot_of[(size_t) e] >= 0 || pitem[(size_t) lm][(size_t) e] >= 0 ||
                (need && (*need)[(size_t) e].empty()) || (want && !(*want)[(size_t) e]))
                continue;
            PItem it;
            it.layer = lm;
            it.e = e;
            while (empty < (int32_t) slots && X.expert_in[(size_t) empty] >= 0) ++empty;
            if (empty < (int32_t) slots) {
                X.expert_in[(size_t) empty] = e;
                X.slot_of[(size_t) e] = empty;
                X.used[(size_t) empty] = ++clock;
                it.dst = slot_ptr(X, empty);
            } else {
                it.ring = (int32_t) (ring_next++ % ring_n);
                it.dst = ring + (size_t) it.ring * blob;
            }
            pitem[(size_t) lm][(size_t) e] = (int32_t) pq.size();
            pq.push_back(it);
        }
        pq_end[(size_t) lm] = pq.size();
    }
    /// Enqueue the planned copies before item `lim` (as far as the ring allows).
    void pq_copy_until(size_t lim) {
        while (pq_copied < pq.size() && pq_copied < lim) {
            PItem& it = pq[pq_copied];
            if (!it.skip) {
                if (it.ring >= 0) {
                    if (ring_busy[(size_t) it.ring]) break;
                    if (ring_rel[(size_t) it.ring] >= 0)
                        cudaStreamWaitEvent(cst, pev[(size_t) ring_rel[(size_t) it.ring]], 0);
                    ring_busy[(size_t) it.ring] = 1;
                }
                const ExpertLayer& X = ex[(size_t) it.layer];
                cudaMemcpyAsync(it.dst, X.up + (size_t) it.e * X.ub, X.ub, cudaMemcpyHostToDevice, cst);
                cudaMemcpyAsync(it.dst + X.ub, X.dn + (size_t) it.e * X.db, X.db, cudaMemcpyHostToDevice, cst);
                ++pq_copies;
            }
            ++pq_copied;
        }
    }
    /// An item no row needs: not copied, or (copied already) its blob is free at once - the next copy into it is
    /// behind this one on `cst`.
    void pq_drop(size_t k) {
        PItem& it = pq[k];
        if (k >= pq_copied) {
            it.skip = true;
            if (it.ring < 0) {   // its empty cache slot stays empty
                ExpertLayer& X = ex[(size_t) it.layer];
                const int32_t c = X.slot_of[(size_t) it.e];
                X.slot_of[(size_t) it.e] = -1;
                X.expert_in[(size_t) c] = -1;
            }
        } else if (it.ring >= 0) {
            ring_busy[(size_t) it.ring] = 0;
        }
    }
    // STRATA_NEMO_PROFILE=1: the stream is synchronized at every phase boundary and the wall time between them is
    // added to the phase (so the profile slows the run down; it says where the time goes, not how fast it is)
    bool prof = std::getenv("STRATA_NEMO_PROFILE") != nullptr;
    uint8_t* zero_blob = nullptr;
    // decode's expert copies go on their own stream `cst` (the compute waits on `cev`, recorded after a layer's
    // copies), so the next MoE layer's predicted misses (STRATA_NEMO_PREFETCH: its router on the current residual)
    // stream in while this layer and the Mamba / attention between compute
    cudaStream_t cst = nullptr;
    cudaEvent_t cev = nullptr;
    cudaEvent_t ovf_ev[2] = {nullptr, nullptr};   ///< decode's ring overflow halves: free after this event
    bool ovf_rec[2] = {false, false};
    bool prefetch = !std::getenv("STRATA_NEMO_PREFETCH") || std::atoi(std::getenv("STRATA_NEMO_PREFETCH")) != 0;
    bool evict_lfu = std::getenv("STRATA_NEMO_EVICT") && !std::strcmp(std::getenv("STRATA_NEMO_EVICT"), "lfu");
    double lfu_decay = std::getenv("STRATA_NEMO_LFU_DECAY") ? std::atof(std::getenv("STRATA_NEMO_LFU_DECAY")) : 0.97;
    int64_t prefetch_rank = std::getenv("STRATA_NEMO_PREFETCH_RANK") ? std::atoll(std::getenv("STRATA_NEMO_PREFETCH_RANK")) : 0;
    int64_t prefetch_rows = std::getenv("STRATA_NEMO_PREFETCH_ROWS") ? std::atoll(std::getenv("STRATA_NEMO_PREFETCH_ROWS")) : 1;
    std::vector<int> moe_block;           ///< MoE index -> block index
    float *pxn = nullptr, *plogits = nullptr, *pw = nullptr;
    int32_t* pids = nullptr;
    float* h_dw = nullptr;                ///< pinned: their routing weights (STRATA_NEMO_SKIP_W)
    float skip_w = std::getenv("STRATA_NEMO_SKIP_W") ? (float) std::atof(std::getenv("STRATA_NEMO_SKIP_W")) : 0.0f;
    /// STRATA_NEMO_PROMPT_SKIP_W=tau (with STRATA_NEMO_PROMPT_SKIP_MAX rows, default 4096): in a prompt chunk of at
    /// most that many rows a miss whose largest routing weight over the chunk is < tau is not copied (its rows get
    /// nothing from it) - a short prompt needs almost every expert, and the copies, not the compute, set its time
    float pskip_w = std::getenv("STRATA_NEMO_PROMPT_SKIP_W") ? (float) std::atof(std::getenv("STRATA_NEMO_PROMPT_SKIP_W")) : 0.0f;
    int64_t pskip_max = std::getenv("STRATA_NEMO_PROMPT_SKIP_MAX") ? std::atoll(std::getenv("STRATA_NEMO_PROMPT_SKIP_MAX")) : 4096;
    int32_t *h_pids = nullptr, *h_dids = nullptr;   ///< pinned: the predicted / the actual ids of a decode step
    /// a decode step's expert list, uploaded in one copy: 2K pointers (up, then down), 2K scale_2, K scratch rows -
    /// entries [0, n_ready) are in VRAM already, the rest wait for the layer's copies.  Pinned, one per MoE layer.
    uint8_t* hp_meta = nullptr;
    uint8_t* d_meta = nullptr;
    size_t meta_bytes = 0;
    uint64_t tok = 0;                     ///< decode steps so far
    bool verify = false;                  ///< forward() is a verify window: the decode path for T <= 8 rows
    std::vector<int32_t> dec_best, dec_where;
    std::vector<float> dec_bw;   ///< per expert, scratch of the decode path (-1 between passes)
    std::vector<std::vector<uint64_t>> fill_tok;   ///< per MoE layer, per slot: the step whose copy filled it
    std::vector<std::vector<uint8_t>> pf;
    // ---- the MTP block (blk.<n_layers>: the nextn head), see mtp_rows ----
    struct Mtp {
        bool on = false;
        Mat eh, q, k, v, o, gate, fc1, fc2, sh_up, sh_dn;
        float *enorm = nullptr, *hnorm = nullptr, *attn_norm = nullptr, *post_norm = nullptr, *head_norm = nullptr;
        float* bias = nullptr;
        uint8_t* experts = nullptr;   ///< its experts' slots: all 512 resident (e at experts + e * stride), or a cache
        int64_t nslots = 0;
        uint8_t* host = nullptr;      ///< the cache's page-locked source (e at host + e * stride)
        std::vector<int32_t> slot_of, expert_in;
        std::vector<uint64_t> used;
        uint64_t clock = 0, hits = 0, misses = 0;
        uint8_t* h_meta = nullptr;    ///< pinned: a draft step's pointers and scales
        cudaEvent_t fill_ev = nullptr;
        uint64_t ub = 0, stride = 0;
        float *s_up = nullptr, *s_dn = nullptr;   ///< device, per expert scale_2
        std::vector<float> su_h, sd_h;
        KvPages kv;                               ///< its own K/V pages (NoPE attention over every position)
        int64_t MC = 0;                           ///< rows a call takes (the prompt goes in pieces of MC)
        float *e = nullptr, *hn = nullptr, *cat = nullptr, *u = nullptr, *a = nullptr;
        float *h = nullptr;       ///< pending: the main model's last normed row, whose MTP entry is not written yet
        float* hnext = nullptr;   ///< the last call's h_nextn (LayerNorm output): the next draft step's h
        float* logits = nullptr;
        int32_t* d_tok = nullptr;
        int32_t* h_tok = nullptr;   ///< pinned
        uint16_t* stage = nullptr;  ///< pinned, MC rows
        NemoWType tu = NemoWType::NVFP4, td = NemoWType::NVFP4;   ///< the experts' types (--mtp-gguf: any)
        const uint8_t** ptr = nullptr;
        float* s2 = nullptr;
        bool pending = false;
        int64_t pending_pos = -1;
        int32_t consumed_tok = -1;   ///< mtp_draft wrote the pending entry with this next token already
        int drafted = 0;             ///< drafts the last mtp_draft made (it may stop early)
        // snapshot of the pending state
        float* snap_h = nullptr;
        bool snap_pending = false;
        int64_t snap_pending_pos = -1;
    } mtp;
    // ---- speculation (spec_step): the window's Mamba inputs for the replay, the state before the window ----
    std::vector<float*> spec_state, spec_conv;
    float* spec_zx = nullptr;      ///< per Mamba block, the window's in_proj rows (MM rows of ssm_in_dim)
    float* vlogits = nullptr;      ///< a verify window's logits (MM rows)
    int32_t* vtok = nullptr;       ///< their argmax
    int32_t* h_vtok = nullptr;     ///< pinned
    std::vector<float> pick_row;   ///< a sampled step's row on the host
    // ---- batch 1 switches (all default off) ----
    static bool env_on(const char* n) { const char* v = std::getenv(n); return v && std::atoi(v) != 0; }
    const bool shared_early = env_on("STRATA_NEMO_SHARED_EARLY");   ///< T1: decode's shared expert before the ids sync
    const bool lfu_fast = env_on("STRATA_NEMO_LFU_FAST");           ///< T2: decayed LFU scores once per pass
    const bool lfu_check = env_on("STRATA_NEMO_LFU_CHECK");         ///< T2: compare it with the per-slot exp
    const bool mamba_swap = env_on("STRATA_NEMO_MAMBA_SWAP");       ///< T3: out-of-place verify scan + pointer swap
    const bool mtp_event = env_on("STRATA_NEMO_MTP_EVENT");         ///< T5: event instead of a drain in mtp_rows
    uint64_t lfu_mismatch = 0, lfu_checked = 0;
    bool verify_swap = false;      ///< this verify window writes the Mamba state into spec_state / spec_conv
    cudaEvent_t ids_ev = nullptr;  ///< T1: after the ids read-back
    cudaEvent_t stage_ev = nullptr;   ///< T5: after the MTP stage's H2D
    float* d_tk_v = nullptr;       ///< T4: top-k candidates (MM rows x 1024), device and pinned
    int32_t* d_tk_i = nullptr;
    float* h_tk_v = nullptr;
    int32_t* h_tk_i = nullptr;
    bool gpu_topk_on() { return env_on("STRATA_NEMO_GPU_TOPK") && d_tk_v && h_tk_v; }
    double spec_ms[4] = {0};
    float spec_min_p = std::getenv("STRATA_NEMO_SPEC_MIN_P") ? (float) std::atof(std::getenv("STRATA_NEMO_SPEC_MIN_P")) : 0.0f;       ///< drafting, verifying, the rest (pick, rollback, MTP entries); windows
    std::chrono::steady_clock::time_point ht[3];
    double hms[4] = {0};
    bool hostprof = std::getenv("STRATA_NEMO_HOSTPROF") != nullptr;
    double plan_ms = 0;   ///< STRATA_NEMO_HOSTPROF: the prompt MoE's host planning (ids back to the lists' upload)
    uint64_t plan_n = 0; ///< per MoE layer, per slot: filled by a prefetch and not used yet
    // STRATA_NEMO_TAIL_SKIP=R: in decode, an expert of rank >= R (0-based, by routing score) that is not in VRAM is
    // skipped instead of copied - Strata's STRATA_ROUTE_TAIL_SKIP for top-22 (it changes the answers slightly)
    int64_t tail_skip = std::getenv("STRATA_NEMO_TAIL_SKIP") ? std::atoll(std::getenv("STRATA_NEMO_TAIL_SKIP")) : 0;
    int64_t topk_cut = std::getenv("STRATA_NEMO_TOPK_CUT") ? std::atoll(std::getenv("STRATA_NEMO_TOPK_CUT")) : 0;
    FILE* route_trace = std::getenv("STRATA_NEMO_ROUTE_TRACE") ? fopen(std::getenv("STRATA_NEMO_ROUTE_TRACE"), "ab") : nullptr;
    // STRATA_NEMO_ZERO_STATS=1 (research): per MoE layer, the experts' relu^2 hidden values in decode / verify windows -
    // rows, values, exact zeros, all-zero blocks of 16 and of 64 (NVFP4's scale block, ggml's superblock); printed at exit
    bool zero_stats = std::getenv("STRATA_NEMO_ZERO_STATS") != nullptr;
    std::vector<std::array<uint64_t, 5>> zs;
    std::vector<float> zs_host;
    FILE* zs_dump = std::getenv("STRATA_NEMO_ZERO_DUMP") ? fopen(std::getenv("STRATA_NEMO_ZERO_DUMP"), "wb") : nullptr;
    int64_t zs_dumped = 0;
    double prof_ms[9] = {0};
    uint64_t prof_tokens = 0;
    std::chrono::steady_clock::time_point prof_t;
    void mark(int phase) {
        if (!prof) return;
        cudaStreamSynchronize(st);
        const auto now = std::chrono::steady_clock::now();
        if (phase >= 0) prof_ms[phase] += std::chrono::duration<double, std::milli>(now - prof_t).count();
        prof_t = now;
    }

    bool alloc(void** p, size_t bytes, std::string& err) {
        cudaError_t e = cudaMalloc(p, bytes);
        if (e != cudaSuccess) {
            err = "cudaMalloc(" + std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(e);
            return false;
        }
        allocs.push_back(*p);
        return true;
    }
    template <class T>
    bool allocT(T** p, size_t n, std::string& err) {
        return alloc((void**) p, n * sizeof(T), err);
    }

    const strata::TensorInfo* tensor(const std::string& name, const uint8_t** data) const {
        size_t s = 0;
        const strata::TensorInfo* t = gguf->find(name, &s);
        if (t && data) *data = gguf->shard(s).tensor_data(*t);
        return t;
    }
    /// A routed-expert tensor: from --experts-gguf when given.
    const strata::TensorInfo* xtensor(const std::string& name, const uint8_t** data) const {
        if (!xgguf) return tensor(name, data);
        size_t s = 0;
        const strata::TensorInfo* t = xgguf->find(name, &s);
        if (t && data) *data = xgguf->shard(s).tensor_data(*t);
        return t;
    }

    /// A tensor copied to the device as it is.
    bool upload(const std::string& name, void** d, std::string& err, uint64_t* bytes_out = nullptr) {
        const uint8_t* src = nullptr;
        const strata::TensorInfo* t = tensor(name, &src);
        if (!t) {
            err = "missing tensor " + name;
            return false;
        }
        const uint64_t b = strata::tensor_payload_bytes(*t);
        if (!alloc(d, b, err)) return false;
        NCK(cudaMemcpy(*d, src, b, cudaMemcpyHostToDevice));
        if (bytes_out) *bytes_out += b;
        return true;
    }

    /// A matrix: its data, its type, and an NVFP4 tensor's scale_2 (`<stem>.scale`).
    bool upload_mat(const std::string& name, Mat& m, std::string& err, uint64_t* bytes) {
        const strata::TensorInfo* t = tensor(name, nullptr);
        if (!t) {
            err = "missing tensor " + name;
            return false;
        }
        if (t->type != 40 && t->type != 8 && t->type != 30 && t->type != 0 && t->type != 2) {
            err = name + " is " + t->type_name() + ": the v1 runner reads NVFP4, Q8_0, Q4_0, BF16 and F32 matrices";
            return false;
        }
        m.t = (NemoWType) t->type;
        m.cols = (int64_t) t->shape[0];
        m.rows = (int64_t) t->shape[1];
        m.s2 = 1.0f;
        std::vector<uint16_t> q4;   // a Q4_0 matrix (an MTP file's) goes in as its exact BF16 values
        const uint8_t* src_override = nullptr;
        if (t->type == 2) {
            const uint8_t* q = nullptr;
            tensor(name, &q);
            q4.resize((size_t) m.rows * m.cols);
            for (int64_t i = 0; i < m.rows * m.cols / 32; ++i) {
                const uint8_t* blk = q + (size_t) i * 18;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d = half_to_float(dh);
                for (int j = 0; j < 16; ++j) {
                    q4[(size_t) i * 32 + j] = float_to_bf16(d * (float) ((blk[2 + j] & 15) - 8));
                    q4[(size_t) i * 32 + j + 16] = float_to_bf16(d * (float) ((blk[2 + j] >> 4) - 8));
                }
            }
            m.t = NemoWType::BF16;
            src_override = reinterpret_cast<const uint8_t*>(q4.data());
        }
        if (t->type == 40) {
            const uint8_t* sp = nullptr;
            const strata::TensorInfo* sc = tensor(name.substr(0, name.size() - 7) + ".scale", &sp);
            if (!sc) {
                err = name + " is NVFP4 without its .scale";
                return false;
            }
            std::memcpy(&m.s2, sp, 4);
        }
        // NVFP4 / Q8_0 / BF16 go in the tiled layout (STRATA_NEMO_TILED=0: the GGUF rows and the CUDA-core kernels)
        if (tiled_ok && strata::kernels::nemo_tc_gemv_ok(m.t, m.cols, 1)) {
            const uint8_t* src = src_override;
            if (!src) tensor(name, &src);
            std::vector<uint8_t> tl((size_t) strata::kernels::nemo_tiled_bytes(m.t, m.rows, m.cols));
            strata::kernels::nemo_tiled_repack(m.t, src, m.rows, m.cols, tl.data());
            if (!alloc(&m.d, tl.size(), err)) return false;
            NCK(cudaMemcpy(m.d, tl.data(), tl.size(), cudaMemcpyHostToDevice));
            if (bytes) *bytes += tl.size();
            m.tiled = true;
            return true;
        }
        if (src_override) {
            if (!alloc(&m.d, q4.size() * 2, err)) return false;
            NCK(cudaMemcpy(m.d, src_override, q4.size() * 2, cudaMemcpyHostToDevice));
            if (bytes) *bytes += q4.size() * 2;
            return true;
        }
        return upload(name, &m.d, err, bytes);
    }
    /// probe STRATA_NEMO_A4 = bits: the prompt's activations rounded through NVFP4 for 1 the dense NVFP4 matrices,
    /// 2 the experts' up, 4 their down (what a W4A4 path would cost)
    int a4 = std::getenv("STRATA_NEMO_A4") ? std::atoi(std::getenv("STRATA_NEMO_A4")) : 0;
    /// the prompt's NVFP4 GEMMs on FP4 tensor cores (nemo_fp4, W4A4), STRATA_NEMO_FP4 = bits as a4 (default 7: all;
    /// 0: the BF16 expansion and cuBLAS); set at load when the device has the instruction
    int fp4 = 0;
    /// the prompt's Q8_0 (bit 1) and BF16 (bit 2) GEMMs on int8 tensor cores (nemo_q8, W8A8), STRATA_NEMO_Q8 = bits
    /// (default 3; 0: the BF16 expansion and cuBLAS)
    /// STRATA_NEMO_LATQ=1 (a probe): the experts' latent rows quantized once per token and gathered by the up GEMM,
    /// instead of per (token, expert); measured even: the quantization saves 26 ms a 16K chunk, the gathering GEMM
    /// loses 25
    bool lat_q_off = std::getenv("STRATA_NEMO_LATQ") == nullptr;
    int q8 = std::getenv("STRATA_NEMO_Q8") ? std::atoi(std::getenv("STRATA_NEMO_Q8")) : 3;
    bool tiled_ok = !std::getenv("STRATA_NEMO_TILED") || std::atoi(std::getenv("STRATA_NEMO_TILED")) != 0;

    void gemm(const Mat& m, const float* x_, int64_t ldx, float* y_, int64_t ldy, int64_t M, bool relu2, bool acc) {
        if (m.tiled && M < TC_MIN) {
            // decode and verify windows: the tensor-core GEMV, 8 rows a call (a row's result does not depend on M)
            for (int64_t m0 = 0; m0 < M; m0 += 8)
                strata::kernels::nemo_tiled_gemv(m.t, m.d, m.s2, x_ + m0 * ldx, ldx, y_ + m0 * ldy, ldy, m.rows,
                                                 m.cols, std::min<int64_t>(8, M - m0), relu2, acc, st);
            return;
        }
        if (M >= TC_MIN && m.t == NemoWType::F32 && wb && (q8 & 4) && ldx % 4 == 0 && m.rows % 128 == 0 &&
            m.cols % 128 == 0) {
            // STRATA_NEMO_Q8 bit 4 (a probe): the router on int8 tensor cores, its FP32 weight rounded to Q8 blocks
            int8_t* aq = (int8_t*) xb;
            float* as = (float*) (aq + (size_t) M * m.cols);
            int8_t* wq = (int8_t*) wb;
            float* ws = (float*) (wq + (size_t) m.rows * m.cols);
            n4_of[0] = n4_of[1] = nullptr;
            strata::kernels::nemo_q8_quant_rows(x_, ldx, M, m.cols, aq, as, st);
            strata::kernels::nemo_q8_quant_rows((const float*) m.d, m.cols, m.rows, m.cols, wq, ws, st);
            strata::kernels::nemo_q8_gemm(aq, as, wq, ws, 1.0f, y_, ldy, M, m.rows, m.cols, acc, st);
            if (relu2) strata::kernels::nemo_relu2_rows(y_, ldy, M, m.rows, st);
            return;
        }
        if (M >= TC_MIN && m.t == NemoWType::F32 && wb) {   // the router: cuBLAS SGEMM (FP32 math)
            strata::kernels::nemo_gemm_f32(x_, ldx, (const float*) m.d, y_, ldy, m.rows, m.cols, M, acc, st);
            if (relu2) strata::kernels::nemo_relu2_rows(y_, ldy, M, m.rows, st);
            return;
        }
        if (M >= TC_MIN && (fp4 & 1) && m.t == NemoWType::NVFP4 && m.tiled && m.rows % 128 == 0 && m.cols % 64 == 0 &&
            wb) {
            // W4A4: the rows to NVFP4 in xb's room, the weight to N4 in wb's, the FP4 tensor cores
            uint8_t* aq = (uint8_t*) xb;
            uint8_t* as = aq + (size_t) M * m.cols / 2;
            float* ags = (float*) (as + (size_t) M * m.cols / 16);
            const size_t slot_b = (wb_elems * 9 / 16 + 255) / 256 * 256;   // an N4 weight: 9/16 byte a value
            int sl = n4_of[0] == m.d ? 0 : n4_of[1] == m.d ? 1 : -1;
            const bool fresh = sl < 0;
            if (fresh) {
                sl = n4_next;
                n4_next ^= 1;
                n4_of[sl] = m.d;
            }
            uint8_t* wq = (uint8_t*) wb + (size_t) sl * slot_b;
            uint8_t* ws = wq + (size_t) m.rows * m.cols / 2;
            strata::kernels::nemo_fp4_quant_rows(x_, ldx, nullptr, M, m.cols, false, aq, as, ags, st);
            if (fresh) strata::kernels::nemo_fp4_from_tiled((const uint8_t*) m.d, m.rows, m.cols, wq, ws, st);
            strata::kernels::nemo_fp4_gemm(aq, as, ags, wq, ws, m.s2, y_, ldy, M, m.rows, m.cols, relu2, acc, st);
            return;
        }
        if (M >= TC_MIN && m.tiled && wb && ldx % 4 == 0 && m.rows % 128 == 0 && m.cols % 128 == 0 &&
            (((q8 & 1) && m.t == NemoWType::Q8_0) || ((q8 & 2) && m.t == NemoWType::BF16))) {
            // W8A8 (nemo_q8): the rows to Q8 blocks in xb's room, the weight to Q8N in wb's (Q8_0 as it is, BF16
            // rounded to Q8_0), the int8 tensor cores
            int8_t* aq = (int8_t*) xb;
            float* as = (float*) (aq + (size_t) M * m.cols);
            int8_t* wq = (int8_t*) wb;
            float* ws = (float*) (wq + (size_t) m.rows * m.cols);
            n4_of[0] = n4_of[1] = nullptr;   // (wb is overwritten)
            strata::kernels::nemo_q8_quant_rows(x_, ldx, M, m.cols, aq, as, st);
            if (m.t == NemoWType::Q8_0) strata::kernels::nemo_q8_from_tiled_q8((const uint8_t*) m.d, m.rows, m.cols, wq, ws, st);
            else strata::kernels::nemo_q8_from_tiled_bf16((const uint8_t*) m.d, m.rows, m.cols, wq, ws, st);
            strata::kernels::nemo_q8_gemm(aq, as, wq, ws, m.s2, y_, ldy, M, m.rows, m.cols, acc, st);
            if (relu2) strata::kernels::nemo_relu2_rows(y_, ldy, M, m.rows, st);
            return;
        }
        if (M >= TC_MIN && m.t != NemoWType::F32 && wb) {
            // a chunk of the prompt: the weight to BF16 (exact for NVFP4; scale_2 is the GEMM's alpha), cuBLAS
            strata::kernels::nemo_f32_to_bf16(x_, ldx, xb, M, m.cols, false, st);
            if ((a4 & 1) && m.t == NemoWType::NVFP4) strata::kernels::nemo_fq_nvfp4_bf16(xb, M, m.cols, st);
            const uint16_t* w = (const uint16_t*) m.d;
            if (m.tiled || m.t != NemoWType::BF16) n4_of[0] = n4_of[1] = nullptr;   // (wb is overwritten)
            if (m.tiled) {
                strata::kernels::nemo_tiled_dequant_bf16(m.t, m.d, wb, m.rows, m.cols, st);
                w = wb;
            } else if (m.t != NemoWType::BF16) {
                strata::kernels::nemo_dequant_bf16(m.t, m.d, wb, m.rows, m.cols, st);
                w = wb;
            }
            strata::kernels::nemo_gemm_bf16(xb, w, m.s2, y_, ldy, m.rows, m.cols, M, acc, st);
            if (relu2) strata::kernels::nemo_relu2_rows(y_, ldy, M, m.rows, st);
            return;
        }
        if (strata::kernels::nemo_gemv_ok(m.t, m.cols, M)) {
            strata::kernels::nemo_gemv(m.t, m.d, m.s2, x_, ldx, y_, ldy, m.rows, m.cols, M, relu2, acc, st);
            return;
        }
        strata::kernels::nemo_gemm(m.t, m.d, m.s2, x_, ldx, y_, ldy, m.rows, m.cols, M, relu2, acc, st);
    }
};

NemoModel::NemoModel() : impl_(new Impl) {}

NemoModel::~NemoModel() {
    if (!impl_) return;
    if (impl_->lfu_check)
        std::fprintf(stderr, "lfu check: %llu victim searches, %llu mismatches\n",
                     (unsigned long long) impl_->lfu_checked, (unsigned long long) impl_->lfu_mismatch);
    if (impl_->zero_stats && !impl_->zs.empty()) {
        const int64_t FF = g_.n_ff_exp;
        std::array<uint64_t, 5> all{};
        std::fprintf(stderr, "zero stats (relu^2 hidden of the experts, decode): moe layer, rows, zero values %%, "
                             "zero 16-blocks %%, zero 64-blocks %%\n");
        for (size_t l = 0; l < impl_->zs.size(); ++l) {
            const auto& z = impl_->zs[l];
            for (int k = 0; k < 5; ++k) all[(size_t) k] += z[(size_t) k];
            if (!z[0]) continue;
            std::fprintf(stderr, "  %2zu %8llu %6.2f %6.2f %6.2f\n", l, (unsigned long long) z[0], 100.0 * z[2] / z[1],
                         100.0 * z[3] / (z[0] * ((FF + 15) / 16)), 100.0 * z[4] / (z[0] * (FF / 64)));
        }
        if (all[0])
            std::fprintf(stderr, "  all %llu rows: zero values %.2f%%, zero 16-blocks %.2f%%, zero 64-blocks %.2f%%\n",
                         (unsigned long long) all[0], 100.0 * all[2] / all[1], 100.0 * all[3] / (all[0] * ((FF + 15) / 16)),
                         100.0 * all[4] / (all[0] * (FF / 64)));
    }
    for (void* p : impl_->allocs) cudaFree(p);
    for (KvPages* l : impl_->kv_lists)
        for (size_t i = 0; i < l->k.size(); ++i) { cudaFree(l->k[i]); cudaFree(l->ks[i]); cudaFree(l->v[i]); }
    for (ExpertLayer& X : impl_->ex)
        for (uint8_t* b : X.bands) cudaFree(b);
    if (impl_->arena) cudaFreeHost(impl_->arena);
    if (impl_->embd_stage) cudaFreeHost(impl_->embd_stage);
    if (impl_->embd_pinned) cudaFreeHost(impl_->embd_pinned);
    if (impl_->img_host) { if (impl_->img_pinned) cudaFreeHost(impl_->img_host); else std::free(impl_->img_host); }
    if (impl_->st) cudaStreamDestroy(impl_->st);
}

bool NemoModel::set_image_rows(const float* rows, const std::vector<std::pair<int64_t, int64_t>>& runs, std::string& err) {
    Impl& I = *impl_;
    const int64_t E = g_.n_embd;
    NCK(cudaStreamSynchronize(I.st));   // a copy from the previous rows may still be in flight
    I.img_runs.clear();
    int64_t total = 0;
    for (const auto& r : runs) {
        if (r.first < 0 || r.second <= 0 || r.first + r.second > max_ctx_) {
            err = "an image run lies outside the context";
            return false;
        }
        I.img_runs.push_back({r.first, r.second, total});
        total += r.second;
    }
    if ((size_t) (total * E) > I.img_cap) {
        if (I.img_host) { if (I.img_pinned) cudaFreeHost(I.img_host); else std::free(I.img_host); }
        I.img_host = nullptr;
        I.img_cap = 0;
        I.img_pinned = cudaMallocHost((void**) &I.img_host, (size_t) total * E * sizeof(float)) == cudaSuccess;
        if (!I.img_pinned) {
            // page-locked RAM is used up by the experts (Windows caps it well below the RAM size): the image rows are
            // a few MB, so pageable memory is fine - the copy to the GPU is synchronous instead of async
            cudaGetLastError();
            I.img_host = (float*) std::malloc((size_t) total * E * sizeof(float));
            if (!I.img_host) {
                I.img_runs.clear();
                err = "no memory for the image rows";
                return false;
            }
        }
        I.img_cap = (size_t) (total * E);
    }
    if (total) std::memcpy(I.img_host, rows, (size_t) total * E * sizeof(float));
    return true;
}

bool NemoModel::load(const NemoOptions& opt, std::string& err) {
    Impl& I = *impl_;
    // the process's VRAM before its first allocation: --vram-limit-mib counts from here (plus the context's, see below)
    size_t free0 = 0, total0 = 0;
    cudaFree(0);
    cudaMemGetInfo(&free0, &total0);
    try {
        I.gguf.reset(new strata::GgufModel(strata::gguf_split_paths(opt.gguf)));
        if (!opt.experts_gguf.empty())
            I.xgguf.reset(new strata::GgufModel(strata::gguf_split_paths(opt.experts_gguf)));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (!nemotron_geometry(*I.gguf, g_, err)) return false;
    NemotronCheck chk;
    if (!nemotron_check(*I.gguf, g_, chk, err)) return false;
    max_ctx_ = opt.max_context;
    I.C = std::max<int64_t>(1, opt.chunk);
    // decode synchronizes with the host once per MoE layer: spin, not yield / block - a yielding wait wakes ~100 us
    // after the read-back lands, 40 times a token (STRATA_NEMO_SYNC=yield|block to change)
    {
        const char* sm = std::getenv("STRATA_NEMO_SYNC");
        const unsigned fl = sm && !std::strcmp(sm, "block") ? cudaDeviceScheduleBlockingSync
                            : sm && !std::strcmp(sm, "yield") ? cudaDeviceScheduleYield : cudaDeviceScheduleSpin;
        cudaSetDeviceFlags(fl);   // (fails harmlessly when the context already exists)
        cudaGetLastError();
    }
    NCK(cudaStreamCreateWithFlags(&I.st, cudaStreamNonBlocking));

    const int64_t E = g_.n_embd, DI = g_.ssm_d_inner, CD = g_.ssm_conv_dim(), H = g_.ssm_n_head;
    const int64_t QD = g_.n_head * g_.head_dim, KD = g_.n_head_kv * g_.head_dim;
    const int64_t NE = g_.n_expert, K = g_.n_expert_used, L = g_.moe_latent, FF = g_.n_ff_exp, SH = g_.n_ff_shexp;
    const int64_t C = I.C;
    uint64_t dense = 0;

    // STRATA_NEMO_VRAM_REPORT=1: the card's used memory after each part of the load (cudaMemGetInfo deltas)
    const bool vram_report = std::getenv("STRATA_NEMO_VRAM_REPORT") != nullptr;
    size_t vram_last = 0;
    auto vram_mark = [&](const char* what) {
        if (!vram_report) return;
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const size_t used = tot - fr;
        if (vram_last) std::fprintf(stderr, "vram: %-36s %8.1f MiB\n", what, (double) (used - vram_last) / 1048576.0);
        vram_last = used;
    };
    vram_mark("start");
    // ---- dense weights and per-sequence state ----
    I.blocks.resize((size_t) g_.n_layers);
    int moe_i = 0, mamba_i = 0;
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        Block& b = I.blocks[(size_t) l];
        b.kind = g_.blocks[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        if (!I.upload(p + "attn_norm.weight", (void**) &b.norm, err, &dense)) return false;
        switch (b.kind) {
        case NemotronBlock::Mamba:
            b.mamba = mamba_i++;
            if (!I.upload_mat(p + "ssm_in.weight", b.in, err, &dense) ||
                !I.upload_mat(p + "ssm_out.weight", b.out, err, &dense) ||
                !I.upload(p + "ssm_conv1d.weight", (void**) &b.conv_w, err, &dense) ||
                !I.upload(p + "ssm_conv1d.bias", (void**) &b.conv_b, err, &dense) ||
                !I.upload(p + "ssm_dt.bias", (void**) &b.dt_b, err, &dense) ||
                !I.upload(p + "ssm_a", (void**) &b.A, err, &dense) ||
                !I.upload(p + "ssm_d", (void**) &b.D, err, &dense) ||
                !I.upload(p + "ssm_norm.weight", (void**) &b.ssm_norm, err, &dense))
                return false;
            if (!I.allocT(&b.state, (size_t) H * g_.ssm_head_dim() * g_.ssm_d_state, err) ||
                !I.allocT(&b.conv, (size_t) CD * (g_.ssm_d_conv - 1), err))
                return false;
            break;
        case NemotronBlock::Attention:
            if (!I.upload_mat(p + "attn_q.weight", b.q, err, &dense) ||
                !I.upload_mat(p + "attn_k.weight", b.k, err, &dense) ||
                !I.upload_mat(p + "attn_v.weight", b.v, err, &dense) ||
                !I.upload_mat(p + "attn_output.weight", b.o, err, &dense))
                return false;
            break;   // (its K/V pages: below, with the MTP's)
        case NemotronBlock::Moe:
            b.moe = moe_i++;
            if (!I.upload_mat(p + "ffn_gate_inp.weight", b.gate, err, &dense) ||
                !I.upload_mat(p + "ffn_latent_down.weight", b.fc1, err, &dense) ||
                !I.upload_mat(p + "ffn_latent_up.weight", b.fc2, err, &dense) ||
                !I.upload_mat(p + "ffn_up_shexp.weight", b.sh_up, err, &dense) ||
                !I.upload_mat(p + "ffn_down_shexp.weight", b.sh_dn, err, &dense) ||
                !I.upload(p + "exp_probs_b.bias", (void**) &b.bias, err, &dense))
                return false;
            break;
        }
    }
    if (!I.upload("output_norm.weight", (void**) &I.out_norm, err, &dense) ||
        !I.upload_mat("output.weight", I.head, err, &dense))
        return false;
    {
        const uint8_t* e = nullptr;
        const strata::TensorInfo* t = I.tensor("token_embd.weight", &e);
        if (t->type != 30) {
            err = std::string("token_embd.weight is ") + t->type_name() + ", the v1 runner reads BF16 embeddings";
            return false;
        }
        // a page-locked copy: read from the mapping, a token's row can page-fault to the disk once the pinned expert
        // arena has pushed the file out of the page cache (~5 ms a decode step, measured)
        const size_t eb = (size_t) g_.n_vocab * E * sizeof(uint16_t);
        uint16_t* copy = nullptr;
        if (cudaMallocHost((void**) &copy, eb) == cudaSuccess) {
            std::memcpy(copy, e, eb);
            I.embd_pinned = copy;
            I.embd = copy;
        } else {
            cudaGetLastError();
            I.embd = (const uint16_t*) e;
        }
    }
    stats_.dense_gib = dense / 1073741824.0;

    vram_mark("dense weights + state");
    // ---- the experts: in place in the mapping, page-locked ----
    I.ex.resize((size_t) moe_i);
    I.pq_first.assign((size_t) moe_i, SIZE_MAX);
    I.pq_end.assign((size_t) moe_i, 0);
    I.pitem.assign((size_t) moe_i, std::vector<int32_t>((size_t) NE, -1));
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        const Block& b = I.blocks[(size_t) l];
        if (b.kind != NemotronBlock::Moe) continue;
        ExpertLayer& X = I.ex[(size_t) b.moe];
        const std::string p = "blk." + std::to_string(l) + ".";
        const strata::TensorInfo* tu = I.xtensor(p + "ffn_up_exps.weight", &X.up);
        const strata::TensorInfo* td = I.xtensor(p + "ffn_down_exps.weight", &X.dn);
        if (!tu || !td) {
            err = p + "ffn_*_exps is missing";
            return false;
        }
        auto fmt = [](uint32_t t, NemoWType* o) {
            switch (t) {
            case 40: *o = NemoWType::NVFP4; return true;
            case 42: *o = NemoWType::Q2_0; return true;
            case 20: *o = NemoWType::IQ4_NL; return true;
            case 23: *o = NemoWType::IQ4_XS; return true;
            case 18: *o = NemoWType::IQ3_XXS; return true;
            default: return false;
            }
        };
        if (!fmt(tu->type, &X.tu) || !fmt(td->type, &X.td) || L % strata::kernels::nemo_block_values(X.tu) != 0 ||
            FF % strata::kernels::nemo_block_values(X.td) != 0) {
            err = "experts in " + std::string(tu->type_name()) + "/" + td->type_name() + " (" + p +
                  "ffn_*_exps): the runner reads NVFP4, Q2_0, IQ4_NL, IQ4_XS and IQ3_XXS";
            return false;
        }
        X.ub = strata::kernels::nemo_row_bytes(X.tu, L) * FF;
        X.db = strata::kernels::nemo_row_bytes(X.td, FF) * L;
        X.stride = (X.ub + X.db + 15) / 16 * 16;
        I.blob = std::max<uint64_t>(I.blob, X.stride);
        X.s_up.assign((size_t) NE, 1.0f);
        X.s_dn.assign((size_t) NE, 1.0f);
        const uint8_t* s = nullptr;
        if (X.tu == NemoWType::NVFP4 && I.xtensor(p + "ffn_up_exps.scale", &s)) std::memcpy(X.s_up.data(), s, NE * 4);
        if (X.td == NemoWType::NVFP4 && I.xtensor(p + "ffn_down_exps.scale", &s)) std::memcpy(X.s_dn.data(), s, NE * 4);
    }
    // A page-locked RAM arena holding every expert: a file mapping cannot be page-locked by cudaHostRegister here
    // (it refused the read-only GGUF mapping), and a copy from pageable memory goes through the driver's staging at
    // a fraction of PCIe's speed.  The arena is filled from the mapping at load (from the page cache when the file is
    // in it), one MoE layer at a time on 8 threads.
    {
        size_t total = 0;
        for (const ExpertLayer& X : I.ex) total += (size_t) NE * (X.ub + X.db);
        uint8_t* arena = nullptr;
        if (cudaMallocHost((void**) &arena, total) != cudaSuccess) {
            cudaGetLastError();
            I.pinned = false;
            std::fprintf(stderr, "strata-nemotron: no page-locked arena for the experts (%.1f GiB): copies go through "
                                 "the driver's staging (slower)\n", total / 1073741824.0);
        } else {
            I.pinned = true;
            I.arena = arena;
            uint8_t* base = arena;
            for (size_t li = 0; li < I.ex.size(); ++li) {
                ExpertLayer& X = I.ex[li];
                // arena layout per layer: all up blocks, then all down blocks (each expert's as in the file)
                std::vector<std::thread> th;
                const size_t parts = 8;
                const size_t ub = (size_t) NE * X.ub, db = (size_t) NE * X.db;
                for (size_t p = 0; p < parts; ++p)
                    th.emplace_back([&, p] {
                        const size_t u0 = ub * p / parts, u1 = ub * (p + 1) / parts;
                        const size_t d0 = db * p / parts, d1 = db * (p + 1) / parts;
                        std::memcpy(base + u0, X.up + u0, u1 - u0);
                        std::memcpy(base + ub + d0, X.dn + d0, d1 - d0);
                    });
                for (auto& t : th) t.join();
                X.up = base;
                X.dn = base + ub;
                base += ub + db;
            }
        }
    }
    stats_.pinned = I.pinned;

    vram_mark("expert arena (host)");
    // ---- scratch for a chunk of C rows ----
    // The Mamba blocks and the shared expert run over the chunk in sub-chunks of SC rows (the scan carries its state
    // across them), so their wide buffers (zx 18,560 floats a row, the shared expert's 5,376) are SC rows, not C.
    // The prompt's MoE adds each group's outputs into `routed` as it goes, so the (t, j) parts are needed only by the
    // CUDA-core path of a chunk below TC_MIN rows.
    I.SC = std::min<int64_t>(C, 2048);
    const int64_t SC = I.SC, PR = C >= Impl::TC_MIN ? 1 : C;
    if (!I.allocT(&I.x, (size_t) C * E, err) || !I.allocT(&I.xn, (size_t) C * E, err) ||
        !I.allocT(&I.zx, (size_t) SC * g_.ssm_in_dim(), err) || !I.allocT(&I.xbc, (size_t) SC * CD, err) ||
        !I.allocT(&I.y, (size_t) SC * DI, err) || !I.allocT(&I.q, (size_t) C * QD, err) ||
        !I.allocT(&I.k, (size_t) C * KD, err) || !I.allocT(&I.v, (size_t) C * KD, err) ||
        !I.allocT(&I.o, (size_t) C * QD, err) || !I.allocT(&I.logits, (size_t) C * NE, err) ||
        !I.allocT(&I.w, (size_t) C * K, err) || !I.allocT(&I.lat, (size_t) C * L, err) ||
        !I.allocT(&I.parts, (size_t) PR * K * L, err) || !I.allocT(&I.routed, (size_t) C * L, err) ||
        !I.allocT(&I.g_in, (size_t) PR * L, err) || !I.allocT(&I.g_h, (size_t) PR * FF, err) ||
        !I.allocT(&I.g_out, (size_t) PR * L, err) || !I.allocT(&I.sh_h, (size_t) SC * SH, err) ||
        !I.allocT(&I.eh, (size_t) strata::kernels::NEMO_GEMV_MAX_M * K * (FF + L), err) ||
        !I.allocT(&I.ids, (size_t) C * K, err) ||
        !I.allocT(&I.idx, (size_t) 2 * C * K, err) || !I.allocT(&I.es2, (size_t) 2 * K, err) ||
        !I.allocT(&I.head_out, (size_t) g_.n_vocab, err) ||
        !I.alloc((void**) &I.eptr, 2 * K * sizeof(void*), err))
        return false;
    {
        strata::kernels::NemoAttnShapes as;
        as.n_head = g_.n_head; as.n_head_kv = g_.n_head_kv; as.head_dim = g_.head_dim;
        if (!I.alloc(&I.attn_scratch, strata::kernels::nemo_attn_scratch_bytes(as, C, max_ctx_), err)) return false;
    }
    NCK(cudaMallocHost(&I.embd_stage, (size_t) C * E * sizeof(uint16_t)));
    if (C >= Impl::TC_MIN) {
        uint64_t most = 0, widest = 0;
        for (const Block& b : I.blocks)
            for (const Mat* mt : {&b.in, &b.out, &b.q, &b.k, &b.v, &b.o, &b.fc1, &b.fc2, &b.sh_up, &b.sh_dn})
                if (mt->d) {
                    most = std::max<uint64_t>(most, (uint64_t) (mt->rows * mt->cols));
                    widest = std::max<uint64_t>(widest, (uint64_t) mt->cols);
                }
        I.tc_rows = std::max<int64_t>(Impl::TC_ROWS, C);
        const int64_t R = I.tc_rows, G = Impl::TC_GROUP;
        // the experts' BF16 / N4 copies: only for the paths that make them (the FP4 GEMMs read the slots' ggml rows)
        const int fp4_bits = strata::kernels::nemo_fp4_ok()
                                 ? (std::getenv("STRATA_NEMO_FP4") ? std::atoi(std::getenv("STRATA_NEMO_FP4")) : 7) : 0;
        const bool copies = (fp4_bits & 6) != 6 || I.gg_max < C;
        const size_t GW = copies ? (size_t) G * FF * L : 16;
        I.wb_elems = most;
        if (!I.allocT(&I.wb, most, err) || !I.allocT(&I.xb, (size_t) C * widest, err) ||
            !I.allocT(&I.eg_up, GW, err) || !I.allocT(&I.eg_dn, GW, err) ||
            !I.allocT(&I.eg_in, (size_t) R * L, err) || !I.allocT(&I.eg_xb, (size_t) R * L, err) ||
            !I.allocT(&I.eg_h, (size_t) R * FF, err) || !I.allocT(&I.eg_hb, (size_t) R * FF, err) ||
            !I.allocT(&I.eg_out, (size_t) R * L, err))
            return false;
        // a pass's group lists: per group 3 R_g + 2 nseg_g + 1 ints (sum R_g = C K, nseg_g <= R_g); 2 pointers an expert
        // + the FP4 path's tiles (3 ints, <= C K / 128 + NE of them) and 2 alphas an expert
        I.eg_idx_cap = (size_t) 5 * C * K + 4 * NE + 64 + 3 * ((size_t) C * K / 128 + 2 * NE) + 2 * NE;
        I.eg_ptr_cap = (size_t) 2 * NE;
        if (!I.allocT(&I.routed_d, (size_t) C * L, err)) return false;
        const int64_t PC = std::min<int64_t>(C, I.pskip_max);
        if (!I.allocT(&I.pp_logits, (size_t) PC * NE, err) || !I.allocT(&I.pp_w, (size_t) PC * K, err) ||
            !I.allocT(&I.pp_ids, (size_t) PC * K, err))
            return false;
        if (!I.allocT(&I.eg_idx, I.eg_idx_cap, err) || !I.alloc((void**) &I.eg_ptr, I.eg_ptr_cap * sizeof(void*), err))
            return false;
        NCK(cudaMallocHost((void**) &I.h_eg_idx, I.eg_idx_cap * sizeof(int32_t)));
        NCK(cudaMallocHost((void**) &I.h_eg_ptr, I.eg_ptr_cap * sizeof(void*)));
        (void) G;
    }
    // the snapshot of the Mamba state
    for (const Block& b : I.blocks) {
        if (b.kind != NemotronBlock::Mamba) continue;
        float *s = nullptr, *c = nullptr;
        if (!I.allocT(&s, (size_t) H * g_.ssm_head_dim() * g_.ssm_d_state, err) ||
            !I.allocT(&c, (size_t) CD * (g_.ssm_d_conv - 1), err))
            return false;
        I.snap_state.push_back(s);
        I.snap_conv.push_back(c);
    }

    vram_mark("chunk scratch");
    // ---- the MTP block, resident (its experts too: ~1.5 GiB that the expert cache does not get) ----
    // --mtp-gguf: the block's tensors from that file (ggml-org's mtp-*.gguf: BF16, Q8_0 or Q4_0), the embeddings and
    // lm_head still the main file's; I.gguf points at it while the block loads
    std::unique_ptr<strata::GgufModel> main_gguf;
    if (opt.mtp && !opt.mtp_gguf.empty()) {
        try {
            main_gguf = std::move(I.gguf);
            I.gguf.reset(new strata::GgufModel(strata::gguf_split_paths(opt.mtp_gguf)));
        } catch (const std::exception& e) {
            err = std::string("--mtp-gguf: ") + e.what();
            return false;
        }
        std::fprintf(stderr, "strata-nemotron: the MTP block from %s\n", opt.mtp_gguf.c_str());
    }
    if (opt.mtp) {
        const std::string p = "blk." + std::to_string(g_.n_layers) + ".";
        Impl::Mtp& M = I.mtp;
        if (!I.tensor(p + "nextn.eh_proj.weight", nullptr)) {
            std::fprintf(stderr, "strata-nemotron: no MTP block (%snextn.*) in the file: drafting is off\n", p.c_str());
        } else {
            if (!I.upload_mat(p + "nextn.eh_proj.weight", M.eh, err, &dense) ||
                !I.upload(p + "nextn.enorm.weight", (void**) &M.enorm, err, &dense) ||
                !I.upload(p + "nextn.hnorm.weight", (void**) &M.hnorm, err, &dense) ||
                !I.upload(p + "attn_norm.weight", (void**) &M.attn_norm, err, &dense) ||
                !I.upload_mat(p + "attn_q.weight", M.q, err, &dense) ||
                !I.upload_mat(p + "attn_k.weight", M.k, err, &dense) ||
                !I.upload_mat(p + "attn_v.weight", M.v, err, &dense) ||
                !I.upload_mat(p + "attn_output.weight", M.o, err, &dense) ||
                !I.upload(p + "post_attention_norm.weight", (void**) &M.post_norm, err, &dense) ||
                !I.upload_mat(p + "ffn_gate_inp.weight", M.gate, err, &dense) ||
                !I.upload(p + "exp_probs_b.bias", (void**) &M.bias, err, &dense) ||
                !I.upload_mat(p + "ffn_latent_down.weight", M.fc1, err, &dense) ||
                !I.upload_mat(p + "ffn_latent_up.weight", M.fc2, err, &dense) ||
                !I.upload_mat(p + "ffn_up_shexp.weight", M.sh_up, err, &dense) ||
                !I.upload_mat(p + "ffn_down_shexp.weight", M.sh_dn, err, &dense) ||
                !I.upload(p + "nextn.shared_head_norm.weight", (void**) &M.head_norm, err, &dense))
                return false;
            const uint8_t *up = nullptr, *dn = nullptr;
            const strata::TensorInfo* tu = I.tensor(p + "ffn_up_exps.weight", &up);
            const strata::TensorInfo* td = I.tensor(p + "ffn_down_exps.weight", &dn);
            auto mtp_type = [](uint32_t ty) { return ty == 40 || ty == 8 || ty == 30 || ty == 2; };
            if (!tu || !td || !mtp_type(tu->type) || !mtp_type(td->type)) {
                err = "the MTP block's experts must be NVFP4, Q8_0, Q4_0 or BF16";
                return false;
            }
            M.tu = (NemoWType) tu->type;
            M.td = (NemoWType) td->type;
            M.ub = strata::kernels::nemo_row_bytes(M.tu, L) * FF;
            const uint64_t mdb = strata::kernels::nemo_row_bytes(M.td, FF) * L;
            M.stride = (M.ub + mdb + 15) / 16 * 16;
            // STRATA_NEMO_MTP_SLOTS=N (< 512): the MTP's experts in a cache of N slots over a page-locked copy; a
            // draft skips a missing expert (it only proposes: a weaker draft is refused, never wrong) and the copy
            // lands for the next draft.  Default: all 512 resident.
            M.nslots = std::getenv("STRATA_NEMO_MTP_SLOTS") ? std::atoll(std::getenv("STRATA_NEMO_MTP_SLOTS")) : NE;
            M.nslots = std::max<int64_t>(K, std::min<int64_t>(NE, M.nslots));
            if (!I.alloc((void**) &M.experts, (size_t) M.nslots * M.stride, err)) return false;
            if (M.nslots == NE) {
                for (int64_t e = 0; e < NE; ++e) {
                    NCK(cudaMemcpy(M.experts + e * M.stride, up + e * M.ub, M.ub, cudaMemcpyHostToDevice));
                    NCK(cudaMemcpy(M.experts + e * M.stride + M.ub, dn + e * mdb, mdb, cudaMemcpyHostToDevice));
                }
            } else {
                NCK(cudaMallocHost((void**) &M.host, (size_t) NE * M.stride));
                for (int64_t e = 0; e < NE; ++e) {
                    std::memcpy(M.host + e * M.stride, up + e * M.ub, M.ub);
                    std::memcpy(M.host + e * M.stride + M.ub, dn + e * mdb, mdb);
                }
                M.slot_of.assign((size_t) NE, -1);
                M.expert_in.assign((size_t) M.nslots, -1);
                M.used.assign((size_t) M.nslots, 0);
                NCK(cudaMallocHost((void**) &M.h_meta, (size_t) 2 * K * (sizeof(void*) + sizeof(float)) + K * 4));
                NCK(cudaEventCreateWithFlags(&M.fill_ev, cudaEventDisableTiming));
                NCK(cudaEventRecord(M.fill_ev, I.cst));
            }
            dense += (uint64_t) M.nslots * M.stride;
            std::vector<float> su((size_t) NE, 1.0f), sd((size_t) NE, 1.0f);
            const uint8_t* sp = nullptr;
            if (I.tensor(p + "ffn_up_exps.scale", &sp)) std::memcpy(su.data(), sp, NE * 4);
            if (I.tensor(p + "ffn_down_exps.scale", &sp)) std::memcpy(sd.data(), sp, NE * 4);
            if (!I.allocT(&M.s_up, (size_t) NE, err) || !I.allocT(&M.s_dn, (size_t) NE, err)) return false;
            M.su_h = su;
            M.sd_h = sd;
            NCK(cudaMemcpy(M.s_up, su.data(), NE * 4, cudaMemcpyHostToDevice));
            NCK(cudaMemcpy(M.s_dn, sd.data(), NE * 4, cudaMemcpyHostToDevice));
            M.MC = std::min<int64_t>(C, 2048);
            const int64_t MC = M.MC;
            if (!I.allocT(&M.e, (size_t) MC * E, err) || !I.allocT(&M.hn, (size_t) MC * E, err) ||
                !I.allocT(&M.cat, (size_t) MC * 2 * E, err) || !I.allocT(&M.u, (size_t) MC * E, err) ||
                !I.allocT(&M.a, (size_t) MC * E, err) || !I.allocT(&M.h, (size_t) E, err) ||
                !I.allocT(&M.snap_h, (size_t) E, err) || !I.allocT(&M.hnext, (size_t) E, err) ||
                !I.allocT(&M.logits, (size_t) g_.n_vocab, err) || !I.allocT(&M.d_tok, (size_t) 8, err) ||
                !I.allocT(&M.s2, (size_t) 2 * K, err) || !I.alloc((void**) &M.ptr, 2 * K * sizeof(void*), err))
                return false;
            NCK(cudaMallocHost((void**) &M.h_tok, 8 * sizeof(int32_t)));
            NCK(cudaMallocHost((void**) &M.stage, (size_t) MC * E * sizeof(uint16_t)));
            M.on = true;
            // speculation's buffers
            const int64_t MMr = strata::kernels::NEMO_GEMV_MAX_M;
            for (const Block& b : I.blocks) {
                if (b.kind != NemotronBlock::Mamba) continue;
                float *ss = nullptr, *cc = nullptr;
                if (!I.allocT(&ss, (size_t) H * g_.ssm_head_dim() * g_.ssm_d_state, err) ||
                    !I.allocT(&cc, (size_t) CD * (g_.ssm_d_conv - 1), err))
                    return false;
                I.spec_state.push_back(ss);
                I.spec_conv.push_back(cc);
            }
            if (!I.allocT(&I.spec_zx, I.spec_state.size() * MMr * g_.ssm_in_dim(), err) ||
                !I.allocT(&I.vlogits, (size_t) MMr * g_.n_vocab, err) || !I.allocT(&I.vtok, (size_t) MMr, err))
                return false;
            NCK(cudaMallocHost((void**) &I.h_vtok, MMr * sizeof(int32_t)));
            if (Impl::env_on("STRATA_NEMO_GPU_TOPK")) {
                if (!I.allocT(&I.d_tk_v, (size_t) MMr * 1024, err) || !I.allocT(&I.d_tk_i, (size_t) MMr * 1024, err))
                    return false;
                NCK(cudaMallocHost((void**) &I.h_tk_v, MMr * 1024 * sizeof(float)));
                NCK(cudaMallocHost((void**) &I.h_tk_i, MMr * 1024 * sizeof(int32_t)));
            }
        }
    }
    if (main_gguf) {   // the main file again (the MTP file stays open: its mapping backs nothing now, but cheap)
        I.mgguf = std::move(I.gguf);
        I.gguf = std::move(main_gguf);
    }

    stats_.dense_gib = dense / 1073741824.0;

    // a zero expert: a skipped miss (STRATA_NEMO_TAIL_SKIP) points here and adds exactly 0
    const size_t zb = std::max<size_t>((size_t) I.blob, (size_t) I.mtp.stride);
    if (!I.alloc((void**) &I.zero_blob, zb, err)) return false;
    NCK(cudaMemset(I.zero_blob, 0, zb));
    NCK(cudaStreamCreateWithFlags(&I.cst, cudaStreamNonBlocking));
    NCK(cudaEventCreateWithFlags(&I.cev, cudaEventDisableTiming));
    if (strata::kernels::nemo_fp4_ok())
        I.fp4 = std::getenv("STRATA_NEMO_FP4") ? std::atoi(std::getenv("STRATA_NEMO_FP4")) : 7;
    // the FP4 down projection writes each expert output to its (t, j) row; one weighted sum in rank order follows
    if ((I.fp4 & 4) && C >= Impl::TC_MIN && I.wb && !I.allocT(&I.parts_bf, (size_t) C * K * L, err)) return false;
    // the prompt's expert ring (see Impl::pq)
    if (C >= Impl::TC_MIN && I.ring_n > 0) {
        I.ring_n = std::max<int64_t>(I.ring_n, 2 * Impl::TC_GROUP);
        if (!I.alloc((void**) &I.ring, (size_t) I.ring_n * I.blob, err)) return false;
        I.ring_rel.assign((size_t) I.ring_n, -1);
        I.ring_busy.assign((size_t) I.ring_n, 0);
        I.pev.resize((size_t) I.ring_n + 64);
        for (cudaEvent_t& e : I.pev) NCK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        NCK(cudaEventCreateWithFlags(&I.pcopy, cudaEventDisableTiming));
    }
    for (size_t i = 0; i < I.blocks.size(); ++i)
        if (I.blocks[i].kind == NemotronBlock::Moe) I.moe_block.push_back((int) i);
    const int64_t MM = strata::kernels::NEMO_GEMV_MAX_M, KM = MM * K;   // a verify window's rows, entries
    if (!I.allocT(&I.pxn, (size_t) MM * E, err) || !I.allocT(&I.plogits, (size_t) MM * NE, err) ||
        !I.allocT(&I.pw, (size_t) KM, err) || !I.allocT(&I.pids, (size_t) KM, err))
        return false;
    NCK(cudaMallocHost((void**) &I.h_pids, (size_t) KM * sizeof(int32_t)));
    NCK(cudaMallocHost((void**) &I.h_dids, (size_t) KM * sizeof(int32_t)));
    NCK(cudaMallocHost((void**) &I.h_dw, (size_t) KM * sizeof(float)));
    I.dec_best.assign((size_t) NE, -1);
    I.dec_where.assign((size_t) NE, -1);
    I.dec_bw.assign((size_t) NE, 0.0f);
    I.meta_bytes = (size_t) 2 * KM * sizeof(void*) + (size_t) 2 * KM * sizeof(float) + (size_t) 2 * KM * sizeof(int32_t);
    NCK(cudaMallocHost((void**) &I.hp_meta, (size_t) moe_i * I.meta_bytes));
    if (!I.alloc((void**) &I.d_meta, I.meta_bytes, err)) return false;

    vram_mark("MTP + spec + ring + prompt buffers");
    // ---- the K/V cache's base pages, then the expert cache: what is left of the card ----
    I.kd = KD;
    I.n_kv_heads = g_.n_head_kv;
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Attention) I.kv_lists.push_back(&b.kv);
    if (I.mtp.on) I.kv_lists.push_back(&I.mtp.kv);
    {
        const int64_t base = std::min<int64_t>(std::max<int64_t>(opt.kv_base, 1), max_ctx_);
        I.kv_base_pages = (base + (1ll << I.kv_shift) - 1) >> I.kv_shift;
        if (((max_ctx_ + (1ll << I.kv_shift) - 1) >> I.kv_shift) > strata::kernels::NEMO_KV_MAX_PAGES) {
            err = "--max-context " + std::to_string(max_ctx_) + " needs more than " +
                  std::to_string(strata::kernels::NEMO_KV_MAX_PAGES) + " K/V pages";
            return false;
        }
        for (int64_t i = 0; i < I.kv_base_pages; ++i)
            if (!I.kv_add_page()) {
                err = "cudaMalloc of the base K/V pages failed";
                return false;
            }
    }
    size_t free_b = 0, total_b = 0;
    NCK(cudaMemGetInfo(&free_b, &total_b));
    const int64_t reserve = opt.vram_reserve_mib * (1ll << 20);
    I.reserve_b = reserve;
    int64_t per = opt.expert_slots;
    uint64_t all_strides = 0;
    for (const ExpertLayer& X : I.ex) all_strides += X.stride;
    int64_t avail = (int64_t) free_b - reserve;
    if (opt.vram_limit_mib > 0) {
        // the limit covers the process: what it allocated since the start, and its CUDA context (~0.5 GiB, measured
        // on CUDA 13 with this engine's kernels loaded)
        const int64_t ctx = 512ll << 20;
        const int64_t mine = (int64_t) free0 - (int64_t) free_b + ctx;
        avail = std::min<int64_t>(avail, opt.vram_limit_mib * (1ll << 20) - mine);
    }
    if (per <= 0) per = avail / (int64_t) all_strides;
    per = std::min<int64_t>(per, NE);
    if (per < K) {
        err = "only " + std::to_string(per) + " expert slots per layer fit (" + std::to_string(std::max<int64_t>(0, avail) >> 20) +
              " MiB left for them); at least " + std::to_string(K) + " are needed";
        return false;
    }
    per = per / Impl::BS * Impl::BS;
    I.slots = per;
    I.real_slots = per;
    I.slots_max = per;
    for (ExpertLayer& X : I.ex) {
        for (int64_t bnd = 0; bnd < per / Impl::BS; ++bnd) {
            uint8_t* bp = nullptr;
            if (cudaMalloc(&bp, (size_t) Impl::BS * X.stride) != cudaSuccess) {
                err = "cudaMalloc of the expert cache failed";
                return false;
            }
            X.bands.push_back(bp);
        }
        X.slot_of.assign((size_t) NE, -1);
        X.expert_in.assign((size_t) per, -1);
        X.used.assign((size_t) per, 0);
        X.score.assign((size_t) NE, 0.0f);
        X.score_at.assign((size_t) NE, 0);
        I.pf.emplace_back((size_t) per, 0);
        I.fill_tok.emplace_back((size_t) per, UINT64_MAX);
    }
    // the chunk scratch's spare tails (rows LC .. C of the big buffers) as loan bands
    if (C > Impl::LC && I.wb) {
        const int64_t LC = Impl::LC, R = I.tc_rows;
        auto tail = [&](void* b, int64_t rows, uint64_t row_bytes) {
            if (b && rows > LC) I.loan_free.emplace_back((uint8_t*) b + (size_t) LC * row_bytes, (uint64_t) (rows - LC) * row_bytes);
        };
        uint64_t widest = 0;
        for (const Block& b : I.blocks)
            for (const Mat* mt : {&b.in, &b.out, &b.q, &b.k, &b.v, &b.o, &b.fc1, &b.fc2, &b.sh_up, &b.sh_dn})
                if (mt->d) widest = std::max<uint64_t>(widest, (uint64_t) mt->cols);
        tail(I.x, C, E * 4); tail(I.xn, C, E * 4); tail(I.q, C, QD * 4); tail(I.o, C, QD * 4);
        tail(I.lat, C, L * 4); tail(I.routed, C, L * 4); tail(I.routed_d, C, L * 8); tail(I.logits, C, NE * 4);
        tail(I.xb, C, widest * 2);
        tail(I.eg_h, R, FF * 4); tail(I.eg_hb, R, FF * 2); tail(I.eg_out, R, L * 4); tail(I.eg_in, R, L * 4);
        tail(I.eg_xb, R, L * 2);
        tail(I.parts_bf, C, K * L * 2);
        if (!std::getenv("STRATA_NEMO_NO_LOAN")) I.carve_loans();
    }
    I.lend();
    stats_.slots_per_layer = I.slots;
    stats_.expert_cache_gib = (double) I.slots * all_strides / 1073741824.0;
    vram_mark("K/V base pages + expert cache");
    reset();
    return true;
}

std::string NemoModel::profile_report(bool clear) {
    Impl& I = *impl_;
    if (std::getenv("STRATA_NEMO_HOSTPROF") && I.spec_ms[3] > 0) {
        std::fprintf(stderr, "spec per window: draft %.2f ms, verify %.2f ms, rest %.2f ms (%g windows)\n",
                     I.spec_ms[0] / I.spec_ms[3], I.spec_ms[1] / I.spec_ms[3], I.spec_ms[2] / I.spec_ms[3], I.spec_ms[3]);
        if (clear) for (double& v : I.spec_ms) v = 0;
    }
    if (I.hostprof && I.plan_n > 0) {
        std::fprintf(stderr, "host per prompt MoE layer: plan %.2f ms (%llu layers)\n", I.plan_ms / I.plan_n,
                     (unsigned long long) I.plan_n);
        if (clear) { I.plan_ms = 0; I.plan_n = 0; }
    }
    if (std::getenv("STRATA_NEMO_HOSTPROF") && I.hms[3] > 0) {
        std::fprintf(stderr, "host per decode MoE layer: book %.1f us, launch %.1f us (%g layers)\n", I.hms[0] / I.hms[3],
                     I.hms[1] / I.hms[3], I.hms[3]);
        if (clear) for (double& v : I.hms) v = 0;
    }
    if (!I.prof) return std::string();
    static const char* names[8] = {"mamba", "attention", "moe dense+router", "moe ids sync", "experts (compute)",
                                   "moe fc2+shared", "head", "expert copies (prompt)"};
    double tot = 0;
    for (int i = 0; i < 8; ++i) tot += I.prof_ms[i];
    char buf[256];
    std::string r = "strata-nemotron profile (" + std::to_string(I.prof_tokens) + " tokens):";
    for (int i = 0; i < 8; ++i) {
        std::snprintf(buf, sizeof buf, " %s %.1f ms (%.0f%%);", names[i], I.prof_ms[i], tot > 0 ? 100 * I.prof_ms[i] / tot : 0.0);
        r += buf;
    }
    if (clear) {
        for (double& v : I.prof_ms) v = 0;
        I.prof_tokens = 0;
    }
    return r;
}

void NemoModel::reset() {
    Impl& I = *impl_;
    I.kv_shrink();
    stats_.slots_per_layer = I.slots;
    stats_.kv_cells = I.kv_pages << I.kv_shift;
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            cudaMemsetAsync(b.state, 0, sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state, I.st);
            cudaMemsetAsync(b.conv, 0, sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1), I.st);
        }
    cudaStreamSynchronize(I.st);
    pos_ = 0;
    ++kv_gen_;        // the K/V cells of the old sequence are overwritten from here on
    snap_pos_ = -1;   // a new sequence overwrites the K/V cells a snapshot relied on
    I.mtp.pending = false;
    I.mtp.consumed_tok = -1;
}

void NemoModel::snapshot() {
    Impl& I = *impl_;
    size_t i = 0;
    const size_t sb = sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state;
    const size_t cb = sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1);
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            cudaMemcpyAsync(I.snap_state[i], b.state, sb, cudaMemcpyDeviceToDevice, I.st);
            cudaMemcpyAsync(I.snap_conv[i], b.conv, cb, cudaMemcpyDeviceToDevice, I.st);
            ++i;
        }
    if (I.mtp.on) {
        cudaMemcpyAsync(I.mtp.snap_h, I.mtp.h, sizeof(float) * g_.n_embd, cudaMemcpyDeviceToDevice, I.st);
        I.mtp.snap_pending = I.mtp.pending;
        I.mtp.snap_pending_pos = I.mtp.pending_pos;
    }
    cudaStreamSynchronize(I.st);
    snap_pos_ = pos_;
    snap_logits_.resize((size_t) g_.n_vocab);
    cudaMemcpy(snap_logits_.data(), I.head_out, sizeof(float) * g_.n_vocab, cudaMemcpyDeviceToHost);
}

bool NemoModel::restore() {
    Impl& I = *impl_;
    if (snap_pos_ < 0) return false;
    size_t i = 0;
    const size_t sb = sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state;
    const size_t cb = sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1);
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            cudaMemcpyAsync(b.state, I.snap_state[i], sb, cudaMemcpyDeviceToDevice, I.st);
            cudaMemcpyAsync(b.conv, I.snap_conv[i], cb, cudaMemcpyDeviceToDevice, I.st);
            ++i;
        }
    if (I.mtp.on) {
        cudaMemcpyAsync(I.mtp.h, I.mtp.snap_h, sizeof(float) * g_.n_embd, cudaMemcpyDeviceToDevice, I.st);
        I.mtp.pending = I.mtp.snap_pending;
        I.mtp.pending_pos = I.mtp.snap_pending_pos;
        I.mtp.consumed_tok = -1;
    }
    cudaStreamSynchronize(I.st);
    pos_ = snap_pos_;
    return true;
}

// ---- conversation cache: checkpoints and K/V cells (nemo_conv_cache.cpp holds the host side) ----

size_t NemoModel::ckpt_bytes() const {
    const size_t per = sizeof(float) * (size_t) (g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state +
                                                g_.ssm_conv_dim() * (g_.ssm_d_conv - 1));
    return per * (size_t) g_.count(NemotronBlock::Mamba);
}

bool NemoModel::checkpoint_save(NemoCkpt& c, std::string& err) {
    Impl& I = *impl_;
    if (!c.mamba) { err = "checkpoint without a buffer"; return false; }
    const size_t sb = sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state;
    const size_t cb = sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1);
    uint8_t* d = c.mamba;
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            NCK(cudaMemcpyAsync(d, b.state, sb, cudaMemcpyDeviceToHost, I.st));
            NCK(cudaMemcpyAsync(d + sb, b.conv, cb, cudaMemcpyDeviceToHost, I.st));
            d += sb + cb;
        }
    c.mtp_h.clear();
    c.mtp_pending = false;
    c.mtp_pending_pos = -1;
    c.mtp_consumed = -1;
    if (I.mtp.on) {
        c.mtp_h.resize((size_t) g_.n_embd);
        NCK(cudaMemcpyAsync(c.mtp_h.data(), I.mtp.h, sizeof(float) * g_.n_embd, cudaMemcpyDeviceToHost, I.st));
        c.mtp_pending = I.mtp.pending;
        c.mtp_pending_pos = I.mtp.pending_pos;
        c.mtp_consumed = I.mtp.consumed_tok;
    }
    c.logits.resize((size_t) g_.n_vocab);
    NCK(cudaMemcpyAsync(c.logits.data(), I.head_out, sizeof(float) * g_.n_vocab, cudaMemcpyDeviceToHost, I.st));
    NCK(cudaStreamSynchronize(I.st));
    c.pos = pos_;
    c.kv_gen = kv_gen_;
    return true;
}

bool NemoModel::checkpoint_restore(const NemoCkpt& c, std::string& err) {
    Impl& I = *impl_;
    if (!c.mamba || c.pos < 0) { err = "empty checkpoint"; return false; }
    const size_t sb = sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state;
    const size_t cb = sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1);
    NCK(cudaStreamSynchronize(I.st));
    const uint8_t* d = c.mamba;
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            NCK(cudaMemcpyAsync(b.state, d, sb, cudaMemcpyHostToDevice, I.st));
            NCK(cudaMemcpyAsync(b.conv, d + sb, cb, cudaMemcpyHostToDevice, I.st));
            d += sb + cb;
        }
    if (I.mtp.on) {
        if (c.mtp_h.size() == (size_t) g_.n_embd)
            NCK(cudaMemcpyAsync(I.mtp.h, c.mtp_h.data(), sizeof(float) * g_.n_embd, cudaMemcpyHostToDevice, I.st));
        I.mtp.pending = c.mtp_pending;
        I.mtp.pending_pos = c.mtp_pending_pos;
        I.mtp.consumed_tok = c.mtp_consumed;
    }
    if (c.logits.size() == (size_t) g_.n_vocab)
        NCK(cudaMemcpyAsync(I.head_out, c.logits.data(), sizeof(float) * g_.n_vocab, cudaMemcpyHostToDevice, I.st));
    NCK(cudaStreamSynchronize(I.st));
    pos_ = c.pos;
    return true;
}

uint64_t NemoModel::kv_cell_bytes() const {
    const Impl& I = *impl_;
    return (uint64_t) I.kv_lists.size() * (uint64_t) (I.kd * (1 + 2) + I.n_kv_heads * 4);
}

bool NemoModel::kv_save(int64_t cells, uint8_t* buf, std::string& err) {
    Impl& I = *impl_;
    if (cells < 0 || cells > (I.kv_pages << I.kv_shift)) { err = "kv_save: more cells than the cache holds"; return false; }
    NCK(cudaStreamSynchronize(I.st));
    NCK(cudaStreamSynchronize(I.cst));
    const int64_t kd = I.kd, nh = I.n_kv_heads, pc = 1ll << I.kv_shift;
    uint8_t* d = buf;
    for (KvPages* l : I.kv_lists) {
        uint8_t* kd_ = d;
        uint8_t* ksd = kd_ + (size_t) cells * kd;
        uint8_t* vd = ksd + (size_t) cells * nh * sizeof(float);
        for (int64_t p = 0; p * pc < cells; ++p) {
            const int64_t n = std::min<int64_t>(pc, cells - p * pc);
            NCK(cudaMemcpyAsync(kd_ + (size_t) p * pc * kd, l->k[(size_t) p], (size_t) n * kd, cudaMemcpyDeviceToHost, I.st));
            NCK(cudaMemcpyAsync(ksd + (size_t) p * pc * nh * 4, l->ks[(size_t) p], (size_t) n * nh * 4, cudaMemcpyDeviceToHost, I.st));
            NCK(cudaMemcpyAsync(vd + (size_t) p * pc * kd * 2, l->v[(size_t) p], (size_t) n * kd * 2, cudaMemcpyDeviceToHost, I.st));
        }
        d = vd + (size_t) cells * kd * 2;
    }
    NCK(cudaStreamSynchronize(I.st));
    return true;
}

bool NemoModel::kv_restore(int64_t cells, const uint8_t* buf, std::string& err) {
    Impl& I = *impl_;
    if (cells < 0 || cells > max_ctx_) { err = "kv_restore: more cells than the context"; return false; }
    if (!I.kv_grow(std::min<int64_t>(cells + 16, max_ctx_), err)) return false;
    stats_.slots_per_layer = I.slots;
    stats_.kv_cells = I.kv_pages << I.kv_shift;
    NCK(cudaStreamSynchronize(I.st));
    const int64_t kd = I.kd, nh = I.n_kv_heads, pc = 1ll << I.kv_shift;
    const uint8_t* d = buf;
    for (KvPages* l : I.kv_lists) {
        const uint8_t* kd_ = d;
        const uint8_t* ksd = kd_ + (size_t) cells * kd;
        const uint8_t* vd = ksd + (size_t) cells * nh * sizeof(float);
        for (int64_t p = 0; p * pc < cells; ++p) {
            const int64_t n = std::min<int64_t>(pc, cells - p * pc);
            NCK(cudaMemcpyAsync(l->k[(size_t) p], kd_ + (size_t) p * pc * kd, (size_t) n * kd, cudaMemcpyHostToDevice, I.st));
            NCK(cudaMemcpyAsync(l->ks[(size_t) p], ksd + (size_t) p * pc * nh * 4, (size_t) n * nh * 4, cudaMemcpyHostToDevice, I.st));
            NCK(cudaMemcpyAsync(l->v[(size_t) p], vd + (size_t) p * pc * kd * 2, (size_t) n * kd * 2, cudaMemcpyHostToDevice, I.st));
        }
        d = vd + (size_t) cells * kd * 2;
    }
    NCK(cudaStreamSynchronize(I.st));
    return true;
}

bool NemoModel::forward_all(const int32_t* tokens, int64_t n, float* all, std::string& err) {
    impl_->all_logits = all;
    const bool ok = forward(tokens, n, nullptr, err);
    impl_->all_logits = nullptr;
    return ok;
}

bool NemoModel::forward(const int32_t* tokens, int64_t n, float* logits_out, std::string& err) {
    Impl& I = *impl_;
    const int64_t E = g_.n_embd, DI = g_.ssm_d_inner, CD = g_.ssm_conv_dim(), IN = g_.ssm_in_dim();
    const int64_t QD = g_.n_head * g_.head_dim, KD = g_.n_head_kv * g_.head_dim;
    const int64_t NE = g_.n_expert, K = g_.n_expert_used, L = g_.moe_latent, FF = g_.n_ff_exp, SH = g_.n_ff_shexp;
    const float eps = (float) g_.rms_eps;
    strata::kernels::MambaShapes ms;
    ms.H = g_.ssm_n_head; ms.P = g_.ssm_head_dim(); ms.N = g_.ssm_d_state; ms.G = g_.ssm_n_group; ms.d_conv = g_.ssm_d_conv;
    strata::kernels::NemoAttnShapes as;
    as.n_head = g_.n_head; as.n_head_kv = g_.n_head_kv; as.head_dim = g_.head_dim;
    cudaStream_t st = I.st;
    std::vector<int32_t> h_ids;
    std::vector<float> h_w;

    for (int64_t done = 0; done < n;) {
        const int64_t T = std::min<int64_t>(I.C, n - done);
        if (T > Impl::LC) I.reclaim();   // (the loan bands live in the rows this chunk needs)
        if (T > 1 && !I.verify) {
            NCK(cudaStreamSynchronize(I.cst));   // decode's prefetches land first
            I.pq.clear();                        // (the last chunk's sequence is consumed)
            I.pq_copied = 0;
        } else {
            ++I.tok;
        }
        if (pos_ + T > max_ctx_) {
            err = "the context is full (" + std::to_string(max_ctx_) + " tokens)";
            return false;
        }
        // (+16: the MTP drafts write a few cells past the window)
        if (!I.kv_grow(std::min<int64_t>(pos_ + T + 16, max_ctx_), err)) return false;
        stats_.slots_per_layer = I.slots;
        stats_.kv_cells = I.kv_pages << I.kv_shift;
        // the token rows, from the mapping through the pinned stage
        for (int64_t t = 0; t < T; ++t) {
            const int32_t id = tokens[done + t];
            if (id < 0 || id >= g_.n_vocab) {
                err = "token id " + std::to_string(id) + " is outside the vocabulary";
                return false;
            }
            std::memcpy(I.embd_stage + t * E, I.embd + (size_t) id * E, E * sizeof(uint16_t));
        }
        uint16_t* d_emb = (uint16_t*) I.q;   // q has room for C * 4096 floats: the BF16 rows fit in its first half
        NCK(cudaMemcpyAsync(d_emb, I.embd_stage, (size_t) T * E * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
        strata::kernels::nemo_bf16_rows_to_f32(d_emb, I.x, T, E, st);
        I.img_overlay(I.x, pos_, T, E, st);

        I.mark(-1);
        for (Block& b : I.blocks) {
            strata::kernels::nemo_rms_norm_rows(I.x, b.norm, I.xn, T, E, eps, st);
            if (b.kind == NemotronBlock::Mamba) {
                // a verify window keeps its in_proj rows for the replay (spec_step)
                float* zx = I.verify ? I.spec_zx + (size_t) b.mamba * strata::kernels::NEMO_GEMV_MAX_M * IN : I.zx;
                for (int64_t s0 = 0; s0 < T; s0 += I.SC) {
                    const int64_t n1 = std::min<int64_t>(I.SC, T - s0);
                    I.gemm(b.in, I.xn + s0 * E, E, zx, IN, n1, false, false);
                    if (n1 == 1) {
                        strata::kernels::mamba_conv_step(b.conv, zx + DI, b.conv_w, b.conv_b, I.xbc, CD, ms.d_conv, st);
                        strata::kernels::mamba_ssm_step(b.state, I.xbc, zx + DI + CD, b.dt_b, b.A, b.D, I.y, ms, st);
                        strata::kernels::mamba_gated_norm(I.y, zx, b.ssm_norm, I.y, DI, ms.G, eps, st);
                    } else {
                        // STRATA_NEMO_MAMBA_SWAP: the verify window leaves the new state in spec_state / spec_conv
                        float* so = I.verify_swap ? I.spec_state[(size_t) b.mamba] : nullptr;
                        float* co = I.verify_swap ? I.spec_conv[(size_t) b.mamba] : nullptr;
                        strata::kernels::mamba_conv_prefill(b.conv, zx, IN, b.conv_w, b.conv_b, I.xbc, n1, ms, st, co);
                        strata::kernels::mamba_ssm_scan(b.state, I.xbc, zx, IN, b.dt_b, b.A, b.D, I.y, n1, ms, st, so);
                        strata::kernels::mamba_gated_norm_rows(I.y, zx, IN, b.ssm_norm, I.y, n1, DI, ms.G, eps, st);
                    }
                    I.gemm(b.out, I.y, DI, I.x + s0 * E, E, n1, false, true);   // x += out_proj(y)
                }
                I.mark(0);
            } else if (b.kind == NemotronBlock::Attention) {
                I.gemm(b.q, I.xn, E, I.q, QD, T, false, false);
                I.gemm(b.k, I.xn, E, I.k, KD, T, false, false);
                I.gemm(b.v, I.xn, E, I.v, KD, T, false, false);
                {
                    const strata::kernels::NemoKV kv = I.kv_of(b.kv);
                    strata::kernels::nemo_kv_append(kv, I.k, I.v, pos_, T, as, st);
                    strata::kernels::nemo_attn(I.q, kv, I.o, pos_, T, as, I.attn_scratch, st);
                }
                I.gemm(b.o, I.o, QD, I.x, E, T, false, true);     // x += o_proj(o)
                I.mark(1);
            } else {
                ExpertLayer& X = I.ex[(size_t) b.moe];
                const bool dec = T == 1 || I.verify;   // the decode path: one token, or a verify window
                const bool pskip = !dec && I.pskip_w > 0 && T <= I.pskip_max;
                I.gemm(b.gate, I.xn, E, I.logits, NE, T, false, false);
                strata::kernels::nemo_router(I.logits, b.bias, I.ids, I.w, T, NE, K, (float) g_.expert_weights_scale,
                                             g_.expert_weights_norm, st);
                if (I.topk_cut > 0 && I.topk_cut < K)   // STRATA_NEMO_TOPK_CUT=R (a quality probe): ranks >= R weigh 0
                    NCK(cudaMemset2DAsync(I.w + I.topk_cut, (size_t) K * sizeof(float), 0,
                                          (size_t) (K - I.topk_cut) * sizeof(float), (size_t) T, st));
                I.gemm(b.fc1, I.xn, E, I.lat, L, T, false, false);
                if (!dec) {
                    // the prompt: the shared expert now (it needs only xn), so the GPU has it to do while the host
                    // reads the routing back and plans the experts
                    for (int64_t s0 = 0; s0 < T; s0 += I.SC) {
                        const int64_t n1 = std::min<int64_t>(I.SC, T - s0);
                        I.gemm(b.sh_up, I.xn + s0 * E, E, I.sh_h, SH, n1, true, false);
                        I.gemm(b.sh_dn, I.sh_h, SH, I.x + s0 * E, E, n1, false, true);
                    }
                }
                I.mark(2);
                // decode: the next MoE layer's router on the residual as it is now (its own norm), for the prefetch
                const bool pred = dec && I.prefetch && b.moe + 1 < (int) I.moe_block.size();
                if (pred) {
                    const Block& nb = I.blocks[(size_t) I.moe_block[(size_t) b.moe + 1]];
                    strata::kernels::nemo_rms_norm_rows(I.x, nb.norm, I.pxn, T, E, eps, st);
                    I.gemm(nb.gate, I.pxn, E, I.plogits, NE, T, false, false);
                    strata::kernels::nemo_router(I.plogits, nb.bias, I.pids, I.pw, T, NE, K,
                                                 (float) g_.expert_weights_scale, g_.expert_weights_norm, st);
                    NCK(cudaMemcpyAsync(I.h_pids, I.pids, (size_t) T * K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                }
                const bool ppred = pskip && I.pskip_pred && I.wb && b.moe + 1 < (int) I.moe_block.size();
                if (ppred) {
                    const Block& nb = I.blocks[(size_t) I.moe_block[(size_t) b.moe + 1]];
                    strata::kernels::nemo_rms_norm_rows(I.x, nb.norm, I.q, T, E, eps, st);   // (q is free here)
                    I.gemm(nb.gate, I.q, E, I.pp_logits, NE, T, false, false);
                    strata::kernels::nemo_router(I.pp_logits, nb.bias, I.pp_ids, I.pp_w, T, NE, K,
                                                 (float) g_.expert_weights_scale, g_.expert_weights_norm, st);
                    I.h_pp_ids.resize((size_t) T * K);
                    I.h_pp_w.resize((size_t) T * K);
                    NCK(cudaMemcpyAsync(I.h_pp_ids.data(), I.pp_ids, (size_t) T * K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                    NCK(cudaMemcpyAsync(I.h_pp_w.data(), I.pp_w, (size_t) T * K * sizeof(float), cudaMemcpyDeviceToHost, st));
                }
                h_ids.resize((size_t) T * K);
                bool sh_done = false;   // STRATA_NEMO_SHARED_EARLY: the shared expert already ran (below)
                if (dec) {   // (pinned: a pageable read-back costs ~100 us of staging per layer)
                    NCK(cudaMemcpyAsync(I.h_dids, I.ids, (size_t) T * K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                    if (I.skip_w > 0)
                        NCK(cudaMemcpyAsync(I.h_dw, I.w, (size_t) T * K * sizeof(float), cudaMemcpyDeviceToHost, st));
                    if (I.shared_early) {
                        // the read-back is queued; the shared expert (it needs only xn) goes behind it and the host
                        // waits for the read-back alone, so the GPU has work while the host books.  (x += shared now,
                        // x += fc2 later: the two adds swap places)
                        if (!I.ids_ev) NCK(cudaEventCreateWithFlags(&I.ids_ev, cudaEventDisableTiming));
                        NCK(cudaEventRecord(I.ids_ev, st));
                        for (int64_t s0 = 0; s0 < T; s0 += I.SC) {
                            const int64_t n1 = std::min<int64_t>(I.SC, T - s0);
                            I.gemm(b.sh_up, I.xn + s0 * E, E, I.sh_h, SH, n1, true, false);
                            I.gemm(b.sh_dn, I.sh_h, SH, I.x + s0 * E, E, n1, false, true);
                        }
                        sh_done = true;
                        NCK(cudaEventSynchronize(I.ids_ev));
                    } else {
                        NCK(cudaStreamSynchronize(st));
                    }
                    I.ht[0] = std::chrono::steady_clock::now();
                    std::memcpy(h_ids.data(), I.h_dids, (size_t) T * K * sizeof(int32_t));
                } else {
                    NCK(cudaMemcpyAsync(h_ids.data(), I.ids, (size_t) T * K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                    if (pskip) {
                        h_w.resize((size_t) T * K);
                        NCK(cudaMemcpyAsync(h_w.data(), I.w, (size_t) T * K * sizeof(float), cudaMemcpyDeviceToHost, st));
                    }
                    NCK(cudaStreamSynchronize(st));
                }
                I.mark(3);
                const auto plan_t0 = std::chrono::steady_clock::now();
                if (I.route_trace) {
                    // STRATA_NEMO_ROUTE_TRACE=file: per MoE pass (int32 layer, int32 T, T*K int32 expert ids)
                    const int32_t hdr[2] = {(int32_t) b.moe, (int32_t) T};
                    fwrite(hdr, sizeof(hdr), 1, I.route_trace);
                    fwrite(h_ids.data(), sizeof(int32_t), (size_t) T * K, I.route_trace);
                }
                // a slot for each expert this pass needs: a hit, or the least recently used slot.  Decode (`hold`)
                // launches all 22 experts at once, so a slot taken in this pass is not a victim there; the prompt
                // computes expert after expert on the same stream, so a copy into a slot this pass already used is
                // ordered after that use and is safe - which is what lets a pass touch more experts than there are slots
                const uint64_t pass0 = I.clock + 1;
                auto ensure = [&](int32_t e, bool hold) -> const uint8_t* {
                    ++I.clock;
                    int32_t s = X.slot_of[(size_t) e];
                    if (s >= 0) {
                        ++stats_.expert_hits;
                    } else {
                        ++stats_.expert_misses;
                        int32_t victim = 0;
                        uint64_t oldest = UINT64_MAX;
                        for (int32_t c = 0; c < (int32_t) I.slots; ++c)
                            if (X.used[(size_t) c] < oldest && !(hold && X.used[(size_t) c] >= pass0)) {
                                oldest = X.used[(size_t) c];
                                victim = c;
                            }
                        if (X.expert_in[(size_t) victim] >= 0) X.slot_of[(size_t) X.expert_in[(size_t) victim]] = -1;
                        X.expert_in[(size_t) victim] = e;
                        X.slot_of[(size_t) e] = victim;
                        s = victim;
                        uint8_t* dst = I.slot_ptr(X, s);
                        cudaMemcpyAsync(dst, X.up + (size_t) e * X.ub, X.ub, cudaMemcpyHostToDevice, st);
                        cudaMemcpyAsync(dst + X.ub, X.dn + (size_t) e * X.db, X.db,
                                        cudaMemcpyHostToDevice, st);
                    }
                    X.used[(size_t) s] = I.clock;
                    return I.slot_ptr(X, s);
                };
                if (dec) {
                    // entries (t, j) = row t's rank-j expert.  The distinct experts first: hits (stamped into this
                    // pass, so no miss of the pass evicts them), then the misses - a copy on the copy stream into the
                    // least recently used slot, or the zero expert when every entry of the expert ranks >= tail_skip
                    // (Strata's rule for a verify window).  The compute of the ready entries starts at once; the
                    // rest waits for the layer's copies (and for this step's prefetches into the layer).
                    const int64_t KT = T * K, KM = (int64_t) strata::kernels::NEMO_GEMV_MAX_M * K;
                    uint8_t* meta = I.hp_meta + (size_t) b.moe * I.meta_bytes;
                    const uint8_t** mptr = (const uint8_t**) meta;
                    float* ms2 = (float*) (meta + (size_t) 2 * KM * sizeof(void*));
                    int32_t* moidx = (int32_t*) (ms2 + 2 * KM);
                    int32_t* mxidx = moidx + KM;
                    std::vector<uint8_t>& PF = I.pf[(size_t) b.moe];
                    std::vector<uint64_t>& FT = I.fill_tok[(size_t) b.moe];
                    ++I.clock;
                    const uint64_t p0 = I.clock;
                    // per distinct expert: its best rank, then its slot (or -2: skipped) and readiness
                    std::vector<int32_t>& best = I.dec_best;
                    std::vector<int32_t>& where = I.dec_where;
                    std::vector<int32_t> uniq;
                    std::vector<float>& bw = I.dec_bw;
                    for (int64_t i = 0; i < KT; ++i) {
                        const int32_t e = h_ids[(size_t) i], j = (int32_t) (i % K);
                        if (best[(size_t) e] < 0) { uniq.push_back(e); bw[(size_t) e] = 0.0f; }
                        if (best[(size_t) e] < 0 || j < best[(size_t) e]) best[(size_t) e] = j;
                        if (I.skip_w > 0) bw[(size_t) e] = std::max(bw[(size_t) e], I.h_dw[i]);
                    }
                    if (I.evict_lfu) {
                        // decayed use counts: every entry of the pass is a use, then the layer's clock moves by T
                        const float ld = (float) std::log(I.lfu_decay);
                        for (int64_t i = 0; i < KT; ++i) {
                            const int32_t e = h_ids[(size_t) i];
                            // (STRATA_NEMO_LFU_FAST: a second entry of the pass has age 0, and exp(0) is exactly 1)
                            if (I.lfu_fast && X.score_at[(size_t) e] == X.ticks) X.score[(size_t) e] += 1.0f;
                            else X.score[(size_t) e] = X.score[(size_t) e] * std::exp(ld * (float) (X.ticks - X.score_at[(size_t) e])) + 1.0f;
                            X.score_at[(size_t) e] = X.ticks;
                        }
                        X.ticks += (uint64_t) T;
                    }
                    std::vector<uint8_t> ready_of(uniq.size(), 1);
                    for (size_t u = 0; u < uniq.size(); ++u) {
                        const int32_t e = uniq[u], sl = X.slot_of[(size_t) e];
                        where[(size_t) e] = (int32_t) u;
                        if (sl < 0) continue;
                        ++stats_.expert_hits;
                        if (PF[(size_t) sl]) { ++stats_.prefetch_used; PF[(size_t) sl] = 0; }
                        X.used[(size_t) sl] = p0;
                        ready_of[u] = FT[(size_t) sl] != I.tok;
                    }
                    // a victim: the oldest slot not used in this pass
                    auto victim = [&](ExpertLayer& XL, std::vector<uint8_t>& PFL, uint64_t since) -> int32_t {
                        int32_t v = -1;
                        if (I.evict_lfu) {
                            // the least decayed use count among the slots not used in this pass (empty slots first)
                            const float ld = (float) std::log(I.lfu_decay);
                            float lo = INFINITY;
                            // STRATA_NEMO_LFU_FAST: every slot's decayed score once per ticks value (the scores and
                            // ticks only move together, in the layer's own pass); a slot that changed hands inside
                            // the pass is used >= since, so the search never reads its stale entry
                            const bool fast = I.lfu_fast;
                            if (fast && XL.dsc_ticks != XL.ticks) {
                                XL.dsc.resize((size_t) I.slots);
                                for (int32_t c = 0; c < (int32_t) I.slots; ++c) {
                                    const int32_t e = XL.expert_in[(size_t) c];
                                    XL.dsc[(size_t) c] = e < 0 ? -1.0f
                                                               : XL.score[(size_t) e] *
                                                                     std::exp(ld * (float) (XL.ticks - XL.score_at[(size_t) e]));
                                }
                                XL.dsc_ticks = XL.ticks;
                            }
                            int32_t v_old = -1;
                            if (!fast || I.lfu_check) {
                                float lo_old = INFINITY;
                                for (int32_t c = 0; c < (int32_t) I.slots; ++c) {
                                    if (XL.used[(size_t) c] >= since) continue;
                                    const int32_t e = XL.expert_in[(size_t) c];
                                    const float sc = e < 0 ? -1.0f
                                                           : XL.score[(size_t) e] *
                                                                 std::exp(ld * (float) (XL.ticks - XL.score_at[(size_t) e]));
                                    if (sc < lo_old) { lo_old = sc; v_old = c; }
                                }
                                lo = lo_old;
                                v = v_old;
                            }
                            if (fast) {
                                float lo_f = INFINITY;
                                int32_t v_f = -1;
                                for (int32_t c = 0; c < (int32_t) I.slots; ++c) {
                                    if (XL.used[(size_t) c] >= since) continue;
                                    const float sc = XL.dsc[(size_t) c];
                                    if (sc < lo_f) { lo_f = sc; v_f = c; }
                                }
                                if (I.lfu_check) {
                                    ++I.lfu_checked;
                                    if (v_f != v_old) ++I.lfu_mismatch;
                                }
                                v = v_f;
                            }
                        } else {
                            uint64_t oldest = UINT64_MAX;
                            for (int32_t c = 0; c < (int32_t) I.slots; ++c)
                                if (XL.used[(size_t) c] < oldest && XL.used[(size_t) c] < since) {
                                    oldest = XL.used[(size_t) c];
                                    v = c;
                                }
                        }
                        if (v >= 0) {
                            if (PFL[(size_t) v]) { ++stats_.prefetch_wasted; PFL[(size_t) v] = 0; }
                            if (XL.expert_in[(size_t) v] >= 0) XL.slot_of[(size_t) XL.expert_in[(size_t) v]] = -1;
                        }
                        return v;
                    };
                    std::vector<uint8_t> skip_of(uniq.size(), 0);
                    // a pass with more distinct misses than free slots (a verify window on a small cache): the extra
                    // ones are copied into half (layer parity) of the prompt's ring - idle in decode - so the window
                    // stays exact; the half is reused two layers on, after the compute that read it (ovf_ev)
                    std::vector<const uint8_t*> ovf(uniq.size(), nullptr);
                    const int ohalf = (int) (b.moe & 1);
                    const int64_t ocap = I.ring ? I.ring_n / 2 : 0;
                    int64_t on = 0;
                    X.dsc_ticks = UINT64_MAX;   // (LFU_FAST: this pass's victim searches share one decay table)
                    for (size_t u = 0; u < uniq.size(); ++u) {
                        const int32_t e = uniq[u];
                        if (X.slot_of[(size_t) e] >= 0) continue;
                        // skipped: every entry ranks >= tail_skip, or (STRATA_NEMO_SKIP_W) weighs < skip_w
                        if ((I.tail_skip > 0 && best[(size_t) e] >= I.tail_skip) ||
                            (I.skip_w > 0 && bw[(size_t) e] < I.skip_w)) {
                            skip_of[u] = 1;
                            ++stats_.expert_skips;
                            continue;
                        }
                        const int32_t sl = victim(X, PF, p0);
                        if (sl < 0 && on < ocap) {
                            if (on == 0 && I.ovf_rec[ohalf]) NCK(cudaStreamWaitEvent(I.cst, I.ovf_ev[ohalf], 0));
                            uint8_t* dst = I.ring + (size_t) (ohalf * ocap + on++) * I.blob;
                            cudaMemcpyAsync(dst, X.up + (size_t) e * X.ub, X.ub, cudaMemcpyHostToDevice, I.cst);
                            cudaMemcpyAsync(dst + X.ub, X.dn + (size_t) e * X.db, X.db, cudaMemcpyHostToDevice, I.cst);
                            ovf[u] = dst;
                            ready_of[u] = 0;
                            ++stats_.expert_misses;
                            ++stats_.expert_overflow;
                            continue;
                        }
                        if (sl < 0) {
                            // no slot and no ring left: this one is skipped too (the window is then not exact)
                            skip_of[u] = 1;
                            ++stats_.expert_skips;
                            ++stats_.expert_forced_skips;
                            continue;
                        }
                        ++stats_.expert_misses;
                        X.expert_in[(size_t) sl] = e;
                        X.slot_of[(size_t) e] = sl;
                        X.used[(size_t) sl] = p0;
                        FT[(size_t) sl] = I.tok;
                        ready_of[u] = 0;
                        uint8_t* dst = I.slot_ptr(X, sl);
                        cudaMemcpyAsync(dst, X.up + (size_t) e * X.ub, X.ub, cudaMemcpyHostToDevice, I.cst);
                        cudaMemcpyAsync(dst + X.ub, X.dn + (size_t) e * X.db, X.db, cudaMemcpyHostToDevice, I.cst);
                    }
                    // the list: ready entries from the front, waiting ones from the back
                    int64_t na = 0, nb_ = 0;
                    for (int64_t i = 0; i < KT; ++i) {
                        const int32_t e = h_ids[(size_t) i];
                        const size_t u = (size_t) where[(size_t) e];
                        const bool ready = skip_of[u] || ready_of[u];
                        const int64_t at = ready ? na++ : KT - 1 - nb_++;
                        const uint8_t* bl = skip_of[u] ? I.zero_blob : ovf[u] ? ovf[u] : I.slot_ptr(X, X.slot_of[(size_t) e]);
                        mptr[at] = bl;
                        mptr[KM + at] = bl + X.ub;
                        ms2[at] = skip_of[u] ? 0.0f : X.s_up[(size_t) e];
                        ms2[KM + at] = skip_of[u] ? 0.0f : X.s_dn[(size_t) e];
                        moidx[at] = (int32_t) i;
                        mxidx[at] = (int32_t) (i / K);
                    }
                    for (int32_t e : uniq) best[(size_t) e] = -1;
                    I.ht[1] = std::chrono::steady_clock::now();
                    const uint8_t** dptr = (const uint8_t**) I.d_meta;
                    float* ds2 = (float*) (I.d_meta + (size_t) 2 * KM * sizeof(void*));
                    int32_t* doidx = (int32_t*) (ds2 + 2 * KM);
                    int32_t* dxidx = doidx + KM;
                    NCK(cudaMemcpyAsync(I.d_meta, meta, I.meta_bytes, cudaMemcpyHostToDevice, st));
                    if (na > 0)
                        strata::kernels::nemo_experts_part(X.tu, X.td, dptr, dptr + KM, ds2, ds2 + KM, doidx, dxidx, na,
                                                           I.lat, I.eh, KT, L, FF, st);
                    NCK(cudaEventRecord(I.cev, I.cst));
                    NCK(cudaStreamWaitEvent(st, I.cev, 0));
                    if (nb_ > 0)
                        strata::kernels::nemo_experts_part(X.tu, X.td, dptr + na, dptr + KM + na, ds2 + na,
                                                           ds2 + KM + na, doidx + na, dxidx + na, nb_, I.lat, I.eh, KT,
                                                           L, FF, st);
                    if (T == 1) strata::kernels::nemo_experts_sum(I.w, I.eh, I.routed, K, L, FF, st);
                    else strata::kernels::nemo_weighted_rows(I.eh + (size_t) KT * FF, I.w, I.routed, T, K, L, st);
                    if (on > 0) {   // the ring half is free again once this compute is done
                        if (!I.ovf_ev[ohalf]) NCK(cudaEventCreateWithFlags(&I.ovf_ev[ohalf], cudaEventDisableTiming));
                        NCK(cudaEventRecord(I.ovf_ev[ohalf], st));
                        I.ovf_rec[ohalf] = true;
                    }
                    if (I.zero_stats) {
                        if (I.zs.size() < I.ex.size()) I.zs.resize(I.ex.size(), std::array<uint64_t, 5>{});
                        I.zs_host.resize((size_t) KT * FF);
                        NCK(cudaMemcpyAsync(I.zs_host.data(), I.eh, (size_t) KT * FF * sizeof(float), cudaMemcpyDeviceToHost, st));
                        NCK(cudaStreamSynchronize(st));
                        std::array<uint64_t, 5>& z = I.zs[(size_t) b.moe];
                        for (int64_t i = 0; i < KT; ++i) {
                            if (skip_of[(size_t) where[(size_t) h_ids[(size_t) i]]]) continue;   // a zero expert
                            const float* hrow = I.zs_host.data() + (size_t) i * FF;
                            if (I.zs_dump && I.zs_dumped < 8192) {   // STRATA_NEMO_ZERO_DUMP=file: the first 8192 rows
                                fwrite(hrow, sizeof(float), (size_t) FF, I.zs_dump);
                                ++I.zs_dumped;
                            }
                            ++z[0];
                            z[1] += (uint64_t) FF;
                            for (int64_t j = 0; j < FF; ++j) z[2] += hrow[j] == 0.0f;
                            for (int64_t j0 = 0; j0 < FF; j0 += 16) {
                                bool zero = true;
                                for (int64_t j = j0; j < j0 + 16 && zero; ++j) zero = hrow[j] == 0.0f;
                                z[3] += zero;
                            }
                            for (int64_t j0 = 0; j0 + 64 <= FF; j0 += 64) {
                                bool zero = true;
                                for (int64_t j = j0; j < j0 + 64 && zero; ++j) zero = hrow[j] == 0.0f;
                                z[4] += zero;
                            }
                        }
                    }
                    I.ht[2] = std::chrono::steady_clock::now();
                    I.hms[0] += std::chrono::duration<double, std::micro>(I.ht[1] - I.ht[0]).count();
                    I.hms[1] += std::chrono::duration<double, std::micro>(I.ht[2] - I.ht[1]).count();
                    I.hms[3] += 1;
                    // the prefetch: the next MoE layer's predicted experts (ranks below tail_skip: the rest would be
                    // skipped on a miss anyway) that are not in its cache, after this layer's copies
                    if (pred) {
                        const int nm = b.moe + 1;
                        ExpertLayer& XN = I.ex[(size_t) nm];
                        std::vector<uint8_t>& PFN = I.pf[(size_t) nm];
                        // ranks below STRATA_NEMO_PREFETCH_RANK (default: tail_skip, else all); a verify window
                        // prefetches for its first STRATA_NEMO_PREFETCH_ROWS rows (default 1: the drafts' rows are
                        // the ones that may be refused, and their guesses evict what the next windows need)
                        int64_t pk = I.tail_skip > 0 ? I.tail_skip : K;
                        if (I.prefetch_rank > 0) pk = std::min<int64_t>(pk, I.prefetch_rank);
                        const int64_t pt = T == 1 ? 1 : std::min<int64_t>(T, I.prefetch_rows);
                        ++I.clock;
                        const uint64_t q0 = I.clock;
                        XN.dsc_ticks = UINT64_MAX;
                        for (int64_t j = 0; j < pk; ++j)
                            for (int64_t t = 0; t < pt; ++t) {
                                const int32_t e = I.h_pids[t * K + j];
                                const int32_t sl0 = XN.slot_of[(size_t) e];
                                if (sl0 >= 0) {
                                    XN.used[(size_t) sl0] = std::max(XN.used[(size_t) sl0], q0 - 1);
                                    continue;
                                }
                                const int32_t sl = victim(XN, PFN, q0 - 1);
                                if (sl < 0) break;
                                ++stats_.prefetch_issued;
                                XN.expert_in[(size_t) sl] = e;
                                XN.slot_of[(size_t) e] = sl;
                                XN.used[(size_t) sl] = q0;
                                PFN[(size_t) sl] = 1;
                                I.fill_tok[(size_t) nm][(size_t) sl] = I.tok;
                                uint8_t* dst = I.slot_ptr(XN, sl);
                                cudaMemcpyAsync(dst, XN.up + (size_t) e * XN.ub, XN.ub, cudaMemcpyHostToDevice, I.cst);
                                cudaMemcpyAsync(dst + XN.ub, XN.dn + (size_t) e * XN.db, XN.db, cudaMemcpyHostToDevice,
                                                I.cst);
                            }
                    }
                } else {
                    // per expert, its tokens.  Groups of at most TC_GROUP experts and tc_rows token rows: their slots
                    // (copies of the misses), their weights to BF16, their tokens' latent rows gathered, one grouped
                    // GEMM up (each expert its own m and its scale_2), relu^2 to BF16, one grouped GEMM down, and the
                    // rows scattered to their (t, j) parts.  Without the tensor-core scratch (a chunk below TC_MIN):
                    // expert after expert on CUDA cores.
                    ExpertRows rows_of;
                    rows_of.build(h_ids.data(), T * K, NE);
                    if (pskip) {
                        // a miss that weighs < pskip_w in every row of the chunk: dropped (not copied, adds nothing)
                        for (int32_t e = 0; e < (int32_t) NE; ++e) {
                            if (rows_of[(size_t) e].empty() || X.slot_of[(size_t) e] >= 0) continue;
                            float mw = 0.0f;
                            for (int32_t i : rows_of[(size_t) e]) mw = std::max(mw, h_w[(size_t) i]);
                            if (mw < I.pskip_w) {
                                rows_of.clear((size_t) e);
                                ++stats_.expert_skips;
                            }
                        }
                    }
                    std::vector<int32_t> experts;
                    for (int32_t e = 0; e < (int32_t) NE; ++e)
                        if (!rows_of[(size_t) e].empty()) experts.push_back(e);
                    if (I.wb) {
                        // the layer's down projections: FP4 into the (t, j) rows (lf4d), else summed as they go
                        const bool lf4d = (I.fp4 & 4) && X.td == NemoWType::NVFP4 && I.parts_bf;
                        if (!lf4d) NCK(cudaMemsetAsync(I.routed_d, 0, (size_t) T * L * sizeof(double), st));
                        else if (pskip)   // (a dropped expert's rows are never written: they must read as 0)
                            NCK(cudaMemsetAsync(I.parts_bf, 0, (size_t) T * K * L * sizeof(uint16_t), st));
                        const int lm = b.moe;
                        // the layer's copies: planned ahead (all its non-resident experts), else now (those with rows)
                        // (a layer planned from a prediction gets the misses the prediction lacked appended)
                        if (I.pq_first[(size_t) lm] == SIZE_MAX || (pskip && I.pq_end[(size_t) lm] == I.pq.size()))
                            I.pq_gen(lm, &rows_of);
                        std::vector<int32_t>& PI = I.pitem[(size_t) lm];
                        const size_t q0 = I.pq_first[(size_t) lm], q1 = I.pq_end[(size_t) lm];
                        for (size_t k = q0; k < q1; ++k)
                            if (rows_of[(size_t) I.pq[k].e].empty()) I.pq_drop(k);
                        // the groups: the hits, then the copied experts in the sequence's order (a group's items span
                        // at most half the ring, so its copies never wait for its own reads)
                        struct Grp {
                            std::vector<int32_t> ex;
                            size_t last = SIZE_MAX;   ///< its last item (SIZE_MAX: hits only)
                            size_t first = SIZE_MAX;
                            int64_t R = 0, nseg = 0, io = 0, po = 0, to = 0, ntiles = 0;
                        };
                        std::vector<Grp> groups;
                        auto add = [&](int32_t e, size_t item) {
                            const int64_t m = (int64_t) rows_of[(size_t) e].size();
                            Grp* g = groups.empty() ? nullptr : &groups.back();
                            const bool hit = item == SIZE_MAX;
                            if (!g || (g->last == SIZE_MAX) != hit || (int64_t) g->ex.size() >= Impl::TC_GROUP ||
                                g->R + m > (I.lent ? Impl::LC : I.tc_rows) ||
                                (!hit && (int64_t) (item - g->first) >= I.ring_n / 2)) {
                                groups.emplace_back();
                                g = &groups.back();
                                g->first = item;
                            }
                            g->ex.push_back(e);
                            g->R += m;
                            if (!hit) g->last = item;
                        };
                        for (int32_t e : experts)
                            if (PI[(size_t) e] < 0) {
                                const int32_t sl = X.slot_of[(size_t) e];
                                X.used[(size_t) sl] = ++I.clock;
                                ++stats_.expert_hits;
                                add(e, SIZE_MAX);
                            }
                        for (size_t k = q0; k < q1; ++k)
                            if (!I.pq[k].skip && !rows_of[(size_t) I.pq[k].e].empty()) {
                                ++stats_.expert_misses;
                                add(I.pq[k].e, k);
                            }
                        // every group's lists, into the pinned stage and up in one copy:
                        //   [0, R) the gather rows (tokens), [R, 2R) the out rows by token, [2R, 3R) their w index,
                        //   then nseg + 1 segment starts and nseg tokens (the per-token sums in expert order)
                        int64_t io = 0, po = 0;
                        for (Grp& g : groups) {
                            g.io = io;
                            g.po = po;
                            const int n = (int) g.ex.size();
                            int32_t* id = I.h_eg_idx + io;
                            const uint8_t** pp = I.h_eg_ptr + po;
                            int64_t r = 0;
                            for (int i = 0; i < n; ++i) {
                                const int32_t e = g.ex[(size_t) i];
                                const int32_t it = PI[(size_t) e];
                                const uint8_t* bl = it < 0 ? I.slot_ptr(X, X.slot_of[(size_t) e])
                                                           : I.pq[(size_t) it].dst;
                                pp[i] = bl;
                                pp[n + i] = bl + X.ub;
                                for (int32_t q : rows_of[(size_t) e]) id[r++] = q / (int32_t) K;
                            }
                            if (lf4d) {
                                // [R, 2R): each packed row's (t, j) slot - the down GEMM writes there; no segments
                                int64_t q2 = 0;
                                for (int i = 0; i < n; ++i)
                                    for (int32_t q : rows_of[(size_t) g.ex[(size_t) i]]) id[g.R + q2++] = q;
                                g.nseg = 0;
                                io += 3 * g.R + 1;
                                goto tiles;
                            }
                            {
                            // the out rows by token (stable: a counting sort over the chunk's tokens), their (t, j),
                            // and the per-token segments
                            std::vector<int32_t>& tc = I.tok_cnt;
                            std::vector<int32_t>& tp = I.tok_pos;
                            if ((int64_t) tc.size() < T) {
                                tc.assign((size_t) T, 0);
                                tp.assign((size_t) T, 0);
                            }
                            for (int64_t q2 = 0; q2 < g.R; ++q2) ++tc[(size_t) id[q2]];
                            int32_t* ss = id + 3 * g.R;
                            int64_t ns = 0;
                            int32_t at = 0;
                            for (int64_t tk = 0; tk < T; ++tk) {
                                if (!tc[(size_t) tk]) continue;
                                ss[ns++] = at;
                                tp[(size_t) tk] = at;
                                at += tc[(size_t) tk];
                            }
                            ss[ns] = at;
                            g.nseg = ns;
                            {
                                int64_t q2 = 0, sg = 0;
                                for (int64_t tk = 0; tk < T; ++tk)
                                    if (tc[(size_t) tk]) {
                                        ss[ns + 1 + sg++] = (int32_t) tk;
                                        tc[(size_t) tk] = 0;
                                    }
                                for (int i = 0; i < n; ++i)
                                    for (int32_t q : rows_of[(size_t) g.ex[(size_t) i]]) {
                                        const int32_t pos = tp[(size_t) (q / (int32_t) K)]++;
                                        id[g.R + pos] = (int32_t) q2++;
                                        id[2 * g.R + pos] = q;
                                    }
                            }
                            io += 3 * g.R + 2 * g.nseg + 1;
                            }
                        tiles:
                            // the FP4 path: its tiles (expert, rows [r0, r1) of the group's packed rows) and alphas
                            g.to = io;
                            int64_t off = 0, nt = 0;
                            int32_t* tl = I.h_eg_idx + io;
                            for (int i = 0; i < n; ++i) {
                                const int64_t mi = (int64_t) rows_of[(size_t) g.ex[(size_t) i]].size();
                                for (int64_t r0 = 0; r0 < mi; r0 += 128) {
                                    tl[3 * nt] = i;
                                    tl[3 * nt + 1] = (int32_t) (off + r0);
                                    tl[3 * nt + 2] = (int32_t) (off + std::min<int64_t>(r0 + 128, mi));
                                    ++nt;
                                }
                                off += mi;
                            }
                            g.ntiles = nt;
                            float* al = (float*) (tl + 3 * nt);
                            for (int i = 0; i < n; ++i) {
                                al[i] = X.s_up[(size_t) g.ex[(size_t) i]];
                                al[n + i] = X.s_dn[(size_t) g.ex[(size_t) i]];
                            }
                            io += 3 * nt + 2 * n;
                            po += 2 * n;
                        }
                        if ((size_t) io > I.eg_idx_cap || (size_t) po > I.eg_ptr_cap) {
                            err = "the prompt's MoE lists overflow their buffers";
                            return false;
                        }
                        NCK(cudaMemcpyAsync(I.eg_idx, I.h_eg_idx, (size_t) io * sizeof(int32_t), cudaMemcpyHostToDevice, st));
                        NCK(cudaMemcpyAsync(I.eg_ptr, I.h_eg_ptr, (size_t) po * sizeof(void*), cudaMemcpyHostToDevice, st));
                        I.mark(7);
                        if (I.hostprof) {
                            I.plan_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_t0).count();
                            ++I.plan_n;
                        }
                        // the up GEMMs' A: the chunk's latent rows to NVFP4 once (every expert of a token reads the
                        // same row: the GEMM gathers it by token), in eg_xb's room
                        const bool latq = (I.fp4 & 2) && X.tu == NemoWType::NVFP4 && !I.lat_q_off;
                        uint8_t* lq = (uint8_t*) I.eg_xb;
                        uint8_t* ls = lq + (size_t) T * L / 2;
                        float* lgs = (float*) (ls + (size_t) T * L / 16);
                        if (latq) strata::kernels::nemo_fp4_quant_rows(I.lat, L, nullptr, T, L, false, lq, ls, lgs, st);
                        for (Grp& g : groups) {
                            const int n = (int) g.ex.size();
                            const int64_t R = g.R, nseg = g.nseg;
                            int32_t* gidx = I.eg_idx + g.io;
                            if (g.last != SIZE_MAX) {
                                // its copies, then the compute waits for them
                                I.pq_copy_until(g.last + 1);
                                if (I.pq_copied < g.last + 1) {
                                    err = "the prompt's expert ring is stuck (STRATA_NEMO_RING too small?)";
                                    return false;
                                }
                                NCK(cudaEventRecord(I.pcopy, I.cst));
                                NCK(cudaStreamWaitEvent(st, I.pcopy, 0));
                            }
                            const bool f4u = (I.fp4 & 2) && X.tu == NemoWType::NVFP4, f4d = (I.fp4 & 4) && X.td == NemoWType::NVFP4;
                            // the GEMMs read the slots' ggml rows themselves (measured no slower than converting them
                            // to N4 first at any length, and much faster on a short chunk, whose groups are mostly
                            // hits with a few rows each); STRATA_NEMO_GG_MAX = rows: convert above that
                            const bool gg = T <= I.gg_max;
                            // the weights out of their slots: N4 for the FP4 GEMMs, else BF16
                            uint8_t* uq = (uint8_t*) I.eg_up;
                            uint8_t* us = uq + (size_t) n * FF * L / 2;
                            uint8_t* dq = (uint8_t*) I.eg_dn;
                            uint8_t* ds = dq + (size_t) n * L * FF / 2;
                            if (f4u && !gg) strata::kernels::nemo_fp4_from_ggml_many(I.eg_ptr + g.po, n, FF, L, uq, us, st);
                            if (!f4u) strata::kernels::nemo_dequant_bf16_many_t(X.tu, I.eg_ptr + g.po, I.eg_up, n, FF, L, st);
                            if (f4d && !gg) strata::kernels::nemo_fp4_from_ggml_many(I.eg_ptr + g.po + n, n, L, FF, dq, ds, st);
                            if (!f4d) strata::kernels::nemo_dequant_bf16_many_t(X.td, I.eg_ptr + g.po + n, I.eg_dn, n, L, FF, st);
                            // the group's ring blobs are free for the copies behind once the last reader is done: the
                            // expansions above, or the FP4 GEMMs below
                            auto release = [&]() {
                                if (g.last == SIZE_MAX) return;
                                // its ring blobs are read: free for the copies behind
                                const size_t ev = I.pev_next++ % I.pev.size();
                                cudaEventRecord(I.pev[ev], st);
                                for (int32_t e : g.ex) {
                                    const Impl::PItem& it = I.pq[(size_t) PI[(size_t) e]];
                                    if (it.ring >= 0) {
                                        I.ring_rel[(size_t) it.ring] = (int32_t) ev;
                                        I.ring_busy[(size_t) it.ring] = 0;
                                    }
                                }
                                I.pq_copy_until(SIZE_MAX);
                            };
                            if ((!f4u && !f4d) || !gg) release();   // (else: after the GEMMs that read the slots)
                            const int32_t* gtiles = gidx + (g.to - g.io);
                            const float* gal = (const float*) (gtiles + 3 * g.ntiles);
                            if (f4u) {
                                // up: the tokens' latent rows gathered straight into NVFP4, relu^2 in the epilogue
                                uint8_t* aq = latq ? lq : (uint8_t*) I.eg_xb;
                                uint8_t* as = latq ? ls : aq + (size_t) R * L / 2;
                                float* ags = latq ? lgs : (float*) (as + (size_t) R * L / 16);
                                if (!latq) strata::kernels::nemo_fp4_quant_rows(I.lat, L, gidx, R, L, false, aq, as, ags, st);
                                // (relu^2 in BF16 into eg_hb when the down GEMM is FP4 too: it is read only to be
                                // quantized, and half the bytes is half the time of that round trip)
                                strata::kernels::nemo_fp4_gemm_grouped(aq, as, ags, uq, us, gal, gtiles, (int) g.ntiles,
                                                                       I.eg_h, FF, L, true, st, f4d ? I.eg_hb : nullptr,
                                                                       nullptr, gg ? I.eg_ptr + g.po : nullptr,
                                                                       latq ? gidx : nullptr);
                            }
                            std::vector<const uint16_t*> xs(n), ws(n), hs(n), wd(n);
                            std::vector<float*> ys(n), os(n);
                            std::vector<int> ms(n);
                            std::vector<float> au(n), ad(n);
                            int64_t off = 0;
                            for (int i = 0; i < n; ++i) {
                                const int64_t mi = (int64_t) rows_of[(size_t) g.ex[(size_t) i]].size();
                                xs[(size_t) i] = I.eg_xb + off * L;
                                ws[(size_t) i] = I.eg_up + (size_t) i * FF * L;
                                ys[(size_t) i] = I.eg_h + off * FF;
                                hs[(size_t) i] = I.eg_hb + off * FF;
                                wd[(size_t) i] = I.eg_dn + (size_t) i * L * FF;
                                os[(size_t) i] = I.eg_out + off * L;
                                ms[(size_t) i] = (int) mi;
                                au[(size_t) i] = X.s_up[(size_t) g.ex[(size_t) i]];
                                ad[(size_t) i] = X.s_dn[(size_t) g.ex[(size_t) i]];
                                off += mi;
                            }
                            if (!f4u) {   // up in BF16 (cuBLAS): eg_h without relu^2
                                strata::kernels::nemo_gather_rows(I.lat, gidx, I.eg_in, R, L, st);
                                strata::kernels::nemo_f32_to_bf16(I.eg_in, L, I.eg_xb, R, L, false, st);
                                if ((I.a4 & 2) && X.tu == NemoWType::NVFP4) strata::kernels::nemo_fq_nvfp4_bf16(I.eg_xb, R, L, st);
                                if (!strata::kernels::nemo_gemm_bf16_grouped(xs.data(), ws.data(), ys.data(), ms.data(),
                                                                            au.data(), n, FF, L, st)) {
                                    err = "cuBLAS grouped GEMM (experts up) failed";
                                    return false;
                                }
                            }
                            if (f4d) {
                                // h to NVFP4 (relu^2 here when the up GEMM left it), the FP4 tensor cores
                                uint8_t* aq = f4u ? (uint8_t*) I.eg_h : (uint8_t*) I.eg_hb;
                                uint8_t* as = aq + (size_t) R * FF / 2;
                                float* ags = (float*) (as + (size_t) R * FF / 16);
                                if (!f4u || !strata::kernels::nemo_fp4_quant_rows_bf16(I.eg_hb, FF, R, FF, aq, as, ags, st)) {
                                    if (f4u) {
                                        err = "BF16 rows the FP4 quantization does not take";
                                        return false;
                                    }
                                    strata::kernels::nemo_fp4_quant_rows(I.eg_h, FF, nullptr, R, FF, true, aq, as, ags, st);
                                }
                                strata::kernels::nemo_fp4_gemm_grouped(aq, as, ags, dq, ds, gal + n, gtiles,
                                                                       (int) g.ntiles, I.eg_out, L, FF, false, st,
                                                                       lf4d ? I.parts_bf : nullptr,
                                                                       lf4d ? gidx + R : nullptr,
                                                                       gg ? I.eg_ptr + g.po + n : nullptr);
                            } else {
                                strata::kernels::nemo_f32_to_bf16(I.eg_h, FF, I.eg_hb, R, FF, !f4u, st);
                                if ((I.a4 & 4) && X.td == NemoWType::NVFP4) strata::kernels::nemo_fq_nvfp4_bf16(I.eg_hb, R, FF, st);
                                if (!strata::kernels::nemo_gemm_bf16_grouped(hs.data(), wd.data(), os.data(), ms.data(),
                                                                            ad.data(), n, L, FF, st)) {
                                    err = "cuBLAS grouped GEMM (experts down) failed";
                                    return false;
                                }
                            }
                            if ((f4u || f4d) && gg) release();
                            if (!lf4d)
                                strata::kernels::nemo_accum_rows_f64(I.eg_out, gidx + 3 * R, gidx + 3 * R + nseg + 1,
                                                                     gidx + R, gidx + 2 * R, I.w, I.routed_d, nseg, L, st);
                        }
                        if (lf4d) strata::kernels::nemo_weighted_rows_bf16(I.parts_bf, I.w, I.routed, T, K, L, st);
                        else strata::kernels::nemo_f64_to_f32(I.routed_d, I.routed, T * L, st);
                        // the layer is planned out: its items are done; queue the next layer's copies blindly
                        for (size_t k = q0; k < q1; ++k) PI[(size_t) I.pq[k].e] = -1;
                        I.pq_first[(size_t) lm] = SIZE_MAX;
                        if (T >= I.blind_min && !pskip && lm + 1 < (int) I.moe_block.size() &&
                            I.pq_first[(size_t) lm + 1] == SIZE_MAX) {
                            I.pq_gen(lm + 1, nullptr);
                            I.pq_copy_until(SIZE_MAX);
                        } else if (ppred && I.pq_first[(size_t) lm + 1] == SIZE_MAX) {
                            // the next layer's predicted misses: largest predicted weight >= pskip_w
                            std::vector<float> mw((size_t) NE, 0.0f);
                            for (int64_t i = 0; i < T * K; ++i)
                                mw[(size_t) I.h_pp_ids[(size_t) i]] = std::max(mw[(size_t) I.h_pp_ids[(size_t) i]], I.h_pp_w[(size_t) i]);
                            std::vector<uint8_t> want((size_t) NE, 0);
                            for (int64_t e = 0; e < NE; ++e) want[(size_t) e] = mw[(size_t) e] >= I.pskip_w;
                            I.pq_gen(lm + 1, nullptr, &want);
                            I.pq_copy_until(SIZE_MAX);
                        }
                    } else {
                        std::vector<int32_t> h_idx;
                        h_idx.reserve((size_t) T * K * 2);
                        for (int32_t e : experts) {
                            for (int32_t i : rows_of[(size_t) e]) h_idx.push_back(i / (int32_t) K);   // the token
                            for (int32_t i : rows_of[(size_t) e]) h_idx.push_back(i);                 // its part row
                        }
                        NCK(cudaMemcpyAsync(I.idx, h_idx.data(), h_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st));
                        size_t at = 0;
                        for (int32_t e : experts) {
                            const auto& r = rows_of[(size_t) e];
                            const int64_t m = (int64_t) r.size();
                            const uint8_t* bl = ensure(e, false);
                            strata::kernels::nemo_gather_rows(I.lat, I.idx + at, I.g_in, m, L, st);
                            for (int64_t m0 = 0; m0 < m; m0 += strata::kernels::NEMO_GEMV_MAX_M) {
                                const int64_t mm = std::min<int64_t>(strata::kernels::NEMO_GEMV_MAX_M, m - m0);
                                strata::kernels::nemo_gemv(X.tu, bl, X.s_up[(size_t) e], I.g_in + m0 * L, L,
                                                           I.g_h + m0 * FF, FF, FF, L, mm, true, false, st);
                                strata::kernels::nemo_gemv(X.td, bl + X.ub, X.s_dn[(size_t) e], I.g_h + m0 * FF, FF,
                                                           I.g_out + m0 * L, L, L, FF, mm, false, false, st);
                            }
                            strata::kernels::nemo_scatter_rows(I.g_out, I.idx + at + m, I.parts, m, L, st);
                            at += (size_t) (2 * m);
                        }
                    }
                    if (!I.wb) strata::kernels::nemo_weighted_rows(I.parts, I.w, I.routed, T, K, L, st);
                }
                I.mark(4);
                I.gemm(b.fc2, I.routed, L, I.x, E, T, false, true);       // x += fc2(routed)
                for (int64_t s0 = 0; s0 < T && dec && !sh_done; s0 += I.SC) {   // x += shared(xn), in sub-chunks
                    const int64_t n1 = std::min<int64_t>(I.SC, T - s0);
                    I.gemm(b.sh_up, I.xn + s0 * E, E, I.sh_h, SH, n1, true, false);
                    I.gemm(b.sh_dn, I.sh_h, SH, I.x + s0 * E, E, n1, false, true);
                }
                I.mark(5);
            }
        }
        // the head on the chunk's last row (every row for forward_all, one row at a time through head_out)
        if (I.verify) {
            // a verify window: every row's logits and argmax (the MTP entries are spec_step's)
            strata::kernels::nemo_rms_norm_rows(I.x, I.out_norm, I.xn, T, E, eps, st);
            I.gemm(I.head, I.xn, E, I.vlogits, g_.n_vocab, T, false, false);
            strata::kernels::nemo_argmax_rows(I.vlogits, T, g_.n_vocab, I.vtok, st);
        } else if (I.all_logits) {
            strata::kernels::nemo_rms_norm_rows(I.x, I.out_norm, I.xn, T, E, eps, st);
            for (int64_t t = 0; t < T; ++t) {
                I.gemm(I.head, I.xn + t * E, E, I.head_out, g_.n_vocab, 1, false, false);
                NCK(cudaMemcpyAsync(I.all_logits + (size_t) (done + t) * g_.n_vocab, I.head_out,
                                    sizeof(float) * g_.n_vocab, cudaMemcpyDeviceToHost, st));
            }
        } else if (I.mtp.on) {
            // every row normed: the MTP block's entries for this chunk's rows (each with the token after it), the
            // pending row of the previous call first; the last row waits for its next token (mtp_draft or the next
            // call)
            Impl::Mtp& M = I.mtp;
            strata::kernels::nemo_rms_norm_rows(I.x, I.out_norm, I.xn, T, E, eps, st);
            if (M.pending && M.consumed_tok != tokens[done] &&
                !mtp_rows(tokens + done, M.h, 1, M.pending_pos, false, err))
                return false;
            for (int64_t r0 = 0; r0 < T - 1; r0 += M.MC)
                if (!mtp_rows(tokens + done + 1 + r0, I.xn + r0 * E, std::min<int64_t>(M.MC, T - 1 - r0), pos_ + r0,
                              false, err))
                    return false;
            NCK(cudaMemcpyAsync(M.h, I.xn + (T - 1) * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, st));
            M.pending = true;
            M.pending_pos = pos_ + T - 1;
            M.consumed_tok = -1;
            I.gemm(I.head, I.xn + (T - 1) * E, E, I.head_out, g_.n_vocab, 1, false, false);
        } else {
            strata::kernels::nemo_rms_norm_rows(I.x + (T - 1) * E, I.out_norm, I.xn, 1, E, eps, st);
            I.gemm(I.head, I.xn, E, I.head_out, g_.n_vocab, 1, false, false);
        }
        NCK(cudaStreamSynchronize(st));
        I.mark(6);
        I.prof_tokens += (uint64_t) T;
        NCK(cudaGetLastError());
        pos_ += T;
        done += T;
    }
    if (!I.lent) {
        I.lend();
        stats_.slots_per_layer = I.slots;
    }
    if (logits_out) NCK(cudaMemcpy(logits_out, I.head_out, sizeof(float) * g_.n_vocab, cudaMemcpyDeviceToHost));
    return true;
}


bool NemoModel::topk_head(int k, float* vals, int32_t* ids, std::string& err) {
    Impl& I = *impl_;
    if (!I.d_tk_v || k <= 0 || k > 1024) { err = "topk_head: STRATA_NEMO_GPU_TOPK is off (or MTP not loaded) or k > 1024"; return false; }
    cudaStream_t st = I.st;
    strata::kernels::nemo_topk_rows(I.head_out, 1, g_.n_vocab, k, I.d_tk_v, I.d_tk_i, st);
    NCK(cudaMemcpyAsync(vals, I.d_tk_v, (size_t) k * sizeof(float), cudaMemcpyDeviceToHost, st));
    NCK(cudaMemcpyAsync(ids, I.d_tk_i, (size_t) k * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
    NCK(cudaStreamSynchronize(st));
    return true;
}

bool NemoModel::has_mtp() const { return impl_->mtp.on; }
void NemoModel::set_spec_min_p(float p) { impl_->spec_min_p = p; }

bool NemoModel::mtp_rows(const int32_t* toks, const float* h, int64_t n, int64_t pos0, bool full, std::string& err) {
    Impl& I = *impl_;
    Impl::Mtp& M = I.mtp;
    const int64_t E = g_.n_embd, QD = g_.n_head * g_.head_dim, KD = g_.n_head_kv * g_.head_dim;
    const int64_t NE = g_.n_expert, K = g_.n_expert_used, L = g_.moe_latent, FF = g_.n_ff_exp, SH = g_.n_ff_shexp;
    const float eps = (float) g_.rms_eps;
    cudaStream_t st = I.st;
    strata::kernels::NemoAttnShapes as;
    as.n_head = g_.n_head; as.n_head_kv = g_.n_head_kv; as.head_dim = g_.head_dim;
    (void) KD;
    // the next tokens' embeddings (the stage is the MTP's own: the main forward's may still be in flight)
    if (I.mtp_event) {
        // STRATA_NEMO_MTP_EVENT: only the previous H2D from the stage must be done (an unrecorded event is complete)
        if (!I.stage_ev) NCK(cudaEventCreateWithFlags(&I.stage_ev, cudaEventDisableTiming));
        NCK(cudaEventSynchronize(I.stage_ev));
    } else {
        NCK(cudaStreamSynchronize(st));
    }
    for (int64_t t = 0; t < n; ++t) std::memcpy(M.stage + t * E, I.embd + (size_t) toks[t] * E, E * sizeof(uint16_t));
    uint16_t* d_emb = (uint16_t*) M.a;
    NCK(cudaMemcpyAsync(d_emb, M.stage, (size_t) n * E * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
    if (I.mtp_event) NCK(cudaEventRecord(I.stage_ev, st));
    strata::kernels::nemo_bf16_rows_to_f32(d_emb, M.e, n, E, st);
    I.img_overlay(M.e, pos0 + 1, n, E, st);   // row t pairs h at pos0 + t with the token at pos0 + t + 1
    // u = eh_proj([enorm(e), hnorm(h)])
    strata::kernels::nemo_rms_norm_rows(M.e, M.enorm, M.u, n, E, eps, st);
    strata::kernels::nemo_rms_norm_rows(h, M.hnorm, M.hn, n, E, eps, st);
    NCK(cudaMemcpy2DAsync(M.cat, 2 * E * sizeof(float), M.u, E * sizeof(float), E * sizeof(float), (size_t) n,
                          cudaMemcpyDeviceToDevice, st));
    NCK(cudaMemcpy2DAsync(M.cat + E, 2 * E * sizeof(float), M.hn, E * sizeof(float), E * sizeof(float), (size_t) n,
                          cudaMemcpyDeviceToDevice, st));
    I.gemm(M.eh, M.cat, 2 * E, M.u, E, n, false, false);
    // NoPE attention over the MTP's own K/V
    strata::kernels::nemo_rms_norm_rows(M.u, M.attn_norm, M.a, n, E, eps, st);
    I.gemm(M.q, M.a, E, I.q, QD, n, false, false);
    I.gemm(M.k, M.a, E, I.k, g_.n_head_kv * g_.head_dim, n, false, false);
    I.gemm(M.v, M.a, E, I.v, g_.n_head_kv * g_.head_dim, n, false, false);
    const strata::kernels::NemoKV mkv = I.kv_of(M.kv);
    strata::kernels::nemo_kv_append(mkv, I.k, I.v, pos0, n, as, st);
    // (STRATA_NEMO_MTP_WINDOW = W: the draft block attends to the last W cells before the call only)
    strata::kernels::nemo_attn(I.q, mkv, I.o, pos0, n, as, I.attn_scratch, st,
                               I.mtp_window > 0 ? pos0 - I.mtp_window : 0);
    I.gemm(M.o, I.o, QD, M.u, E, n, false, true);
    if (full) {
        // the last row: the MoE (resident experts, routed on the device), the head's LayerNorm, the logits
        float* r = M.u + (n - 1) * E;
        strata::kernels::nemo_rms_norm_rows(r, M.post_norm, M.a, 1, E, eps, st);
        I.gemm(M.gate, M.a, E, I.logits, NE, 1, false, false);
        strata::kernels::nemo_router(I.logits, M.bias, I.ids, I.w, 1, NE, K, (float) g_.expert_weights_scale,
                                     g_.expert_weights_norm, st);
        I.gemm(M.fc1, M.a, E, I.lat, L, 1, false, false);
        if (M.nslots == g_.n_expert) {
            strata::kernels::nemo_resident_expert_ptrs(I.ids, M.experts, M.stride, M.ub, M.s_up, M.s_dn, M.ptr, M.s2, K,
                                                       st);
        } else {
            // the cache: hits from their slots, misses skipped (zero expert) and copied in for the next drafts
            int32_t* hid = (int32_t*) (M.h_meta + (size_t) 2 * K * (sizeof(void*) + sizeof(float)));
            NCK(cudaMemcpyAsync(hid, I.ids, (size_t) K * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            NCK(cudaStreamSynchronize(st));
            const uint8_t** hp = (const uint8_t**) M.h_meta;
            float* hs = (float*) (M.h_meta + (size_t) 2 * K * sizeof(void*));
            NCK(cudaStreamWaitEvent(st, M.fill_ev, 0));   // the earlier drafts' copies: this one may hit them
            const uint64_t c0 = ++M.clock;
            for (int64_t j = 0; j < K; ++j) {
                const int32_t e = hid[j], sl = M.slot_of[(size_t) e];
                if (sl >= 0) M.used[(size_t) sl] = c0;
            }
            for (int64_t j = 0; j < K; ++j) {
                const int32_t e = hid[j];
                int32_t sl = M.slot_of[(size_t) e];
                if (sl >= 0) {
                    ++M.hits;
                    hp[j] = M.experts + (size_t) sl * M.stride;
                    hp[K + j] = hp[j] + M.ub;
                    hs[j] = M.su_h[(size_t) e];
                    hs[K + j] = M.sd_h[(size_t) e];
                    continue;
                }
                ++M.misses;
                hp[j] = I.zero_blob;
                hp[K + j] = I.zero_blob + M.ub;
                hs[j] = hs[K + j] = 0.0f;
                // the copy, into the oldest slot not used now
                int32_t v = -1;
                uint64_t oldest = UINT64_MAX;
                for (int32_t c = 0; c < (int32_t) M.nslots; ++c)
                    if (M.used[(size_t) c] < oldest && M.used[(size_t) c] < c0) { oldest = M.used[(size_t) c]; v = c; }
                if (v < 0) continue;
                if (M.expert_in[(size_t) v] >= 0) M.slot_of[(size_t) M.expert_in[(size_t) v]] = -1;
                M.expert_in[(size_t) v] = e;
                M.slot_of[(size_t) e] = v;
                M.used[(size_t) v] = c0;
                cudaMemcpyAsync(M.experts + (size_t) v * M.stride, M.host + (size_t) e * M.stride, M.stride,
                                cudaMemcpyHostToDevice, I.cst);
            }
            NCK(cudaEventRecord(M.fill_ev, I.cst));
            NCK(cudaMemcpyAsync(M.ptr, hp, 2 * K * sizeof(void*), cudaMemcpyHostToDevice, st));
            NCK(cudaMemcpyAsync(M.s2, hs, 2 * K * sizeof(float), cudaMemcpyHostToDevice, st));
        }
        strata::kernels::nemo_experts_part(M.tu, M.td, M.ptr, M.ptr + K, M.s2, M.s2 + K,
                                           nullptr, nullptr, K, I.lat, I.eh, K, L, FF, st);
        strata::kernels::nemo_experts_sum(I.w, I.eh, I.routed, K, L, FF, st);
        I.gemm(M.fc2, I.routed, L, r, E, 1, false, true);
        I.gemm(M.sh_up, M.a, E, I.sh_h, SH, 1, true, false);
        I.gemm(M.sh_dn, I.sh_h, SH, r, E, 1, false, true);
        strata::kernels::nemo_layer_norm_rows(r, M.head_norm, M.hnext, 1, E, eps, st);
        I.gemm(I.head, M.hnext, E, M.logits, g_.n_vocab, 1, false, false);
        strata::kernels::nemo_argmax_rows(M.logits, 1, g_.n_vocab, M.d_tok, st, (float*) (M.d_tok + 4));
    }
    NCK(cudaGetLastError());
    return true;
}

bool NemoModel::mtp_draft(int32_t next, int n, int32_t* drafts, std::string& err) {
    Impl& I = *impl_;
    Impl::Mtp& M = I.mtp;
    if (!M.on || !M.pending || n <= 0) return false;
    int32_t tok = next;
    const float* hsrc = M.h;
    for (int i = 0; i < n; ++i) {
        if (M.pending_pos + i >= max_ctx_) return false;
        if (!mtp_rows(&tok, hsrc, 1, M.pending_pos + i, true, err)) return false;
        NCK(cudaMemcpyAsync(M.h_tok, M.d_tok, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, I.st));
        NCK(cudaStreamSynchronize(I.st));
        drafts[i] = M.h_tok[0];
        tok = drafts[i];
        hsrc = M.hnext;
        float pr;
        std::memcpy(&pr, M.h_tok + 4, 4);
        // the chain stops at a draft the MTP is unsure of (STRATA_NEMO_SPEC_MIN_P; the first draft always stays)
        if (i + 1 < n && pr < I.spec_min_p) {
            M.consumed_tok = next;
            M.drafted = i + 1;
            return true;
        }
    }
    M.consumed_tok = next;
    M.drafted = n;
    return true;
}


bool NemoModel::spec_step(int32_t next, int n, std::vector<int32_t>& out, std::string& err,
                          const std::function<int32_t(const float*)>* pick,
                          const std::function<int32_t(const float*, const int32_t*, int)>* pick_topk, int topk) {
    Impl& I = *impl_;
    Impl::Mtp& M = I.mtp;
    out.clear();
    const bool gtk = pick && pick_topk && topk > 0 && topk <= 1024 && I.gpu_topk_on();
    n = std::min<int>(n, strata::kernels::NEMO_GEMV_MAX_M - 1);
    if (pos_ + n + 1 > max_ctx_) n = (int) (max_ctx_ - pos_ - 1);
    std::vector<int32_t> w((size_t) n + 1);
    w[0] = next;
    const auto tq0 = std::chrono::steady_clock::now();
    if (!M.on || !M.pending || n <= 0 || !mtp_draft(next, n, w.data() + 1, err)) {
        if (!err.empty()) return false;
        // no drafting possible: one plain step
        if (gtk) {
            if (!forward(&next, 1, nullptr, err) || !topk_head(topk, I.h_tk_v, I.h_tk_i, err)) return false;
            out.push_back((*pick_topk)(I.h_tk_v, I.h_tk_i, topk));
            return true;
        }
        std::vector<float> lg((size_t) g_.n_vocab);
        if (!forward(&next, 1, lg.data(), err)) return false;
        out.push_back(pick ? (*pick)(lg.data()) : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin()));
        return true;
    }
    n = M.drafted;
    w.resize((size_t) n + 1);
    const int64_t E = g_.n_embd, IN = g_.ssm_in_dim(), Mw = n + 1;
    const size_t sb = sizeof(float) * g_.ssm_n_head * g_.ssm_head_dim() * g_.ssm_d_state;
    const size_t cb = sizeof(float) * g_.ssm_conv_dim() * (g_.ssm_d_conv - 1);
    cudaStream_t st = I.st;
    // the Mamba state before the window (STRATA_NEMO_MAMBA_SWAP: none; the window's scan writes elsewhere)
    const bool swap = I.mamba_swap && Mw >= 2 && Mw < 64 && Mw <= I.SC;
    if (!swap)
    for (Block& b : I.blocks)
        if (b.kind == NemotronBlock::Mamba) {
            NCK(cudaMemcpyAsync(I.spec_state[(size_t) b.mamba], b.state, sb, cudaMemcpyDeviceToDevice, st));
            NCK(cudaMemcpyAsync(I.spec_conv[(size_t) b.mamba], b.conv, cb, cudaMemcpyDeviceToDevice, st));
        }
    const int64_t p0 = pos_;
    const auto tq1 = std::chrono::steady_clock::now();
    I.verify = true;
    I.verify_swap = swap;
    const bool ok = forward(w.data(), Mw, nullptr, err);
    I.verify = false;
    I.verify_swap = false;
    if (!ok) return false;
    const auto tq2 = std::chrono::steady_clock::now();
    int64_t a = 1;   // rows kept: the fed token and the drafts the main model agrees with
    int32_t last = -1;
    if (!pick) {
        NCK(cudaMemcpyAsync(I.h_vtok, I.vtok, (size_t) Mw * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        NCK(cudaStreamSynchronize(st));
        while (a < Mw && w[(size_t) a] == I.h_vtok[a - 1]) ++a;
        last = I.h_vtok[a - 1];
    } else if (gtk) {
        // sampled, top-k candidates from the device: one kernel over the Mw rows, one read-back, then the host draws
        // exactly as the full-row path does
        strata::kernels::nemo_topk_rows(I.vlogits, Mw, g_.n_vocab, topk, I.d_tk_v, I.d_tk_i, st);
        NCK(cudaMemcpyAsync(I.h_tk_v, I.d_tk_v, (size_t) Mw * topk * sizeof(float), cudaMemcpyDeviceToHost, st));
        NCK(cudaMemcpyAsync(I.h_tk_i, I.d_tk_i, (size_t) Mw * topk * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        NCK(cudaStreamSynchronize(st));
        for (int64_t r = 0; r < Mw; ++r) {
            const int32_t y = (*pick_topk)(I.h_tk_v + r * topk, I.h_tk_i + r * topk, topk);
            if (r + 1 < Mw && y == w[(size_t) r + 1]) { ++a; continue; }
            last = y;
            break;
        }
    } else {
        // a sampled step: each row's token drawn from the main model's own distribution, a draft kept while it is
        // the token drawn (the kept tokens are exactly what plain sampling would have drawn)
        std::vector<float>& row = I.pick_row;
        row.resize((size_t) g_.n_vocab);
        for (int64_t r = 0; r < Mw; ++r) {
            NCK(cudaMemcpy(row.data(), I.vlogits + (size_t) r * g_.n_vocab, sizeof(float) * g_.n_vocab,
                           cudaMemcpyDeviceToHost));
            const int32_t y = (*pick)(row.data());
            if (r + 1 < Mw && y == w[(size_t) r + 1]) { ++a; continue; }
            last = y;
            break;
        }
    }
    stats_.spec_windows += 1;
    stats_.spec_drafted += (uint64_t) n;
    stats_.spec_accepted += (uint64_t) (a - 1);
    if (swap && a == Mw) {
        // a full accept: the window's state is the new one
        for (Block& b : I.blocks)
            if (b.kind == NemotronBlock::Mamba) {
                std::swap(b.state, I.spec_state[(size_t) b.mamba]);
                std::swap(b.conv, I.spec_conv[(size_t) b.mamba]);
            }
    }
    if (a < Mw) {
        // back to the state before the window, then the kept rows again (conv + scan only: their outputs were used)
        strata::kernels::MambaShapes ms;
        ms.H = g_.ssm_n_head; ms.P = g_.ssm_head_dim(); ms.N = g_.ssm_d_state; ms.G = g_.ssm_n_group;
        ms.d_conv = g_.ssm_d_conv;
        for (Block& b : I.blocks)
            if (b.kind == NemotronBlock::Mamba) {
                if (!swap) {   // (swap: b.state / b.conv were never touched)
                    NCK(cudaMemcpyAsync(b.state, I.spec_state[(size_t) b.mamba], sb, cudaMemcpyDeviceToDevice, st));
                    NCK(cudaMemcpyAsync(b.conv, I.spec_conv[(size_t) b.mamba], cb, cudaMemcpyDeviceToDevice, st));
                }
                const float* zx = I.spec_zx + (size_t) b.mamba * strata::kernels::NEMO_GEMV_MAX_M * IN;
                strata::kernels::mamba_conv_prefill(b.conv, zx, IN, b.conv_w, b.conv_b, I.xbc, a, ms, st);
                strata::kernels::mamba_ssm_scan(b.state, I.xbc, zx, IN, b.dt_b, b.A, b.D, I.y, a, ms, st);
            }
        pos_ = p0 + a;
    }
    // the MTP block's entries for the kept rows but the last (true h, true next token), then the last row pending
    if (a > 1 && !mtp_rows(w.data() + 1, I.xn, a - 1, p0, false, err)) return false;
    NCK(cudaMemcpyAsync(M.h, I.xn + (a - 1) * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, st));
    M.pending = true;
    M.pending_pos = p0 + a - 1;
    M.consumed_tok = -1;
    NCK(cudaMemcpyAsync(I.head_out, I.vlogits + (size_t) (a - 1) * g_.n_vocab, sizeof(float) * g_.n_vocab,
                        cudaMemcpyDeviceToDevice, st));
    for (int64_t r = 1; r < a; ++r) out.push_back(w[(size_t) r]);
    out.push_back(last);
    NCK(cudaStreamSynchronize(st));
    const auto tq3 = std::chrono::steady_clock::now();
    I.spec_ms[0] += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    I.spec_ms[1] += std::chrono::duration<double, std::milli>(tq2 - tq1).count();
    I.spec_ms[2] += std::chrono::duration<double, std::milli>(tq3 - tq2).count();
    I.spec_ms[3] += 1;
    return true;
}

}  // namespace strata::core
