// src/program/nemo_main.cpp - strata-nemotron: the engine process for Nemotron-H GGUFs (strata-nemotron).
//
// Two modes:
//
//   --ppl IDS     perplexity over a token file (whitespace-separated ids), llama-perplexity's way: chunks of --ctx
//                 tokens, the first replaced by BOS, the second half of each chunk scored.  The same numbers as
//                 `llama-perplexity -c <ctx>` on the same ids are the engine's whole-model check.  --ppl-decode
//                 feeds the scored half token by token (the decode path: expert cache, tail skip); --ll-dump FILE
//                 writes each scored token's log-probability.
//   --serve       Strata's engine wire contract, so serve/server.py drives it like the qwen4exp engine:
//                 stdout  INFO k=v ... ; READY <context> stop ; per request T <id> ... DONE <generated> <prompt>
//                         <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> ; ERR <message>
//                 stdin   GEN <max_new> [key=value ...] <id,id,...>  (temperature, top_p, top_k, min_p, seed;
//                         other keys are accepted and ignored); GENI <max_new> [key=value ...] <file> <id,...>:
//                         the same with images, the file strata-vision's records in prompt order, each one the rows
//                         of an <img><image>...</img> run (img_keys=k1,k2: their hashes, for the reuse below);
//                         STOP (the running request ends at its next token);
//                         WRAP (the same, for the server's shutdown); QUIT
//
// Conversation reuse: after a prompt is read its state is set aside; a request whose prompt starts with that prompt
// continues from there and reads only the rest (Mamba's state cannot be rolled back, so this is the one reuse there
// is: exactly "the previous prompt, then more").
//
// Usage: strata-nemotron --gguf FILE [--experts-gguf FILE] [--max-context N] [--chunk N] [--vram-reserve-mib N] [--expert-slots N]
//                        [--kv-base N] [--spec N] [--tail-skip R] [--prefetch 0|1] [--prefetch-rank R]
//                        (--serve | --ppl IDS [--ctx 1024] [--chunks 16] | --bench IDS [--ctx N] [--gen N] [--temp T])
#include "strata/core/nemo_conv_cache.hpp"
#include "strata/core/nemo_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>  // read()
using ssize_t = long long;
#include <cstdlib>
inline int setenv(const char* k, const char* v, int overwrite) {
    if (!overwrite && std::getenv(k)) return 0;
    return _putenv_s(k, v);
}
#else
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    uint64_t seed = 0;
};

/// The draw of sample() over the top-k candidates (descending logits `v` with their ids), temperature > 0.
int32_t sample_topk(const float* v, const int32_t* ids, int k, const Sampling& s, std::mt19937_64& rng) {
    std::vector<double> p((size_t) k);
    const double mx = v[0];
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += (p[(size_t) i] = std::exp((v[i] - mx) / s.temperature));
    int keep = k;
    double cum = 0;
    for (int i = 0; i < k; ++i) {
        p[(size_t) i] /= sum;
        if (s.min_p > 0 && p[(size_t) i] < s.min_p * p[0]) { keep = i; break; }
        cum += p[(size_t) i];
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max(keep, 1);
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (int i = 0; i < keep; ++i) {
        r -= p[(size_t) i];
        if (r <= 0) return ids[i];
    }
    return ids[keep - 1];
}

/// STRATA_NEMO_GPU_TOPK=1: the candidates come from the device (k = the request's top_k, or 1024 when unset; a
/// top_k above 1024 keeps the full-row path).  Returns k, or 0 when off.
int gpu_topk_k(const Sampling& s) {
    const char* e = std::getenv("STRATA_NEMO_GPU_TOPK");
    if (!e || !std::atoi(e) || s.temperature <= 0.0f || s.top_k > 1024) return 0;
    return s.top_k > 0 ? s.top_k : 1024;
}

/// Greedy at temperature <= 0; otherwise temperature, then top-k, top-p and min-p over the softmax, then a draw.
int32_t sample(const float* logits, int64_t n, const Sampling& s, std::mt19937_64& rng) {
    if (s.temperature <= 0.0f) return (int32_t) (std::max_element(logits, logits + n) - logits);
    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = s.top_k > 0 && s.top_k < n ? s.top_k : (int) std::min<int64_t>(n, 1024);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    idx.resize((size_t) k);
    std::vector<double> p((size_t) k);
    const double mx = logits[idx[0]];
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += (p[(size_t) i] = std::exp((logits[idx[(size_t) i]] - mx) / s.temperature));
    int keep = k;
    double cum = 0;
    for (int i = 0; i < k; ++i) {
        p[(size_t) i] /= sum;
        if (s.min_p > 0 && p[(size_t) i] < s.min_p * p[0]) { keep = i; break; }
        cum += p[(size_t) i];
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max(keep, 1);
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (int i = 0; i < keep; ++i) {
        r -= p[(size_t) i];
        if (r <= 0) return idx[(size_t) i];
    }
    return idx[(size_t) keep - 1];
}

/// A GENI request's embeddings file: strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats)
/// one after another, in prompt order.  `rows` gets every record's rows, `counts` each record's n.
bool read_sve(const std::string& path, int64_t n_embd, std::vector<float>& rows, std::vector<int64_t>& counts,
              std::string& err) {
    rows.clear();
    counts.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open the embeddings file " + path;
        return false;
    }
    int32_t h[5];
    bool ok = true;
    while (std::fread(h, sizeof h, 1, f) == 1) {
        if (h[0] != 0x31455653 || h[1] <= 0 || h[4] != n_embd) {
            err = "the embeddings file " + path + " is not strata-vision's (or its rows are not " +
                  std::to_string(n_embd) + " wide)";
            ok = false;
            break;
        }
        const size_t at = rows.size();
        rows.resize(at + (size_t) h[1] * n_embd);
        if (std::fread(rows.data() + at, sizeof(float), (size_t) h[1] * n_embd, f) != (size_t) h[1] * n_embd) {
            err = "the embeddings file " + path + " is cut short";
            ok = false;
            break;
        }
        counts.push_back(h[1]);
    }
    std::fclose(f);
    return ok;
}

/// The prompt's images: runs of <image> (18) between <img> (19) and </img> (20), as NVIDIA's processor writes them.
std::vector<std::pair<int64_t, int64_t>> image_runs(const std::vector<int32_t>& p) {
    std::vector<std::pair<int64_t, int64_t>> runs;
    for (size_t i = 1; i < p.size(); ++i) {
        if (p[i] != 18 || p[i - 1] != 19) continue;
        size_t j = i;
        while (j < p.size() && p[j] == 18) ++j;
        if (j < p.size() && p[j] == 20) runs.push_back({(int64_t) i, (int64_t) (j - i)});
        i = j;
    }
    return runs;
}

bool parse_ids(const std::string& s, std::vector<int32_t>& out) {
    out.clear();
    const char* p = s.c_str();
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) return false;
        out.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return !out.empty();
}

int run_ppl(strata::core::NemoModel& m, const char* path, int64_t ctx, int64_t chunks, bool bos_first, bool decode,
            const char* dump) {
    std::vector<int32_t> ids;
    {
        FILE* f = std::fopen(path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
        long v;
        while (std::fscanf(f, "%ld", &v) == 1) ids.push_back((int32_t) v);
        std::fclose(f);
    }
    const int64_t n_vocab = m.geometry().n_vocab;
    const int64_t have = (int64_t) ids.size() / ctx;
    if (chunks <= 0 || chunks > have) chunks = have;
    std::vector<float> lg((size_t) (ctx * n_vocab));
    double nll = 0;
    int64_t scored = 0;
    const int32_t bos = 1;
    FILE* df = dump ? std::fopen(dump, "w") : nullptr;
    const auto t0 = Clock::now();
    for (int64_t c = 0; c < chunks; ++c) {
        std::vector<int32_t> tok(ids.begin() + c * ctx, ids.begin() + (c + 1) * ctx);
        if (bos_first) tok[0] = bos;
        m.reset();
        std::string err;
        if (decode) {
            // the first half as a prompt, then the scored half token by token through the decode path
            const int64_t h = ctx / 2;
            if (!m.forward(tok.data(), h, lg.data() + (h - 1) * n_vocab, err)) { std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
            for (int64_t t = h; t < ctx - 1; ++t)
                if (!m.forward(tok.data() + t, 1, lg.data() + t * n_vocab, err)) { std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
        } else if (!m.forward_all(tok.data(), ctx, lg.data(), err)) { std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
        for (int64_t t = ctx / 2; t < ctx - 1; ++t) {
            const float* l = lg.data() + t * n_vocab;
            const float mx = *std::max_element(l, l + n_vocab);
            double s = 0;
            for (int64_t i = 0; i < n_vocab; ++i) s += std::exp((double) l[i] - mx);
            const double ll = (double) l[tok[(size_t) t + 1]] - mx - std::log(s);
            nll -= ll;
            ++scored;
            if (df) std::fprintf(df, "%.6f\n", ll);
        }
        std::printf("[%lld] %.4f\n", (long long) c + 1, std::exp(nll / (double) scored));
        std::fflush(stdout);
    }
    std::printf("Final estimate: PPL = %.4f over %lld scored tokens (%lld chunks of %lld), %.1f s\n",
                std::exp(nll / (double) scored), (long long) scored, (long long) chunks, (long long) ctx,
                ms_since(t0) / 1000.0);
    if (df) std::fclose(df);
    const auto st = m.stats();
    if (decode)
        std::printf("expert hits %llu, misses %llu, skipped %llu\n", (unsigned long long) st.expert_hits,
                    (unsigned long long) st.expert_misses, (unsigned long long) st.expert_skips);
    return 0;
}

int g_spec = 0;   ///< --spec N: drafts a window (serve), 0 = no speculation
strata::core::ConversationCache g_cc;   ///< --prompt-cache N / --conversation-cache-mib X (docs/nemotron/CONVERSATION_CACHE.md)
int64_t g_every = 16384;                ///< --prompt-cache-every M: a checkpoint every M tokens of a long read (0 = off)

int run_serve(strata::core::NemoModel& m) {
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // windows has no SIGPIPE; a closed socket returns an error instead
#endif
    const int64_t n_vocab = m.geometry().n_vocab;
    const auto st = m.stats();
    std::printf("INFO engine=strata-nemotron-v1 engine_kind=nemotron context=%lld kv=bf16 expert_slots=%lld "
                "expert_cache_mib=%lld experts=%lld pinned=%d logit_bias=0\n",
                (long long) m.max_context(), (long long) (st.slots_per_layer * m.geometry().count(strata::core::NemotronBlock::Moe)),
                (long long) (st.expert_cache_gib * 1024), (long long) (m.geometry().n_expert * m.geometry().count(strata::core::NemotronBlock::Moe)),
                st.pinned ? 1 : 0);
    std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);

    std::atomic<bool> stop_req{false}, quit_req{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::thread([&] {
        std::string buf, l;
        char chunk[65536];
        for (;;) {
            const size_t nl = buf.find('\n');
            if (nl != std::string::npos) {
                l.assign(buf, 0, nl);
                buf.erase(0, nl + 1);
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP" || l.rfind("STOP ", 0) == 0 || l == "WRAP" || l.rfind("PAUSE", 0) == 0) {
                    stop_req.store(true);
                    continue;
                }
                if (l == "QUIT") { quit_req.store(true); stop_req.store(true); }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
                continue;
            }
            const ssize_t n = ::read(0, chunk, sizeof chunk);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            buf.append(chunk, (size_t) n);
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }).detach();

    const std::vector<int32_t> eos = {2, 11};   // </s>, <|im_end|> (generation_config.json)
    std::vector<int32_t> snap_prompt;           // the prompt the model's snapshot is the end of
    std::vector<std::string> snap_keys;         // the keys of the pictures in snap_prompt
    const int64_t n_embd = m.geometry().n_embd;
    std::vector<float> logits((size_t) n_vocab);
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !lines.empty() || eof; });
            if (lines.empty()) break;
            line = std::move(lines.front());
            lines.pop_front();
        }
        if (line == "QUIT") break;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        const bool geni = cmd == "GENI";
        if (cmd != "GEN" && !geni) {
            if (!cmd.empty()) std::printf("ERR unknown command %s\n", cmd.c_str());
            std::fflush(stdout);
            continue;
        }
        stop_req.store(false);
        long long max_new = 0;
        ss >> max_new;
        Sampling sp;
        std::string tok, ids_s, emb_path;
        std::vector<std::string> img_keys;
        bool sampled = false;
        int nocache = 0;
        while (ss >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                if (geni && emb_path.empty()) emb_path = tok;   // GENI <max_new> [key=value ...] <file> <ids>
                else ids_s = tok;
                continue;
            }
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "img_keys") {   // each image's hash: the conversation reuse matches pictures by them
                std::stringstream ks(v);
                for (std::string kk; std::getline(ks, kk, ',');) img_keys.push_back(kk);
                continue;
            }
            if (k == "nocache") { nocache = std::atoi(v.c_str()); continue; }   // debugging: 1 = a cold read, 2 = the single snapshot of before
            if (k == "temperature") { sp.temperature = (float) std::atof(v.c_str()); sampled = sp.temperature > 0; }
            else if (k == "top_p") sp.top_p = (float) std::atof(v.c_str());
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = (float) std::atof(v.c_str());
            else if (k == "seed") sp.seed = (uint64_t) std::atoll(v.c_str());
        }
        if (!sampled) sp.temperature = 0.0f;
        std::vector<int32_t> prompt;
        if (max_new <= 0 || !parse_ids(ids_s, prompt)) {
            std::printf("ERR malformed GEN line\n");
            std::fflush(stdout);
            continue;
        }
        if (const char* dump = std::getenv("STRATA_NEMO_DUMP_PROMPT")) {   // debugging: the ids the server rendered
            if (FILE* f = std::fopen(dump, "ab")) {
                for (size_t i = 0; i < prompt.size(); ++i) std::fprintf(f, i ? ",%d" : "%d", prompt[i]);
                std::fprintf(f, "\n");
                std::fclose(f);
            }
        }
        std::mt19937_64 rng(sp.seed ? sp.seed : (uint64_t) Clock::now().time_since_epoch().count());
        std::string err;
        const auto t0 = Clock::now();
        // images (GENI): each <img><image>...</img> run of the prompt takes one record of the file, in order; a GEN
        // prompt's runs (none from the server) stay plain token rows
        const std::vector<std::pair<int64_t, int64_t>> runs = geni ? image_runs(prompt)
                                                                   : std::vector<std::pair<int64_t, int64_t>>();
        {
            std::vector<float> rows;
            std::vector<int64_t> counts;
            bool img_ok = !geni || read_sve(emb_path, n_embd, rows, counts, err);
            if (img_ok && counts.size() != runs.size()) {
                err = "the prompt has " + std::to_string(runs.size()) + " images and the embeddings file " +
                      std::to_string(counts.size());
                img_ok = false;
            }
            for (size_t i = 0; img_ok && i < runs.size(); ++i)
                if (counts[i] != runs[i].second) {
                    err = "image " + std::to_string(i + 1) + " has " + std::to_string(runs[i].second) +
                          " <image> tokens in the prompt and " + std::to_string(counts[i]) + " rows in the file";
                    img_ok = false;
                }
            if (!img_ok || !m.set_image_rows(rows.data(), runs, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
        }
        // an image's key ("?" when the server sent none: such a picture never matches)
        const auto key_of = [&](size_t i) { return i < img_keys.size() ? img_keys[i] : std::string("?"); };
        // reuse: the previous prompt is a prefix of this one, its pictures the same ones
        size_t reused = 0;
        const bool cc_on = g_cc.enabled() && nocache == 0;
        const strata::core::ConvImages cimg{&runs, &img_keys};
        bool cont = false;
        if (!cc_on && nocache != 1) {
            cont = !snap_prompt.empty() && m.snapshot_pos() == (int64_t) snap_prompt.size() &&
                   snap_prompt.size() <= prompt.size() &&
                   std::equal(snap_prompt.begin(), snap_prompt.end(), prompt.begin());
            for (size_t i = 0; cont && i < snap_keys.size(); ++i)
                cont = snap_keys[i] != "?" && key_of(i) == snap_keys[i];
        }
        // the snapshot goes at the end of the prompt's last COMPLETE message (after its <|im_end|> and newline), not at
        // the prompt's end: the next turn re-renders this turn's generation prompt (<|im_start|>assistant\n<think>\n)
        // as the answer it became, so only the messages before it are a prefix the next prompt will share
        int64_t boundary = (int64_t) prompt.size();
        for (int64_t i = (int64_t) prompt.size() - 1; i >= 0; --i)
            if (prompt[(size_t) i] == 11) {   // <|im_end|>
                boundary = i + 1 + ((size_t) i + 1 < prompt.size() && prompt[(size_t) i + 1] == 1010 ? 1 : 0);   // + "\n"
                break;
            }
        bool ok = true;
        int64_t at = 0;
        if (cc_on) {
            // conversation cache: the deepest checkpoint (live, or a parked conversation's) that is a prefix of the prompt
            g_cc.sync(m);
            const auto ch = g_cc.choose(prompt, cimg);
            if (ch.src == strata::core::ConversationCache::Choice::Live) {
                // a branch before the deepest checkpoint overwrites cells the other branch needs: park it first
                if (g_cc.deepest_live() > ch.len) g_cc.park_current(m, 0);
                ok = g_cc.restore_live(m, ch.idx, logits, err);
            } else {
                g_cc.park_current(m, ch.src == strata::core::ConversationCache::Choice::Park ? ch.id : 0);
                if (ch.src == strata::core::ConversationCache::Choice::Park) ok = g_cc.unpark(m, ch, logits, err);
                else { m.reset(); g_cc.clear_live(); }
            }
            if (ok && ch.src != strata::core::ConversationCache::Choice::None) {
                reused = (size_t) ch.len;
                at = ch.len;
            }
            if (ok && ch.src != strata::core::ConversationCache::Choice::None)
                std::fprintf(stderr, "strata-nemotron: conv-cache: %s checkpoint at %lld of %zu tokens, restore %.1f ms\n",
                             ch.src == strata::core::ConversationCache::Choice::Live ? "live" : "parked",
                             (long long) ch.len, prompt.size(), g_cc.last_restore_ms);
        } else if (cont && m.restore()) {
            reused = snap_prompt.size();
            at = (int64_t) reused;
        } else {
            m.reset();
        }
        // STRATA_NEMO_GPU_TOPK: the first token's candidates come from head_out; the full logits are not read back
        const bool gtk_first = gpu_topk_k(sp) > 0 && g_spec > 0 && m.has_mtp();
        bool host_logits = !gtk_first;
        const auto feed = [&](int64_t upto) {   // the prompt from `at` to `upto`, a checkpoint every g_every tokens
            while (ok && upto > at) {
                int64_t to = upto;
                if (cc_on && g_every > 0) to = std::min(upto, (at / g_every + 1) * g_every);
                ok = m.forward(prompt.data() + at, to - at, gtk_first ? nullptr : logits.data(), err);
                at = to;
                if (ok && cc_on && g_every > 0 && at < upto && at % g_every == 0)
                    ok = g_cc.take(m, prompt.data(), cimg, strata::core::ConversationCache::Chunk, err);
            }
        };
        if (ok && boundary > at) {
            feed(boundary);
            if (ok && cc_on) ok = g_cc.take(m, prompt.data(), cimg, strata::core::ConversationCache::Message, err);
            else if (ok) {
                m.snapshot();
                snap_prompt.assign(prompt.begin(), prompt.begin() + boundary);
                snap_keys.clear();   // the pictures the snapshot has read
                for (size_t i = 0; i < runs.size() && runs[i].first + runs[i].second <= boundary; ++i)
                    snap_keys.push_back(key_of(i));
            }
        }
        feed((int64_t) prompt.size());
        if (ok && at == (int64_t) reused && reused == prompt.size()) {
            // nothing was fed: the logits come from the checkpoint (conversation cache restores them itself)
            if (!cc_on) logits = m.snapshot_logits();
            host_logits = true;
        }
        if (!ok) {
            snap_prompt.clear();
            g_cc.clear_live();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (const char* dl = std::getenv("STRATA_NEMO_DUMP_LOGITS")) {   // debugging: the logits after the prompt
            if (FILE* f = std::fopen(dl, "ab")) {
                std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, f);
                std::fclose(f);
            }
        }
        const double prompt_ms = ms_since(t0);
        std::printf("REUSED %zu\n", reused);
        const auto t1 = Clock::now();
        const auto sp0 = m.stats();
        const uint64_t spec_w0 = sp0.spec_windows, spec_a0 = sp0.spec_accepted, spec_d0 = sp0.spec_drafted;
        long long produced = 0;
        const char* finish = "length";
        std::vector<int32_t> gen_ids, got;   // emitted tokens; the spec window's kept tokens not yet emitted (reversed)
        if (g_spec > 0 && m.has_mtp()) {
            // speculation: the first token from the prompt's logits, then windows of MTP drafts verified at once
            const std::function<int32_t(const float*)> pick = [&](const float* lg) { return sample(lg, n_vocab, sp, rng); };
            const std::function<int32_t(const float*, const int32_t*, int)> pick_tk =
                [&](const float* v, const int32_t* ids, int k) { return sample_topk(v, ids, k, sp, rng); };
            const int gk = gpu_topk_k(sp);
            int32_t id;
            if (gk && !host_logits) {
                std::vector<float> tv((size_t) gk);
                std::vector<int32_t> ti((size_t) gk);
                if (!m.topk_head(gk, tv.data(), ti.data(), err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }
                id = sample_topk(tv.data(), ti.data(), gk, sp, rng);
            } else {
                id = sample(logits.data(), n_vocab, sp, rng);
            }
            for (;;) {
                std::printf("T %d\n", id);
                std::fflush(stdout);
                ++produced;
                gen_ids.push_back(id);
                if (std::find(eos.begin(), eos.end(), id) != eos.end()) { finish = "stop"; break; }
                if (produced >= max_new) break;
                if (got.empty()) {
                    if (stop_req.load()) { finish = "cancel"; break; }
                    if (m.pos() + 1 > m.max_context()) { finish = "length"; break; }
                    int64_t nd = std::min<int64_t>(g_spec, m.max_context() - m.pos() - 1);
                    // conversation cache: a window must not feed tokens past max_new (the next prompt would not
                    // have them): at most max_new - produced more tokens are emitted, a window emits nd + 1
                    if (cc_on) nd = std::min<int64_t>(nd, max_new - produced - 1);
                    if (nd <= 0 && cc_on) {   // the last token: one plain step
                        if (!m.forward(&id, 1, logits.data(), err)) {
                            std::printf("ERR %s\n", err.c_str());
                            ok = false;
                            break;
                        }
                        got.assign(1, sample(logits.data(), n_vocab, sp, rng));
                    } else if (!m.spec_step(id, (int) nd, got, err, sp.temperature > 0 ? &pick : nullptr, gk ? &pick_tk : nullptr, gk)) {
                        std::printf("ERR %s\n", err.c_str());
                        ok = false;
                        break;
                    } else {
                        std::reverse(got.begin(), got.end());
                    }
                }
                id = got.back();
                got.pop_back();
            }
        }
        while (ok && produced < max_new && !(g_spec > 0 && m.has_mtp())) {
            if (stop_req.load()) { finish = "cancel"; break; }
            const int32_t id = sample(logits.data(), n_vocab, sp, rng);
            std::printf("T %d\n", id);
            std::fflush(stdout);
            ++produced;
            gen_ids.push_back(id);
            if (std::find(eos.begin(), eos.end(), id) != eos.end()) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (m.pos() + 1 > m.max_context()) { finish = "length"; break; }
            if (!m.forward(&id, 1, logits.data(), err)) {
                std::printf("ERR %s\n", err.c_str());
                ok = false;
                break;
            }
        }
        if (!ok) { std::fflush(stdout); continue; }
        const double decode_ms = ms_since(t1);
        const auto s2 = m.stats();
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %llu %llu\n", produced, prompt.size(), prompt_ms, decode_ms,
                    finish, reused, (unsigned long long) s2.expert_hits,
                    (unsigned long long) (s2.expert_hits + s2.expert_misses));
        std::fflush(stdout);
        if (cc_on) {
            // after the answer: the next turn that extends it verbatim resumes here.  The tokens the model has fed are
            // the prompt and a prefix of the generated ones (emitted, then the kept ones not emitted yet)
            std::vector<int32_t> seq = prompt;
            seq.insert(seq.end(), gen_ids.begin(), gen_ids.end());
            seq.insert(seq.end(), got.rbegin(), got.rend());
            if (m.pos() >= (int64_t) prompt.size() && m.pos() <= (int64_t) seq.size()) {
                std::string cerr;
                if (!g_cc.take(m, seq.data(), cimg, strata::core::ConversationCache::Answer, cerr)) {
                    std::fprintf(stderr, "strata-nemotron: conv-cache: %s\n", cerr.c_str());
                    g_cc.clear_live();
                }
            }
        }
        if (s2.spec_windows > spec_w0)
            std::fprintf(stderr, "strata-nemotron: drafts accepted %llu of %llu (%llu windows)\n",
                         (unsigned long long) (s2.spec_accepted - spec_a0), (unsigned long long) (s2.spec_drafted - spec_d0),
                         (unsigned long long) (s2.spec_windows - spec_w0));
        std::fprintf(stderr, "strata-nemotron: prompt %zu tokens (%zu reused) in %.0f ms (%.0f tok/s), %lld generated in "
                             "%.0f ms (%.1f tok/s), expert hit rate %.1f%% (%lld slots a layer, K/V %lld cells)\n",
                     prompt.size(), reused, prompt_ms, (prompt.size() - reused) / (prompt_ms / 1000.0), produced,
                     decode_ms, produced > 1 ? (produced - 1) / (decode_ms / 1000.0) : 0.0,
                     100.0 * s2.expert_hits / std::max<double>(1.0, (double) (s2.expert_hits + s2.expert_misses)),
                     (long long) s2.slots_per_layer, (long long) s2.kv_cells);
        {
            const std::string pr = m.profile_report(true);
            if (!pr.empty()) std::fprintf(stderr, "%s\n", pr.c_str());
        }
        if (quit_req.load()) break;
    }
    return 0;
}

/// --bench IDS: the first --ctx ids of the file as the prompt, then --gen greedy tokens; the speeds, the expert hit
/// rate, the profile (STRATA_NEMO_PROFILE=1) and the generated ids (a regression check: an exact optimization keeps them).
int run_bench(strata::core::NemoModel& m, const char* path, int64_t ctx, int64_t gen, int probe, int spec,
              float temp) {
    std::vector<int32_t> ids;
    {
        FILE* f = std::fopen(path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
        long v;
        while ((int64_t) ids.size() < ctx && std::fscanf(f, "%ld", &v) == 1) ids.push_back((int32_t) v);
        std::fclose(f);
    }
    const int64_t n_vocab = m.geometry().n_vocab;
    std::vector<float> lg((size_t) n_vocab);
    std::string err;
    m.reset();
    auto t0 = Clock::now();
    // STRATA_NEMO_GPU_TOPK with --temp: the first token's candidates come from the device too
    const bool bench_gtk = spec > 0 && temp > 0 && m.has_mtp() && std::getenv("STRATA_NEMO_GPU_TOPK") && std::atoi(std::getenv("STRATA_NEMO_GPU_TOPK"));
    if (!m.forward(ids.data(), (int64_t) ids.size(), bench_gtk ? nullptr : lg.data(), err)) { std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
    const double pms = ms_since(t0);
    m.profile_report(true);
    const auto s0 = m.stats();
    std::vector<int32_t> out;
    t0 = Clock::now();
    if (spec > 0) {
        // greedy speculation: spec_step until `gen` tokens
        // --temp T: sampled (top_p 0.95, top_k 64, seed 1), else greedy
        Sampling sp;
        sp.temperature = temp;
        sp.top_p = 0.95f;
        sp.top_k = 64;
        std::mt19937_64 rng(1);
        const std::function<int32_t(const float*)> pick = [&](const float* l) { return sample(l, n_vocab, sp, rng); };
        const std::function<int32_t(const float*, const int32_t*, int)> pick_tk =
            [&](const float* v, const int32_t* i2, int k) { return sample_topk(v, i2, k, sp, rng); };
        const int gk = bench_gtk ? gpu_topk_k(sp) : 0;
        std::vector<float> tv(1024);
        std::vector<int32_t> ti(1024);
        if (gk && !m.topk_head(gk, tv.data(), ti.data(), err)) { std::fprintf(stderr, "topk: %s\n", err.c_str()); return 1; }
        int32_t t = gk ? sample_topk(tv.data(), ti.data(), gk, sp, rng) : temp > 0 ? pick(lg.data()) : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        std::vector<int32_t> got;
        while ((int64_t) out.size() < gen) {
            out.push_back(t);
            if (!m.spec_step(t, spec, got, err, temp > 0 ? &pick : nullptr, gk ? &pick_tk : nullptr, gk)) { std::fprintf(stderr, "spec: %s\n", err.c_str()); return 1; }
            for (size_t k = 0; k + 1 < got.size() && (int64_t) out.size() < gen; ++k) out.push_back(got[k]);
            t = got.back();
        }
        out.resize((size_t) gen);
        const double dms = ms_since(t0);
        m.profile_report(true);
        const auto s1 = m.stats();
        const double h = (double) (s1.expert_hits - s0.expert_hits), mi = (double) (s1.expert_misses - s0.expert_misses);
        std::printf("bench: expert hits %llu, misses %llu, skipped %llu (%llu with no slot left), %llu misses through the ring; prefetch issued %llu, used %llu, wasted %llu\n",
                    (unsigned long long) (s1.expert_hits - s0.expert_hits),
                    (unsigned long long) (s1.expert_misses - s0.expert_misses),
                    (unsigned long long) (s1.expert_skips - s0.expert_skips),
                    (unsigned long long) (s1.expert_forced_skips - s0.expert_forced_skips),
                    (unsigned long long) (s1.expert_overflow - s0.expert_overflow),
                    (unsigned long long) (s1.prefetch_issued - s0.prefetch_issued),
                    (unsigned long long) (s1.prefetch_used - s0.prefetch_used),
                    (unsigned long long) (s1.prefetch_wasted - s0.prefetch_wasted));
        std::printf("bench: spec %d: %llu windows, %.2f of %d drafts accepted a window (%.0f%%), %.2f tokens a window\n",
                    spec, (unsigned long long) (s1.spec_windows - s0.spec_windows),
                    (double) (s1.spec_accepted - s0.spec_accepted) / (double) (s1.spec_windows - s0.spec_windows), spec,
                    100.0 * (double) (s1.spec_accepted - s0.spec_accepted) / (double) (s1.spec_drafted - s0.spec_drafted),
                    1.0 + (double) (s1.spec_accepted - s0.spec_accepted) / (double) (s1.spec_windows - s0.spec_windows));
        std::printf("bench: prompt %zu tokens in %.0f ms (%.1f tok/s); %lld generated in %.0f ms (%.2f tok/s), expert hit %.1f%%\n",
                    ids.size(), pms, ids.size() * 1000.0 / pms, (long long) gen, dms, gen * 1000.0 / dms, 100 * h / (h + mi));
        std::printf("ids:");
        for (int32_t v : out) std::printf(" %d", v);
        std::printf("\n");
        return 0;
    }
    std::vector<double> step;
    std::vector<std::vector<int32_t>> drafts;   // --mtp-probe: the chain drafted before feeding out[i]
    double draft_ms = 0;
    for (int64_t i = 0; i < gen; ++i) {
        const int32_t t = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        out.push_back(t);
        if (probe > 0 && m.has_mtp()) {
            std::vector<int32_t> d((size_t) probe, -1);
            const auto td = Clock::now();
            if (!m.mtp_draft(t, probe, d.data(), err) && !err.empty()) { std::fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
            draft_ms += ms_since(td);
            drafts.push_back(d);
        }
        const auto ts = Clock::now();
        if (!m.forward(&t, 1, lg.data(), err)) { std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1; }
        step.push_back(ms_since(ts));
    }
    if (!step.empty()) {
        std::vector<double> q(step);
        std::sort(q.begin(), q.end());
        std::printf("bench: step ms p10 %.2f, p50 %.2f, p90 %.2f, max %.2f\n", q[q.size() / 10], q[q.size() / 2],
                    q[q.size() * 9 / 10], q.back());
    }
    const double dms = ms_since(t0);
    const auto s1 = m.stats();
    const double h = (double) (s1.expert_hits - s0.expert_hits), mi = (double) (s1.expert_misses - s0.expert_misses);
    std::printf("bench: prefetch issued %llu, used %llu, wasted %llu; skipped %llu\n",
                (unsigned long long) (s1.prefetch_issued - s0.prefetch_issued),
                (unsigned long long) (s1.prefetch_used - s0.prefetch_used),
                (unsigned long long) (s1.prefetch_wasted - s0.prefetch_wasted),
                (unsigned long long) (s1.expert_skips - s0.expert_skips));
    std::printf("bench: prompt %zu tokens in %.0f ms (%.1f tok/s); %lld generated in %.0f ms (%.2f tok/s), expert hit %.1f%%\n",
                ids.size(), pms, ids.size() * 1000.0 / pms, (long long) gen, dms, gen * 1000.0 / dms, 100 * h / (h + mi));
    const std::string pr = m.profile_report(true);
    if (!pr.empty()) std::printf("%s\n", pr.c_str());
    if (!drafts.empty()) {
        // drafts[i][j] guesses out[i + 1 + j]; a chain is accepted up to its first miss
        std::vector<int64_t> at_depth((size_t) probe + 1, 0);
        int64_t n = 0, sum = 0;
        for (size_t i = 0; i + (size_t) probe < out.size(); ++i) {
            int k = 0;
            while (k < probe && drafts[i][(size_t) k] == out[i + 1 + (size_t) k]) ++k;
            ++at_depth[(size_t) k];
            sum += k;
            ++n;
        }
        std::printf("mtp probe: %lld chains of %d, mean accepted %.2f, draft %.2f ms a chain; accepted-length "
                    "histogram:", (long long) n, probe, (double) sum / (double) n, draft_ms / (double) drafts.size());
        for (int k = 0; k <= probe; ++k) std::printf(" %d:%lld", k, (long long) at_depth[(size_t) k]);
        std::printf("\n");
    }
    std::printf("ids:");
    for (int32_t t : out) std::printf(" %d", t);
    std::printf("\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool chunk_given = false;
    strata::core::NemoOptions o;
    bool serve = false;
    const char* ppl = nullptr;
    const char* bench = nullptr;
    int64_t ctx = 1024, chunks = 16, gen = 256;
    int probe = 0, spec = 0;
    float bench_temp = 0.0f;
    int cc_n = 0;
    int64_t cc_mib = 0;
    const char* spec_list = nullptr;
    const char* minp_list = nullptr;
    bool bos_first = true, ppl_decode = false, ctx_given = false;
    const char* ll_dump = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](void) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--gguf") o.gguf = next();
        else if (a == "--experts-gguf") o.experts_gguf = next();
        else if (a == "--max-context") { o.max_context = std::atoll(next()); ctx_given = true; }
        else if (a == "--chunk") { o.chunk = std::atoll(next()); chunk_given = true; }
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoll(next());
        else if (a == "--vram-limit-mib") o.vram_limit_mib = std::atoll(next());
        else if (a == "--expert-slots") o.expert_slots = std::atoll(next());
        else if (a == "--kv-base") o.kv_base = std::atoll(next());
        else if (a == "--serve") serve = true;
        else if (a == "--vision") {}   // the server adds it with a "vision" section: GENI is always taken
        else if (a == "--ppl") ppl = next();
        else if (a == "--bench") bench = next();
        else if (a == "--gen") gen = std::atoll(next());
        else if (a == "--mtp") o.mtp = true;
        else if (a == "--mtp-gguf") o.mtp_gguf = next();   // the MTP block from another GGUF (with --spec)
        else if (a == "--mtp-probe") { probe = std::atoi(next()); o.mtp = true; }
        else if (a == "--spec") { spec = std::atoi(next()); g_spec = spec; o.mtp = true; }
        else if (a == "--tail-skip") setenv("STRATA_NEMO_TAIL_SKIP", next(), 1);
        else if (a == "--prefetch") setenv("STRATA_NEMO_PREFETCH", next(), 1);
        else if (a == "--prefetch-rank") setenv("STRATA_NEMO_PREFETCH_RANK", next(), 1);
        else if (a == "--spec-min-p") setenv("STRATA_NEMO_SPEC_MIN_P", next(), 1);
        else if (a == "--mtp-slots") setenv("STRATA_NEMO_MTP_SLOTS", next(), 1);
        else if (a == "--evict") setenv("STRATA_NEMO_EVICT", next(), 1);
        else if (a == "--skip-w") setenv("STRATA_NEMO_SKIP_W", next(), 1);
        else if (a == "--prompt-skip-w") setenv("STRATA_NEMO_PROMPT_SKIP_W", next(), 1);
        else if (a == "--prompt-skip-max") setenv("STRATA_NEMO_PROMPT_SKIP_MAX", next(), 1);
        else if (a == "--prompt-cache") cc_n = std::atoi(next());
        else if (a == "--prompt-cache-every") g_every = std::atoll(next());
        else if (a == "--conversation-cache-mib") cc_mib = std::atoll(next());
        else if (a == "--temp") bench_temp = (float) std::atof(next());
        else if (a == "--spec-list") { spec_list = next(); o.mtp = true; }
        else if (a == "--min-p-list") minp_list = next();
        else if (a == "--ctx") ctx = std::atoll(next());
        else if (a == "--chunks") chunks = std::atoll(next());
        else if (a == "--no-bos") bos_first = false;
        else if (a == "--ppl-decode") ppl_decode = true;
        else if (a == "--ll-dump") ll_dump = next();
        else std::fprintf(stderr, "strata-nemotron: ignoring %s\n", a.c_str());
    }
    if (o.gguf.empty() || (!serve && !ppl && !bench)) {
        std::fprintf(stderr, "usage: %s --gguf FILE [--max-context N] [--chunk N] [--vram-reserve-mib N] "
                             "[--expert-slots N] [--vram-limit-mib N] (--serve | --ppl IDS [--ctx N] [--chunks N] | --bench IDS [--ctx N] [--gen N])\n", argv[0]);
        return 2;
    }
    if ((ppl || bench) && !ctx_given) o.max_context = std::max<int64_t>(ctx + gen, 1024);
    {
        // a card (or --vram-limit-mib) under 20 GiB: the small-card profile, unless set by hand - an 8192-row chunk
        // and a ring of 48 expert blobs for the prompt (~140 MiB; 384: ~1.1 GiB).  At 15,800 MiB with MTP that leaves
        // 40 expert slots a layer: prefill 4,600 tok/s at 32K (2,000 with a 2048-row chunk and 48 slots), decode
        // 61-93 tok/s with MTP.  Use the precise decode (--prefetch 1): with few slots the fast one costs PPL x1.17
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const int64_t card = o.vram_limit_mib > 0 ? o.vram_limit_mib : (int64_t) (tot >> 20);
        if (card < 20480) {
            if (!chunk_given) o.chunk = 8192;
            if (!std::getenv("STRATA_NEMO_RING")) setenv("STRATA_NEMO_RING", "48", 1);
            std::fprintf(stderr, "strata-nemotron: a %lld MiB card: the small-card profile (chunk %lld, prompt ring %s)\n",
                         (long long) card, (long long) o.chunk, std::getenv("STRATA_NEMO_RING"));
            if (const char* pf = std::getenv("STRATA_NEMO_PREFETCH"); pf && std::atoi(pf) == 0)
                std::fprintf(stderr, "strata-nemotron: --prefetch 0 on a small card skips most expert misses: decode PPL "
                                     "x1.17 at 16 GB (x1.01 with --prefetch 1)\n");
        }
    }
    strata::core::NemoModel m;
    std::string err;
    const auto t0 = Clock::now();
    if (!m.load(o, err)) {
        std::fprintf(stderr, "strata-nemotron: %s\n", err.c_str());
        if (serve) std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const auto st = m.stats();
    std::fprintf(stderr, "strata-nemotron: loaded in %.1f s: dense %.2f GiB in VRAM, %lld expert slots per MoE layer "
                         "(%.1f GiB), K/V %lld cells (grows to %lld), experts %s\n", ms_since(t0) / 1000.0,
                 st.dense_gib, (long long) st.slots_per_layer, st.expert_cache_gib, (long long) st.kv_cells,
                 (long long) m.max_context(), st.pinned ? "page-locked in place" : "NOT page-locked");
    if (serve && (cc_n > 0 || cc_mib > 0)) {
        if (cc_mib > 0 && cc_n < 2) cc_n = 2;   // parking needs the checkpoints
        std::string cerr;
        const auto tc = Clock::now();
        if (!g_cc.init(m, cc_n, cc_mib, cerr))
            std::fprintf(stderr, "strata-nemotron: conversation cache off: %s\n", cerr.c_str());
        else
            std::fprintf(stderr, "strata-nemotron: conversation cache: %d checkpoints every %lld tokens, park %lld MiB "
                                 "(pools allocated in %.1f s)\n", cc_n, (long long) g_every, (long long) cc_mib,
                         ms_since(tc) / 1000.0);
    }
    if (bench) {
        // a sweep: every (spec, min_p) of --spec-list / --min-p-list over every file of a comma-separated --bench
        std::vector<std::string> files;
        {
            std::string b = bench, cur;
            for (char ch : b) { if (ch == ',') { files.push_back(cur); cur.clear(); } else cur += ch; }
            if (!cur.empty()) files.push_back(cur);
        }
        auto list = [](const char* l, std::vector<double> dflt) {
            if (!l) return dflt;
            std::vector<double> v;
            std::stringstream ss(l);
            std::string x;
            while (std::getline(ss, x, ',')) v.push_back(std::atof(x.c_str()));
            return v;
        };
        const std::vector<double> specs = list(spec_list, {(double) spec});
        const std::vector<double> minps = list(minp_list, {-1.0});
        int rc = 0;
        for (double sv : specs)
            for (double mp : minps) {
                if (mp >= 0) m.set_spec_min_p((float) mp);
                for (const std::string& f : files) {
                    std::printf("== spec %d min_p %.2f %s\n", (int) sv, mp, f.c_str());
                    rc |= run_bench(m, f.c_str(), ctx, gen, probe, (int) sv, bench_temp);
                    std::fflush(stdout);
                }
            }
        return rc;
    }
    return ppl ? run_ppl(m, ppl, ctx, chunks, bos_first, ppl_decode, ll_dump) : run_serve(m);
}
