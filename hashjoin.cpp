// ==============================================================================
// Skew-Resilient Parallel Hash Join Benchmark
//
// Pipeline Architecture (following the project methodology slides):
//   1. Data Generation: Seeded relation synthesis (Uniform R, Zipf-skewed S)
//   2. Sequential Reference Join: std::unordered_map oracle for correctness verification
//   3. Two-Pass Radix Partitioning: 5 + 5 bits -> 1024 cache-resident partitions
//   4. Partition Statistics: Max/Avg imbalance computation
//   5. Parallel Build Phase: Open-addressing hash table per partition of R (static OpenMP)
//   6. Parallel Probe Phase: Evaluates 4 probe schedulers:
//        - static    : Static coarse whole-partition scheduling (Paper Fig. 16a baseline)
//        - paper     : Paper-inspired fine-grained dynamic chunking (Balkesen Fig. 16b)
//        - adaptive1 : Cost-formulated adaptive splitting with bounded task budget
//        - adaptive2 : Adaptive tasks + per-worker Chase-Lev deques + cost-gated stealing
//   7. Validation & Metrics: Match count verification against sequential oracle & CSV logging
//
// Build command:
//   g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
// ==============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

using namespace std;

typedef uint32_t u32;
typedef uint64_t u64;

// 2^10 = 1024 partitions (two passes of 5 bits each)
constexpr int NP = 1024;

// Relational tuple: 32-bit key, 32-bit payload
struct Tuple {
    u32 key;
    u32 payload;
};

// Radix hash function (top 10 bits indicate partition ID)
static inline u32 h1(u32 k) {
    return k * 0x9E3779B1u;
}

// Murmur3 32-bit finalizer hash for hash-table slot indexing
static inline u32 fmix(u32 h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// ==============================================================================
// Configuration
// ==============================================================================

struct Cfg {
    size_t n = 5000000;                      // |R|: Build relation tuple count
    size_t ns = 0;                           // |S|: Probe relation tuple count (0 = same as |R|)
    u32 key_range = 1000000;                 // Key domain range [1, key_range]
    vector<double> zs = {0, 0.5, 0.75, 1.0, 1.25, 1.5}; // Probe relation Zipf skew parameters
    vector<int> threads = {1, 2, 4};         // Thread counts to benchmark
    int reps = 5;                            // Number of timed repetitions
    int warmup = 1;                          // Number of untimed warmup runs
    vector<string> methods = {"static", "paper", "adaptive1", "adaptive2"};
    string out = "results.csv";              // Destination CSV file path
    u64 seedR = 123456789ULL;                // Random seed for R
    u64 seedS = 987654321ULL;                // Random seed for S

    // Paper-style fine-grained baseline parameters
    double paper_factor = 2.0;               // Split S_i if |S_i| > factor * average
    u32 paper_chunk = 16384;                 // Fixed chunk size (tuples)

    // Adaptive cost-based scheduler parameters
    double granularity = 4.0;                // Each thread's fair share cut into this many pieces
    u32 min_chunk = 2048;                    // Minimum allowable chunk size (tuples)
    int budget_per_thread = 32;              // Maximum extra tasks = budget_per_thread * threads
    double task_overhead_ns = 5000;          // Per-task overhead used in H(k) = k * overhead
    bool ovh_auto = true;                    // Automatically calibrate task overhead
    bool tau_auto = true;                    // Automatically calibrate tau_steal
    double plan_gate = 1.5;                  // Skip sampling when max/avg |S_i| is below this
    bool expand = false;                     // Many-to-many workload (walk all matching R tuples)
    double r_z = 0.0;                        // Zipf skew of the R (build) relation
    int samples = 128;                       // Sample count per partition for output estimation
    bool gamma_auto = true;                  // Auto-calibrate per-match output cost
    double gamma_ns = 0.5;                   // Fallback per-match output walk cost (ns)
    string init = "lpt";                     // Initial placement: "lpt" (cost-balanced) | "rr"
    int include_build = 0;                   // 1: C_i = alpha|R_i| + beta|S_i|; 0: probe-focused
    double tau_ns = 20000;                   // Work-stealing threshold (ns) if not auto-calibrated
};

static vector<string> split(const string& s, char delimiter = ',') {
    vector<string> tokens;
    string token;
    stringstream ss(s);
    while (getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// ==============================================================================
// Synthetic Data Generation
// ==============================================================================

static vector<Tuple> gen_keys(size_t n, u32 K, double z, u64 seed) {
    mt19937_64 rng(seed);
    vector<Tuple> data(n);

    if (z <= 0.0) {
        // Uniform distribution over [1, K]
        uniform_int_distribution<u32> dist(1, K);
        for (size_t i = 0; i < n; i++) {
            data[i] = {dist(rng), static_cast<u32>(i)};
        }
    } else {
        // Zipf distribution over ranks 1..K: P(r) ~ 1 / r^z
        vector<double> cdf(K);
        double cumulative = 0.0;
        for (u32 r = 1; r <= K; r++) {
            cumulative += pow(static_cast<double>(r), -z);
            cdf[r - 1] = cumulative;
        }

        uniform_real_distribution<double> dist(0.0, cumulative);
        for (size_t i = 0; i < n; i++) {
            double sample = dist(rng);
            u32 rank = static_cast<u32>(
                lower_bound(cdf.begin(), cdf.end(), sample) - cdf.begin()
            ) + 1;
            if (rank > K) {
                rank = K;
            }
            data[i] = {rank, static_cast<u32>(i)};
        }
    }
    return data;
}

// ==============================================================================
// Sequential Reference Join (Correctness Oracle)
// ==============================================================================

static void seq_join(const vector<Tuple>& R, const vector<Tuple>& S,
                     bool expand, u64& matches, u64& checksum) {
    matches = 0;
    checksum = 0;

    if (!expand) {
        // Count-only: hash map key -> match frequency
        unordered_map<u32, u32> map;
        map.reserve(R.size() * 2);
        for (const auto& r : R) {
            map[r.key]++;
        }
        for (const auto& s : S) {
            auto it = map.find(s.key);
            if (it != map.end()) {
                matches += it->second;
            }
        }
    } else {
        // Expand mode: payload checksum = sum of all matching R payloads
        unordered_map<u32, pair<u32, u64>> map;
        map.reserve(R.size() * 2);
        for (const auto& r : R) {
            auto& entry = map[r.key];
            entry.first++;
            entry.second += r.payload;
        }
        for (const auto& s : S) {
            auto it = map.find(s.key);
            if (it != map.end()) {
                matches += it->second.first;
                checksum += it->second.second;
            }
        }
    }
}

// ==============================================================================
// Two-Pass Radix Partitioning (5 + 5 bits -> 1024 partitions)
// ==============================================================================

struct Part {
    unique_ptr<Tuple[]> buf;
    size_t n = 0;
    vector<u32> off; // off has NP + 1 entries indicating slice boundaries

    Tuple* data() const { return buf.get(); }
    size_t sz(int i) const { return off[i + 1] - off[i]; }
};

static void partition2(const vector<Tuple>& in, Part& out) {
    size_t n = in.size();
    int P = omp_get_max_threads();

    if (!out.buf || out.n != n) {
        out.buf.reset(new Tuple[n]);
        out.n = n;
    }
    out.off.assign(NP + 1, 0);

    static unique_ptr<Tuple[]> tmpbuf;
    static size_t tmpcap = 0;
    if (tmpcap < n) {
        tmpbuf.reset(new Tuple[n]);
        tmpcap = n;
    }

    Tuple* tmp = tmpbuf.get();
    Tuple* out_tuples = out.data();

    vector<array<u32, 32>> hist(P);
    vector<array<u32, 32>> pos(P);
    u32 boff[33];

    // Pass 1: Partition into 32 intermediate buckets using top 5 bits
    #pragma omp parallel num_threads(P)
    {
        int tid = omp_get_thread_num();
        size_t lo = n * tid / P;
        size_t hi = n * (tid + 1) / P;

        hist[tid].fill(0);
        for (size_t i = lo; i < hi; i++) {
            hist[tid][h1(in[i].key) >> 27]++;
        }

        #pragma omp barrier

        #pragma omp single
        {
            u32 running = 0;
            for (int b = 0; b < 32; b++) {
                boff[b] = running;
                for (int q = 0; q < P; q++) {
                    pos[q][b] = running;
                    running += hist[q][b];
                }
            }
            boff[32] = running;
        }

        for (size_t i = lo; i < hi; i++) {
            u32 bucket = h1(in[i].key) >> 27;
            tmp[pos[tid][bucket]++] = in[i];
        }
    }

    // Pass 2: Partition each 32-bucket slice into 32 partitions (next 5 bits)
    size_t bigthr = (P > 1) ? (n / (2 * static_cast<size_t>(P))) : static_cast<size_t>(-1);
    vector<array<u32, 32>> bh(P);
    vector<array<u32, 32>> bp(P);
    vector<int> smalls;

    for (int b = 0; b < 32; b++) {
        u32 lo0 = boff[b];
        u32 len = boff[b + 1] - boff[b];

        if (len <= bigthr) {
            smalls.push_back(b);
            continue;
        }

        // Oversized buckets are processed concurrently by all threads
        #pragma omp parallel num_threads(P)
        {
            int tid = omp_get_thread_num();
            size_t lo = lo0 + static_cast<size_t>(len) * tid / P;
            size_t hi = lo0 + static_cast<size_t>(len) * (tid + 1) / P;

            bh[tid].fill(0);
            for (size_t i = lo; i < hi; i++) {
                bh[tid][(h1(tmp[i].key) >> 22) & 31]++;
            }

            #pragma omp barrier

            #pragma omp single
            {
                u32 running = lo0;
                for (int s2 = 0; s2 < 32; s2++) {
                    out.off[b * 32 + s2] = running;
                    for (int q = 0; q < P; q++) {
                        bp[q][s2] = running;
                        running += bh[q][s2];
                    }
                }
            }

            for (size_t i = lo; i < hi; i++) {
                u32 sub_bucket = (h1(tmp[i].key) >> 22) & 31;
                out_tuples[bp[tid][sub_bucket]++] = tmp[i];
            }
        }
    }

    // Standard small buckets are processed dynamically across threads
    #pragma omp parallel for schedule(dynamic, 1) num_threads(P)
    for (size_t k = 0; k < smalls.size(); k++) {
        int b = smalls[k];
        u32 lo = boff[b];
        u32 hi = boff[b + 1];
        u32 counts[32] = {0};

        for (u32 i = lo; i < hi; i++) {
            counts[(h1(tmp[i].key) >> 22) & 31]++;
        }

        u32 running = lo;
        u32 q[32];
        for (int s2 = 0; s2 < 32; s2++) {
            out.off[b * 32 + s2] = running;
            q[s2] = running;
            running += counts[s2];
        }

        for (u32 i = lo; i < hi; i++) {
            u32 sub_bucket = (h1(tmp[i].key) >> 22) & 31;
            out_tuples[q[sub_bucket]++] = tmp[i];
        }
    }

    out.off[NP] = static_cast<u32>(n);
}

// ==============================================================================
// Local Open-Addressing Hash Tables
// ==============================================================================

struct Slot {
    u32 key;
    u32 cnt;
};

struct SlotE {
    u32 key;
    u32 cnt;
    u32 start;
    u32 fill;
};

struct HTs {
    unique_ptr<Slot[]> arena;
    unique_ptr<SlotE[]> arenaE;
    unique_ptr<u32[]> pl;
    size_t cap_slots = 0;
    size_t cap_pl = 0;
    vector<size_t> off;
    vector<u32> cap;
    vector<u32> plo;
    bool expand = false;
};

static inline u32 cap_for(size_t len) {
    u32 c = 16;
    while (c < 2 * len) {
        c <<= 1;
    }
    return c;
}

static void build_one(const Tuple* r, size_t len, Slot* T, u32 cap) {
    memset(T, 0, sizeof(Slot) * cap);
    u32 mask = cap - 1;
    for (size_t i = 0; i < len; i++) {
        u32 k = r[i].key;
        u32 j = fmix(k) & mask;
        while (T[j].key != 0 && T[j].key != k) {
            j = (j + 1) & mask;
        }
        T[j].key = k;
        T[j].cnt++;
    }
}

static void build_one_e(const Tuple* r, size_t len, SlotE* T, u32 cap, u32* pl) {
    memset(T, 0, sizeof(SlotE) * cap);
    u32 mask = cap - 1;
    for (size_t i = 0; i < len; i++) {
        u32 k = r[i].key;
        u32 j = fmix(k) & mask;
        while (T[j].key != 0 && T[j].key != k) {
            j = (j + 1) & mask;
        }
        T[j].key = k;
        T[j].cnt++;
    }

    u32 running = 0;
    for (u32 j = 0; j < cap; j++) {
        if (T[j].key != 0) {
            T[j].start = running;
            running += T[j].cnt;
        }
    }

    for (size_t i = 0; i < len; i++) {
        u32 k = r[i].key;
        u32 j = fmix(k) & mask;
        while (T[j].key != k) {
            j = (j + 1) & mask;
        }
        pl[T[j].start + T[j].fill++] = r[i].payload;
    }
}

static void build_all(const Part& R, HTs& H, bool expand) {
    H.expand = expand;
    H.off.assign(NP + 1, 0);
    H.cap.assign(NP, 0);
    H.plo.assign(R.off.begin(), R.off.end());

    size_t total_slots = 0;
    for (int i = 0; i < NP; i++) {
        H.cap[i] = cap_for(R.sz(i));
        H.off[i] = total_slots;
        total_slots += H.cap[i];
    }
    H.off[NP] = total_slots;

    if (!expand) {
        if (total_slots > H.cap_slots) {
            H.arena.reset(new Slot[total_slots]);
            H.cap_slots = total_slots;
        }
    } else {
        if (total_slots > H.cap_slots) {
            H.arenaE.reset(new SlotE[total_slots]);
            H.cap_slots = total_slots;
        }
        if (R.n > H.cap_pl) {
            H.pl.reset(new u32[R.n]);
            H.cap_pl = R.n;
        }
    }

    // Step 5 of proposal: OpenMP parallel static build loop
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NP; i++) {
        if (!expand) {
            build_one(R.data() + R.off[i], R.sz(i), H.arena.get() + H.off[i], H.cap[i]);
        } else {
            build_one_e(R.data() + R.off[i], R.sz(i), H.arenaE.get() + H.off[i],
                        H.cap[i], H.pl.get() + H.plo[i]);
        }
    }
}

static inline u64 probe_range(const HTs& H, int p, const Tuple* s, size_t len) {
    const Slot* T = H.arena.get() + H.off[p];
    u32 mask = H.cap[p] - 1;
    u64 match_count = 0;

    for (size_t i = 0; i < len; i++) {
        u32 k = s[i].key;
        u32 j = fmix(k) & mask;
        while (true) {
            const Slot& entry = T[j];
            if (entry.key == k) {
                match_count += entry.cnt;
                break;
            }
            if (entry.key == 0) {
                break;
            }
            j = (j + 1) & mask;
        }
    }
    return match_count;
}

static inline void probe_expand(const HTs& H, int p, const Tuple* s, size_t len,
                                u64& matches, u64& checksum) {
    const SlotE* T = H.arenaE.get() + H.off[p];
    const u32* pl = H.pl.get() + H.plo[p];
    u32 mask = H.cap[p] - 1;

    for (size_t i = 0; i < len; i++) {
        u32 k = s[i].key;
        u32 j = fmix(k) & mask;
        while (true) {
            const SlotE& entry = T[j];
            if (entry.key == k) {
                matches += entry.cnt;
                const u32* payloads = pl + entry.start;
                u64 sum = 0;
                for (u32 x = 0; x < entry.cnt; x++) {
                    sum += payloads[x];
                }
                checksum += sum;
                break;
            }
            if (entry.key == 0) {
                break;
            }
            j = (j + 1) & mask;
        }
    }
}

static inline void probe_task(const HTs& H, int p, const Tuple* s, size_t len,
                              u64& matches, u64& checksum) {
    if (H.expand) {
        probe_expand(H, p, s, len, matches, checksum);
    } else {
        matches += probe_range(H, p, s, len);
    }
}

// ==============================================================================
// Metrics & Statistics
// ==============================================================================

struct alignas(64) ThreadStats {
    double fin = 0;
    u64 matches = 0;
    u64 checksum = 0;
    u64 tasks = 0;
    u64 steal_attempts = 0;
    u64 successful_steals = 0;
};

struct RunStats {
    u64 matches = 0;
    u64 checksum = 0;
    u64 tasks = 0;
    u64 steal_attempts = 0;
    u64 successful_steals = 0;
    double lambda = 1.0;
    double probe_s = 0.0;
};

static RunStats finalize(const vector<ThreadStats>& stats, double wall_time) {
    RunStats result;
    double max_time = 0.0;
    double sum_time = 0.0;
    int P = static_cast<int>(stats.size());

    for (const auto& ts : stats) {
        max_time = max(max_time, ts.fin);
        sum_time += ts.fin;
        result.matches += ts.matches;
        result.checksum += ts.checksum;
        result.tasks += ts.tasks;
        result.steal_attempts += ts.steal_attempts;
        result.successful_steals += ts.successful_steals;
    }

    // Load Imbalance lambda = max(T_i) / avg(T_i)
    result.lambda = (sum_time > 0.0) ? (max_time / (sum_time / P)) : 1.0;
    result.probe_s = wall_time;
    return result;
}

struct Task {
    u32 part;
    u32 begin;
    u32 end;
    double cost;
};

// ==============================================================================
// Scheduler 1: Static Coarse Baseline (Balkesen Fig. 16a)
// ==============================================================================

static RunStats run_static(const HTs& H, const Part& S, int P) {
    vector<ThreadStats> ts(P);
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        int tid = omp_get_thread_num();
        u64 m = 0;
        u64 chk = 0;
        u64 tk = 0;

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < NP; i++) {
            probe_task(H, i, S.data() + S.off[i], S.sz(i), m, chk);
            tk++;
        }

        ts[tid].fin = omp_get_wtime() - t0;
        ts[tid].matches = m;
        ts[tid].checksum = chk;
        ts[tid].tasks = tk;
    }

    return finalize(ts, omp_get_wtime() - t0);
}

// ==============================================================================
// Scheduler 2 & 3: Shared Dynamic Queue Claiming (Used by paper & adaptive1)
// ==============================================================================

static RunStats run_queue(const vector<Task>& tasks, const HTs& H, const Part& S, int P) {
    vector<ThreadStats> ts(P);
    atomic<size_t> next_task{0};
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        int tid = omp_get_thread_num();
        u64 m = 0;
        u64 chk = 0;
        u64 tk = 0;

        while (true) {
            size_t idx = next_task.fetch_add(1, memory_order_relaxed);
            if (idx >= tasks.size()) {
                break;
            }
            const Task& task = tasks[idx];
            probe_task(H, task.part, S.data() + task.begin, task.end - task.begin, m, chk);
            tk++;
        }

        ts[tid].fin = omp_get_wtime() - t0;
        ts[tid].matches = m;
        ts[tid].checksum = chk;
        ts[tid].tasks = tk;
    }

    return finalize(ts, omp_get_wtime() - t0);
}

// ==============================================================================
// Task Decomposition: Paper-Style Fixed Chunking (Balkesen Fig. 16b)
// ==============================================================================

static vector<Task> make_paper_tasks(const Part& S, const Cfg& c) {
    vector<Task> tasks;
    double avg_size = static_cast<double>(S.n) / NP;

    for (int i = 0; i < NP; i++) {
        size_t len = S.sz(i);
        if (len == 0) {
            continue;
        }

        // Oversized partitions are broken into fixed chunks
        if (len > c.paper_factor * avg_size) {
            for (size_t b = 0; b < len; b += c.paper_chunk) {
                u32 chunk_end = static_cast<u32>(S.off[i] + min(len, b + c.paper_chunk));
                tasks.push_back({static_cast<u32>(i),
                                 static_cast<u32>(S.off[i] + b),
                                 chunk_end,
                                 0.0});
            }
        } else {
            tasks.push_back({static_cast<u32>(i), S.off[i], S.off[i + 1], 0.0});
        }
    }
    return tasks;
}

static pair<u64, u64> task_chunk_stats(const vector<Task>& tasks) {
    vector<u64> chunk_counts(NP, 0);
    for (const Task& task : tasks) {
        chunk_counts[task.part]++;
    }

    u64 split_partitions = 0;
    u64 max_chunks = 0;
    for (u64 count : chunk_counts) {
        if (count > 1) {
            split_partitions++;
        }
        max_chunks = max(max_chunks, count);
    }
    return {split_partitions, max_chunks};
}

// ==============================================================================
// Task Formation: Adaptive Cost-Aware Splitting with Bounded Task Budget
// ==============================================================================

static vector<Task> make_adaptive_tasks(const Part& R, const Part& S, const Cfg& c,
                                        double alpha, double beta, double gamma,
                                        const vector<double>* estimated_matches, int P) {
    vector<double> C(NP);
    double sumC = 0.0;

    for (int i = 0; i < NP; i++) {
        C[i] = (c.include_build ? (alpha * R.sz(i)) : 0.0)
               + beta * S.sz(i)
               + (estimated_matches ? (gamma * (*estimated_matches)[i]) : 0.0);
        sumC += C[i];
    }

    double fair_share = sumC / (P * c.granularity);
    double task_overhead_s = c.task_overhead_ns * 1e-9;
    vector<int> k(NP, 1);
    long extra_tasks = 0;

    for (int i = 0; i < NP; i++) {
        size_t len = S.sz(i);
        if (fair_share <= 0.0 || len < 2ull * c.min_chunk) {
            continue;
        }

        int kmax = static_cast<int>(len / c.min_chunk);
        int ki = min(kmax, static_cast<int>(ceil(C[i] / fair_share)));

        // Split only if saving exceeds overhead H(k) = k * overhead
        while (ki > 1 && !(C[i] * (1.0 - 1.0 / ki) > ki * task_overhead_s)) {
            ki--;
        }
        k[i] = ki;
        extra_tasks += ki - 1;
    }

    // Enforce bounded task budget: extra tasks <= budget_per_thread * P
    long budget = static_cast<long>(c.budget_per_thread) * P;
    if (extra_tasks > budget) {
        for (int i = 0; i < NP; i++) {
            if (k[i] > 1) {
                k[i] = 1 + static_cast<int>(static_cast<long>(k[i] - 1) * budget / extra_tasks);
            }
        }
    }

    vector<Task> tasks;
    for (int i = 0; i < NP; i++) {
        size_t len = S.sz(i);
        if (len == 0) {
            continue;
        }

        for (int j = 0; j < k[i]; j++) {
            size_t b = len * j / k[i];
            size_t e = len * (j + 1) / k[i];
            if (e > b) {
                tasks.push_back({static_cast<u32>(i),
                                 static_cast<u32>(S.off[i] + b),
                                 static_cast<u32>(S.off[i] + e),
                                 C[i] * static_cast<double>(e - b) / static_cast<double>(len)});
            }
        }
    }

    // Sort tasks largest-first (Longest Processing Time first for load balancing)
    sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
        return a.cost > b.cost;
    });

    return tasks;
}

// ==============================================================================
// Scheduler 4: Adaptive Work Stealing with Chase-Lev Lock-Free Deques (adaptive2)
// ==============================================================================

struct alignas(64) Deque {
    atomic<long> top{0};
    atomic<long> bottom{0};
    atomic<long long> rem{0}; // Estimated queued work remaining in nanoseconds
    vector<Task> buf;
    long mask = 0;

    void init(size_t n) {
        size_t cap = 2;
        while (cap < n + 1) {
            cap <<= 1;
        }
        buf.assign(cap, Task{0, 0, 0, 0.0});
        mask = static_cast<long>(cap) - 1;
    }

    void push(const Task& task) {
        long b = bottom.load(memory_order_relaxed);
        buf[b & mask] = task;
        bottom.store(b + 1, memory_order_relaxed);
    }

    bool pop(Task& task) {
        long b = bottom.load(memory_order_relaxed) - 1;
        bottom.store(b, memory_order_relaxed);
        atomic_thread_fence(memory_order_seq_cst);
        long t = top.load(memory_order_relaxed);

        if (t <= b) {
            task = buf[b & mask];
            if (t == b) {
                bool success = top.compare_exchange_strong(
                    t, t + 1, memory_order_seq_cst, memory_order_relaxed
                );
                bottom.store(b + 1, memory_order_relaxed);
                return success;
            }
            return true;
        }
        bottom.store(b + 1, memory_order_relaxed);
        return false;
    }

    // 0: empty, 1: stolen successfully, 2: lost CAS race
    int steal(Task& task) {
        long t = top.load(memory_order_acquire);
        atomic_thread_fence(memory_order_seq_cst);
        long b = bottom.load(memory_order_acquire);

        if (t < b) {
            task = buf[t & mask];
            if (!top.compare_exchange_strong(t, t + 1, memory_order_seq_cst, memory_order_relaxed)) {
                return 2;
            }
            return 1;
        }
        return 0;
    }
};

static RunStats run_steal(const vector<Task>& tasks, const HTs& H, const Part& S,
                          int P, const Cfg& c) {
    unique_ptr<Deque[]> deques(new Deque[P]);
    for (int w = 0; w < P; w++) {
        deques[w].init(tasks.size());
    }

    // Initial placement: LPT greedy balancing or round-robin
    vector<double> assigned_load(P, 0.0);
    size_t round_robin_idx = 0;

    for (const Task& task : tasks) {
        int worker = 0;
        if (c.init == "rr") {
            worker = static_cast<int>(round_robin_idx % P);
        } else {
            worker = static_cast<int>(
                min_element(assigned_load.begin(), assigned_load.end()) - assigned_load.begin()
            );
        }
        round_robin_idx++;

        deques[worker].push(task);
        assigned_load[worker] += task.cost;
        deques[worker].rem.fetch_add(static_cast<long long>(task.cost * 1e9));
    }

    long long tau_steal = static_cast<long long>(c.tau_ns);
    vector<ThreadStats> ts(P);
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        int tid = omp_get_thread_num();
        u64 m = 0;
        u64 chk = 0;
        u64 tk = 0;
        u64 sa = 0;
        u64 ss = 0;
        Task current_task;

        while (true) {
            bool have_task = false;

            // 1. Try popping from local deque (LIFO - cache hot)
            if (deques[tid].pop(current_task)) {
                deques[tid].rem.fetch_sub(static_cast<long long>(current_task.cost * 1e9));
                have_task = true;
            } else {
                // 2. Try work stealing from a victim (FIFO from victim's top)
                for (int attempt = 0; attempt < 8 && !have_task; attempt++) {
                    int victim = -1;
                    long long max_remaining = 0;

                    for (int w = 0; w < P; w++) {
                        if (w != tid) {
                            long long r = deques[w].rem.load(memory_order_relaxed);
                            if (r > max_remaining) {
                                max_remaining = r;
                                victim = w;
                            }
                        }
                    }

                    // Cost-threshold gate: steal only if victim remaining work > tau_steal
                    if (victim < 0 || max_remaining <= tau_steal) {
                        break;
                    }

                    sa++;
                    if (deques[victim].steal(current_task) == 1) {
                        deques[victim].rem.fetch_sub(static_cast<long long>(current_task.cost * 1e9));
                        ss++;
                        have_task = true;
                    }
                }
            }

            if (!have_task) {
                break;
            }

            probe_task(H, current_task.part, S.data() + current_task.begin,
                       current_task.end - current_task.begin, m, chk);
            tk++;
        }

        ts[tid].fin = omp_get_wtime() - t0;
        ts[tid].matches = m;
        ts[tid].checksum = chk;
        ts[tid].tasks = tk;
        ts[tid].steal_attempts = sa;
        ts[tid].successful_steals = ss;
    }

    return finalize(ts, omp_get_wtime() - t0);
}

// ==============================================================================
// Runtime Calibration of Costs & Overheads
// ==============================================================================

static void calibrate(const Part& R, const Part& S, const HTs& H,
                      double& alpha, double& beta) {
    double avgS = static_cast<double>(S.n) / NP;
    vector<int> sample_partitions;

    for (int j = 0; j < 16; j++) {
        int i = (7 + 61 * j) % NP;
        if (S.sz(i) <= 4 * avgS && R.sz(i) > 0) {
            sample_partitions.push_back(i);
        }
    }

    size_t nr = 0;
    size_t ns = 0;
    double tb = 0.0;
    double tp = 0.0;
    volatile u64 sink = 0;
    vector<Slot> scratch;

    for (int rep = 0; rep < 3; rep++) {
        for (int i : sample_partitions) {
            scratch.resize(cap_for(R.sz(i)));
            double a = omp_get_wtime();
            build_one(R.data() + R.off[i], R.sz(i), scratch.data(), static_cast<u32>(scratch.size()));
            tb += (omp_get_wtime() - a);
            nr += R.sz(i);

            u64 m = 0;
            u64 chk = 0;
            a = omp_get_wtime();
            probe_task(H, i, S.data() + S.off[i], S.sz(i), m, chk);
            tp += (omp_get_wtime() - a);
            ns += S.sz(i);
            sink = sink + m + chk;
        }
    }

    alpha = nr ? (tb / nr) : 1e-8;
    beta = ns ? (tp / ns) : 1e-8;
}

static double calibrate_gamma() {
    vector<u32> test_vec(4096, 3);
    volatile u64 sink = 0;
    const int reps = 20000;

    double a = omp_get_wtime();
    u64 sum = 0;
    for (int r = 0; r < reps; r++) {
        u64 q = 0;
        for (size_t x = 0; x < test_vec.size(); x++) {
            q += test_vec[x];
        }
        sum += q;
    }
    double elapsed = omp_get_wtime() - a;
    sink = sink + sum;
    return elapsed / (static_cast<double>(reps) * test_vec.size());
}

static void estimate_outputs(const HTs& H, const Part& S, int samples_per_part,
                             vector<double>& est) {
    est.assign(NP, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NP; i++) {
        size_t len = S.sz(i);
        if (len == 0) {
            continue;
        }
        size_t cnt = min<size_t>(samples_per_part, len);
        size_t stride = len / cnt;
        u64 mm = 0;
        u64 chk = 0;

        for (size_t j = 0; j < cnt; j++) {
            probe_expand(H, i, S.data() + S.off[i] + j * stride, 1, mm, chk);
        }
        est[i] = (static_cast<double>(mm) / cnt) * len;
    }
}

// Measures:
//  cas_ns : Contended atomic compare-and-swap latency across P threads
//  mig_ns : Cache-miss reload latency to pull a hash table into local cache
static void calibrate_overheads(const HTs& H, int P, double& cas_ns, double& mig_ns) {
    atomic<long> counter{0};
    const int K = 100000;
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(P)
    {
        for (int i = 0; i < K; i++) {
            long v = counter.load(memory_order_relaxed);
            while (!counter.compare_exchange_weak(v, v + 1, memory_order_seq_cst, memory_order_relaxed)) {
                // backoff retry
            }
        }
    }
    cas_ns = (omp_get_wtime() - t0) / K * 1e9;

    vector<char> evict_buffer(64u << 20); // 64 MB eviction buffer
    volatile u64 sink = 0;
    double total_migration_time = 0.0;
    int sample_count = 0;

    for (int j = 0; j < 8; j++) {
        int i = (100 + 97 * j) % NP;
        for (size_t k = 0; k < evict_buffer.size(); k += 64) {
            evict_buffer[k] = static_cast<char>(k + j);
        }
        sink = sink + evict_buffer[(j * 64) % evict_buffer.size()];

        const char* base_ptr = H.expand ? reinterpret_cast<const char*>(H.arenaE.get() + H.off[i])
                                        : reinterpret_cast<const char*>(H.arena.get() + H.off[i]);
        size_t bytes = H.cap[i] * (H.expand ? sizeof(SlotE) : sizeof(Slot));

        double a = omp_get_wtime();
        u64 sum = 0;
        for (size_t k = 0; k < bytes; k += 64) {
            sum += base_ptr[k];
        }
        total_migration_time += (omp_get_wtime() - a);
        sink = sink + sum;
        sample_count++;
    }

    mig_ns = (total_migration_time / sample_count) * 1e9;
}

// ==============================================================================
// Main Experiment Driver & CLI Options
// ==============================================================================

static double compute_median(vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    sort(v.begin(), v.end());
    size_t sz = v.size();
    return (sz % 2) ? v[sz / 2] : 0.5 * (v[sz / 2 - 1] + v[sz / 2]);
}

int main(int argc, char** argv) {
    Cfg cfg;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        auto get_val = [&]() {
            return string(i + 1 < argc ? argv[++i] : "");
        };

        if (arg == "--n") cfg.n = stoull(get_val());
        else if (arg == "--key_range") cfg.key_range = stoul(get_val());
        else if (arg == "--zs") {
            cfg.zs.clear();
            for (const auto& s : split(get_val())) cfg.zs.push_back(stod(s));
        }
        else if (arg == "--threads") {
            cfg.threads.clear();
            for (const auto& s : split(get_val())) cfg.threads.push_back(stoi(s));
        }
        else if (arg == "--reps") cfg.reps = stoi(get_val());
        else if (arg == "--warmup") cfg.warmup = stoi(get_val());
        else if (arg == "--methods") cfg.methods = split(get_val());
        else if (arg == "--out") cfg.out = get_val();
        else if (arg == "--paper_factor") cfg.paper_factor = stod(get_val());
        else if (arg == "--paper_chunk") cfg.paper_chunk = stoul(get_val());
        else if (arg == "--granularity") cfg.granularity = stod(get_val());
        else if (arg == "--min_chunk") cfg.min_chunk = stoul(get_val());
        else if (arg == "--budget_per_thread") cfg.budget_per_thread = stoi(get_val());
        else if (arg == "--task_overhead_ns") {
            string v = get_val();
            if (v == "auto") {
                cfg.ovh_auto = true;
            } else {
                cfg.ovh_auto = false;
                cfg.task_overhead_ns = stod(v);
            }
        }
        else if (arg == "--ns") cfg.ns = stoull(get_val());
        else if (arg == "--expand") cfg.expand = (stoi(get_val()) != 0);
        else if (arg == "--r_z") cfg.r_z = stod(get_val());
        else if (arg == "--samples") cfg.samples = stoi(get_val());
        else if (arg == "--plan_gate") cfg.plan_gate = stod(get_val());
        else if (arg == "--gamma_ns") {
            string v = get_val();
            if (v == "auto") {
                cfg.gamma_auto = true;
            } else {
                cfg.gamma_auto = false;
                cfg.gamma_ns = stod(v);
            }
        }
        else if (arg == "--init") cfg.init = get_val();
        else if (arg == "--include_build") cfg.include_build = stoi(get_val());
        else if (arg == "--tau_ns") {
            string v = get_val();
            if (v == "auto") {
                cfg.tau_auto = true;
            } else {
                cfg.tau_auto = false;
                cfg.tau_ns = stod(v);
            }
        }
        else {
            fprintf(stderr, "Unknown command line option: %s\n", arg.c_str());
            return 2;
        }
    }

    omp_set_dynamic(0);

    FILE* out_file = fopen(cfg.out.c_str(), "w");
    if (!out_file) {
        perror("Failed to open output CSV file");
        return 2;
    }

    fprintf(out_file,
            "z,threads,method,rep,warmup,n,ns,partition_s,build_s,probe_s,total_s,seq_s,matches,correct,"
            "lambda,max_avg_part,tasks,split_partitions,max_chunks_per_partition,steal_attempts,steals,"
            "alpha,beta,tau_ns,overhead_ns,plan_s,expand,r_z,init,paper_factor,paper_chunk,"
            "granularity,min_chunk,budget_per_thread,include_build,key_range,seed_r,seed_s\n");

    int failure_count = 0;
    printf("|R|=%zu |S|=%zu key_range=%u reps=%d (+%d warmup) init=%s workload=%s r_z=%.2f\n",
           cfg.n, cfg.ns ? cfg.ns : cfg.n, cfg.key_range, cfg.reps, cfg.warmup,
           cfg.init.c_str(), cfg.expand ? "expand" : "count", cfg.r_z);

    for (double z : cfg.zs) {
        size_t nS = cfg.ns ? cfg.ns : cfg.n;
        vector<Tuple> R_raw = gen_keys(cfg.n, cfg.key_range, cfg.r_z, cfg.seedR);
        for (size_t i = 0; i < R_raw.size(); i++) {
            R_raw[i].payload = static_cast<u32>(i);
        }
        vector<Tuple> S_raw = gen_keys(nS, cfg.key_range, z, cfg.seedS);

        u64 ref_matches = 0;
        u64 ref_checksum = 0;
        vector<double> seq_timings;

        for (int r = 0; r < cfg.warmup + cfg.reps; r++) {
            double t_start = omp_get_wtime();
            seq_join(R_raw, S_raw, cfg.expand, ref_matches, ref_checksum);
            double duration = omp_get_wtime() - t_start;
            if (r >= cfg.warmup) {
                seq_timings.push_back(duration);
            }
        }
        double seq_s = compute_median(seq_timings);
        printf("\n== z=%.2f  sequential reference: %.3fs  matches=%llu\n",
               z, seq_s, static_cast<unsigned long long>(ref_matches));

        for (int threads : cfg.threads) {
            omp_set_num_threads(threads);
            Part R_part;
            Part S_part;
            HTs H_tables;
            double cas_ns = 0.0;
            double mig_ns = 0.0;
            double gamma_val = 0.0;
            bool overheads_calibrated = false;

            for (int rep = 0; rep < cfg.warmup + cfg.reps; rep++) {
                double t_start = omp_get_wtime();
                partition2(R_raw, R_part);
                partition2(S_raw, S_part);
                double partition_s = omp_get_wtime() - t_start;

                t_start = omp_get_wtime();
                build_all(R_part, H_tables, cfg.expand);
                double build_s = omp_get_wtime() - t_start;

                double alpha = 0.0;
                double beta = 0.0;
                calibrate(R_part, S_part, H_tables, alpha, beta);

                size_t max_part_size = 0;
                for (int i = 0; i < NP; i++) {
                    max_part_size = max(max_part_size, S_part.sz(i));
                }
                double max_avg_part = static_cast<double>(max_part_size) /
                                      (static_cast<double>(S_part.n) / NP);

                Cfg run_cfg = cfg;
                if (!overheads_calibrated) {
                    gamma_val = cfg.gamma_auto ? calibrate_gamma() : (cfg.gamma_ns * 1e-9);
                    if (cfg.tau_auto || cfg.ovh_auto) {
                        calibrate_overheads(H_tables, threads, cas_ns, mig_ns);
                    }
                    overheads_calibrated = true;
                }

                if (cfg.tau_auto) {
                    run_cfg.tau_ns = cas_ns + mig_ns;
                }
                if (cfg.ovh_auto) {
                    run_cfg.task_overhead_ns = cas_ns + mig_ns;
                }

                t_start = omp_get_wtime();
                vector<Task> paper_tasks = make_paper_tasks(S_part, cfg);
                double plan_paper = omp_get_wtime() - t_start;

                t_start = omp_get_wtime();
                vector<double> estimated_matches;
                bool do_sample = cfg.expand && (max_avg_part >= cfg.plan_gate);
                if (do_sample) {
                    estimate_outputs(H_tables, S_part, cfg.samples, estimated_matches);
                }
                vector<Task> adaptive_tasks = make_adaptive_tasks(
                    R_part, S_part, run_cfg, alpha, beta, gamma_val,
                    do_sample ? &estimated_matches : nullptr, threads
                );
                double plan_adaptive = omp_get_wtime() - t_start;

                auto paper_stats = task_chunk_stats(paper_tasks);
                auto adaptive_stats = task_chunk_stats(adaptive_tasks);

                for (size_t mi = 0; mi < cfg.methods.size(); mi++) {
                    // Rotate execution order to avoid bias across repetitions
                    const string& method = cfg.methods[(mi + rep) % cfg.methods.size()];
                    RunStats st;

                    if (method == "static") {
                        st = run_static(H_tables, S_part, threads);
                    } else if (method == "paper") {
                        st = run_queue(paper_tasks, H_tables, S_part, threads);
                    } else if (method == "adaptive1") {
                        st = run_queue(adaptive_tasks, H_tables, S_part, threads);
                    } else if (method == "adaptive2") {
                        st = run_steal(adaptive_tasks, H_tables, S_part, threads, run_cfg);
                    } else {
                        fprintf(stderr, "Unknown scheduler method: %s\n", method.c_str());
                        return 2;
                    }

                    bool correct = (st.matches == ref_matches) &&
                                   (!cfg.expand || (st.checksum == ref_checksum));
                    if (!correct) {
                        failure_count++;
                    }

                    double plan_time = (method == "static") ? 0.0 :
                                       (method == "paper" ? plan_paper : plan_adaptive);
                    double total_time = partition_s + build_s + plan_time + st.probe_s;

                    u64 split_partitions = 0;
                    u64 max_chunks = 1;
                    if (method == "paper") {
                        split_partitions = paper_stats.first;
                        max_chunks = paper_stats.second;
                    } else if (method == "adaptive1" || method == "adaptive2") {
                        split_partitions = adaptive_stats.first;
                        max_chunks = adaptive_stats.second;
                    }

                    fprintf(out_file,
                            "%.2f,%d,%s,%d,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%d,%.4f,%.2f,%llu,%llu,%llu,%llu,%llu,"
                            "%.3e,%.3e,%.0f,%.0f,%.6f,%d,%.2f,%s,%.2f,%u,%.2f,%u,%d,%d,%u,%llu,%llu\n",
                            z, threads, method.c_str(), rep, (rep < cfg.warmup),
                            cfg.n, S_part.n, partition_s, build_s, st.probe_s, total_time, seq_s,
                            static_cast<unsigned long long>(st.matches), correct ? 1 : 0,
                            st.lambda, max_avg_part, static_cast<unsigned long long>(st.tasks),
                            static_cast<unsigned long long>(split_partitions),
                            static_cast<unsigned long long>(max_chunks),
                            static_cast<unsigned long long>(st.steal_attempts),
                            static_cast<unsigned long long>(st.successful_steals),
                            alpha, beta, run_cfg.tau_ns, run_cfg.task_overhead_ns, plan_time,
                            static_cast<int>(cfg.expand), cfg.r_z, cfg.init.c_str(),
                            cfg.paper_factor, cfg.paper_chunk, cfg.granularity,
                            cfg.min_chunk, cfg.budget_per_thread, cfg.include_build, cfg.key_range,
                            static_cast<unsigned long long>(cfg.seedR),
                            static_cast<unsigned long long>(cfg.seedS));

                    printf("  p=%d rep=%d %-9s total=%.1fms probe=%.2fms lambda=%.2f tasks=%llu steals=%llu %s\n",
                           threads, rep, method.c_str(), total_time * 1e3, st.probe_s * 1e3,
                           st.lambda, static_cast<unsigned long long>(st.tasks),
                           static_cast<unsigned long long>(st.successful_steals),
                           correct ? "PASS" : "FAIL");
                }
                fflush(out_file);
            }
        }
    }

    fclose(out_file);
    printf("\n%s (%d incorrect runs). Results saved in %s\n",
           failure_count ? "FAILED" : "ALL CORRECT", failure_count, cfg.out.c_str());
    return failure_count ? 1 : 0;
}
