// Skew-resilient parallel hash join: experiment harness + 4 probe-phase schedulers.
//
// Pipeline (follows the methodology slides):
//   1 data generation (seeded, uniform R, Zipf-skewed S)
//   2 sequential reference join (std::unordered_map)  -> ground truth + time reference
//   3 two-pass radix partitioning (5+5 bits -> 1024 partitions)
//   4 partition statistics (Max/Avg)
//   5 build phase (one hash table per R-partition, parallel static)
//   6 probe phase, scheduled by one of:
//        static     : static coarse scheduling of whole partitions (baseline)
//        paper     : paper-inspired fine-grained scheduling of oversized S_i chunks,
//                     threads claim tasks from one shared list
//        adaptive1  : cost-based splitting + bounded task budget + dynamic claiming
//        adaptive2  : adaptive1 tasks + per-worker Chase-Lev deques + cost-gated stealing
//   7 correctness check against the sequential result + CSV output
//
// Build: g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin

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

constexpr int NP = 1024;  // 2^10 partitions (two passes of 5 bits)

struct Tuple { u32 key, payload; };

static inline u32 h1(u32 k) { return k * 0x9E3779B1u; }  // radix hash (top 10 bits = partition id)
static inline u32 fmix(u32 h) {                          // hash-table hash (murmur finalizer)
  h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16; return h;
}

// ------------------------------------------------------------------ config
struct Cfg {
  size_t n = 5000000;          // |R|
  size_t ns = 0;               // |S| (0 = same as |R|)
  u32 key_range = 1000000;
  vector<double> zs = {0, 0.5, 0.75, 1.0, 1.25, 1.5};
  vector<int> threads = {1, 2, 4};
  int reps = 5, warmup = 1;
  vector<string> methods = {"static", "paper", "adaptive1", "adaptive2"};
  string out = "results.csv";
  u64 seedR = 123456789ULL, seedS = 987654321ULL;
  // paper-style baseline
  double paper_factor = 2.0;   // split S_i if |S_i| > factor * average
  u32 paper_chunk = 16384;     // fixed chunk size (tuples)
  // adaptive (cost-based) scheduler
  double granularity = 4.0;    // each thread's fair share is cut into this many pieces
  u32 min_chunk = 2048;        // never create chunks smaller than this (tuples)
  int budget_per_thread = 32;  // task budget: max extra tasks = budget_per_thread * threads
  double task_overhead_ns = 5000;  // per-task overhead used in H(k) = k * overhead
  bool ovh_auto = true;        // measure it (CAS + cache migration) instead of using the number above
  bool tau_auto = true;        // measure tau_steal = Cost(CAS) + Cost(cache migration)
  double plan_gate = 1.5;      // skip the sampling pass when max/avg |S_i| is below this (data nearly uniform)
  bool expand = false;         // many-to-many workload: probe walks every matching R tuple
  double r_z = 0.0;            // Zipf skew of the R (build) side keys
  int samples = 128;           // sampled probes per partition for output-size estimation (expand mode)
  bool gamma_auto = true; double gamma_ns = 0.5;  // per-match cost (expand mode)
  string init = "lpt";         // adaptive2 initial task assignment: lpt (cost-balanced) | rr (round robin)
  int include_build = 0;       // 1: C_i = alpha|R_i| + beta|S_i|  (0: probe-only, see README)
  double tau_ns = 20000;       // steal threshold tau_steal in nanoseconds (used when tau_auto is false)
};

static vector<string> split(const string& s, char d = ',') {
  vector<string> r; string t; stringstream ss(s);
  while (getline(ss, t, d)) if (!t.empty()) r.push_back(t);
  return r;
}

// ------------------------------------------------------------------ data

static vector<Tuple> gen_keys(size_t n, u32 K, double z, u64 seed) {
  mt19937_64 g(seed); vector<Tuple> S(n);
  if (z <= 0) {
    uniform_int_distribution<u32> d(1, K);
    for (size_t i = 0; i < n; i++) S[i] = {d(g), (u32)i};
  } else {  // Zipf over ranks 1..K, P(r) ~ 1/r^z, key = rank
    vector<double> cdf(K); double acc = 0;
    for (u32 r = 1; r <= K; r++) { acc += pow((double)r, -z); cdf[r - 1] = acc; }
    uniform_real_distribution<double> d(0.0, acc);
    for (size_t i = 0; i < n; i++) {
      u32 r = (u32)(lower_bound(cdf.begin(), cdf.end(), d(g)) - cdf.begin()) + 1;
      if (r > K) r = K;
      S[i] = {r, (u32)i};
    }
  }
  return S;
}

// ------------------------------------------------------------------ sequential reference
static void seq_join(const vector<Tuple>& R, const vector<Tuple>& S, bool expand, u64& matches, u64& chk) {
  matches = 0; chk = 0;
  if (!expand) {
    unordered_map<u32, u32> m; m.reserve(R.size() * 2);
    for (auto& r : R) m[r.key]++;
    for (auto& s : S) { auto it = m.find(s.key); if (it != m.end()) matches += it->second; }
  } else {  // checksum = sum over all matching (S,R) pairs of the R payload
    unordered_map<u32, pair<u32, u64>> m; m.reserve(R.size() * 2);
    for (auto& r : R) { auto& e = m[r.key]; e.first++; e.second += r.payload; }
    for (auto& s : S) { auto it = m.find(s.key); if (it != m.end()) { matches += it->second.first; chk += it->second.second; } }
  }
}

// ------------------------------------------------------------------ two-pass radix partitioning
struct Part {
  unique_ptr<Tuple[]> buf; size_t n = 0; vector<u32> off;  // off has NP+1 entries
  Tuple* data() const { return buf.get(); }
  size_t sz(int i) const { return off[i + 1] - off[i]; }
};

static void partition2(const vector<Tuple>& in, Part& out) {
  size_t n = in.size(); int P = omp_get_max_threads();
  if (!out.buf || out.n != n) { out.buf.reset(new Tuple[n]); out.n = n; }  // uninitialised: pages touched in parallel
  out.off.assign(NP + 1, 0);
  static unique_ptr<Tuple[]> tmpbuf; static size_t tmpcap = 0;           // reused scratch buffer
  if (tmpcap < n) { tmpbuf.reset(new Tuple[n]); tmpcap = n; }
  Tuple* tmp = tmpbuf.get(); Tuple* o = out.data();
  vector<array<u32, 32>> hist(P), pos(P);
  u32 boff[33];
  // pass 1: top 5 bits, per-thread histogram + prefix sum + scatter
  #pragma omp parallel num_threads(P)
  {
    int t = omp_get_thread_num();
    size_t lo = n * t / P, hi = n * (t + 1) / P;
    hist[t].fill(0);
    for (size_t i = lo; i < hi; i++) hist[t][h1(in[i].key) >> 27]++;
    #pragma omp barrier
    #pragma omp single
    {
      u32 run = 0;
      for (int b = 0; b < 32; b++) {
        boff[b] = run;
        for (int q = 0; q < P; q++) { pos[q][b] = run; run += hist[q][b]; }
      }
      boff[32] = run;
    }
    for (size_t i = lo; i < hi; i++) tmp[pos[t][h1(in[i].key) >> 27]++] = in[i];
  }
  // pass 2: next 5 bits inside each of the 32 buckets. Big buckets (the ones holding hot keys) are
  // processed by ALL threads one after another; the remaining small buckets are processed in parallel.
  size_t bigthr = P > 1 ? n / (2 * (size_t)P) : (size_t)-1;
  vector<array<u32, 32>> bh(P), bp(P); vector<int> smalls;
  for (int b = 0; b < 32; b++) {
    u32 lo0 = boff[b], len = boff[b + 1] - boff[b];
    if (len <= bigthr) { smalls.push_back(b); continue; }
    #pragma omp parallel num_threads(P)
    {
      int t = omp_get_thread_num();
      size_t lo = lo0 + (size_t)len * t / P, hi = lo0 + (size_t)len * (t + 1) / P;
      bh[t].fill(0);
      for (size_t i = lo; i < hi; i++) bh[t][(h1(tmp[i].key) >> 22) & 31]++;
      #pragma omp barrier
      #pragma omp single
      {
        u32 run = lo0;
        for (int s2 = 0; s2 < 32; s2++) {
          out.off[b * 32 + s2] = run;
          for (int q = 0; q < P; q++) { bp[q][s2] = run; run += bh[q][s2]; }
        }
      }
      for (size_t i = lo; i < hi; i++) o[bp[t][(h1(tmp[i].key) >> 22) & 31]++] = tmp[i];
    }
  }
  #pragma omp parallel for schedule(dynamic, 1) num_threads(P)
  for (size_t k = 0; k < smalls.size(); k++) {
    int b = smalls[k]; u32 lo = boff[b], hi = boff[b + 1]; u32 c[32] = {0};
    for (u32 i = lo; i < hi; i++) c[(h1(tmp[i].key) >> 22) & 31]++;
    u32 run = lo, q[32];
    for (int s2 = 0; s2 < 32; s2++) { out.off[b * 32 + s2] = run; q[s2] = run; run += c[s2]; }
    for (u32 i = lo; i < hi; i++) o[q[(h1(tmp[i].key) >> 22) & 31]++] = tmp[i];
  }
  out.off[NP] = (u32)n;
}

// ------------------------------------------------------------------ hash tables (key -> count of R tuples)
struct Slot { u32 key, cnt; };                   // key 0 = empty (keys start at 1)
struct SlotE { u32 key, cnt, start, fill; };     // expand mode: payloads of the R tuples with this key live at pl[start..start+cnt)
struct HTs {
  unique_ptr<Slot[]> arena; unique_ptr<SlotE[]> arenaE; unique_ptr<u32[]> pl;
  size_t cap_slots = 0, cap_pl = 0; vector<size_t> off; vector<u32> cap, plo; bool expand = false;
};

static u32 cap_for(size_t len) { u32 c = 16; while (c < 2 * len) c <<= 1; return c; }

static void build_one(const Tuple* r, size_t len, Slot* T, u32 cap) {
  memset(T, 0, sizeof(Slot) * cap); u32 m = cap - 1;
  for (size_t i = 0; i < len; i++) {
    u32 k = r[i].key, j = fmix(k) & m;
    while (T[j].key != 0 && T[j].key != k) j = (j + 1) & m;
    T[j].key = k; T[j].cnt++;
  }
}

static void build_one_e(const Tuple* r, size_t len, SlotE* T, u32 cap, u32* pl) {
  memset(T, 0, sizeof(SlotE) * cap); u32 m = cap - 1;
  for (size_t i = 0; i < len; i++) {
    u32 k = r[i].key, j = fmix(k) & m;
    while (T[j].key != 0 && T[j].key != k) j = (j + 1) & m;
    T[j].key = k; T[j].cnt++;
  }
  u32 run = 0;
  for (u32 j = 0; j < cap; j++) if (T[j].key) { T[j].start = run; run += T[j].cnt; }
  for (size_t i = 0; i < len; i++) {
    u32 k = r[i].key, j = fmix(k) & m;
    while (T[j].key != k) j = (j + 1) & m;
    pl[T[j].start + T[j].fill++] = r[i].payload;
  }
}

static void build_all(const Part& R, HTs& H, bool expand) {
  H.expand = expand; H.off.assign(NP + 1, 0); H.cap.assign(NP, 0); H.plo.assign(R.off.begin(), R.off.end()); size_t tot = 0;
  for (int i = 0; i < NP; i++) { H.cap[i] = cap_for(R.sz(i)); H.off[i] = tot; tot += H.cap[i]; }
  H.off[NP] = tot;
  if (!expand) { if (tot > H.cap_slots) { H.arena.reset(new Slot[tot]); H.cap_slots = tot; } }
  else {
    if (tot > H.cap_slots) { H.arenaE.reset(new SlotE[tot]); H.cap_slots = tot; }
    if (R.n > H.cap_pl) { H.pl.reset(new u32[R.n]); H.cap_pl = R.n; }
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < NP; i++) {
    if (!expand) build_one(R.data() + R.off[i], R.sz(i), H.arena.get() + H.off[i], H.cap[i]);
    else build_one_e(R.data() + R.off[i], R.sz(i), H.arenaE.get() + H.off[i], H.cap[i], H.pl.get() + H.plo[i]);
  }
}

static inline u64 probe_range(const HTs& H, int p, const Tuple* s, size_t len) {
  const Slot* T = H.arena.get() + H.off[p]; u32 m = H.cap[p] - 1; u64 sum = 0;
  for (size_t i = 0; i < len; i++) {
    u32 k = s[i].key, j = fmix(k) & m;
    for (;;) {
      const Slot& e = T[j];
      if (e.key == k) { sum += e.cnt; break; }
      if (e.key == 0) break;
      j = (j + 1) & m;
    }
  }
  return sum;
}

// expand mode: for every match, walk all matching R payloads (work grows with the number of matches)
static inline void probe_expand(const HTs& H, int p, const Tuple* s, size_t len, u64& mm, u64& chk) {
  const SlotE* T = H.arenaE.get() + H.off[p]; const u32* pl = H.pl.get() + H.plo[p]; u32 m = H.cap[p] - 1;
  for (size_t i = 0; i < len; i++) {
    u32 k = s[i].key, j = fmix(k) & m;
    for (;;) {
      const SlotE& e = T[j];
      if (e.key == k) { mm += e.cnt; const u32* q = pl + e.start; u64 a = 0; for (u32 x = 0; x < e.cnt; x++) a += q[x]; chk += a; break; }
      if (e.key == 0) break;
      j = (j + 1) & m;
    }
  }
}

static inline void probe_task(const HTs& H, int p, const Tuple* s, size_t len, u64& m, u64& chk) {
  if (H.expand) probe_expand(H, p, s, len, m, chk); else m += probe_range(H, p, s, len);
}

// ------------------------------------------------------------------ stats
struct alignas(64) TS { double fin = 0; u64 m = 0, chk = 0, tasks = 0, sa = 0, ss = 0; };
struct RunStats { u64 matches = 0, chk = 0, tasks = 0, steal_att = 0, steals = 0; double lambda = 1, probe_s = 0; };

static RunStats finalize(vector<TS>& ts, double wall) {
  RunStats r; double mx = 0, sum = 0; int P = (int)ts.size();
  for (auto& t : ts) {
    mx = max(mx, t.fin); sum += t.fin;
    r.matches += t.m; r.chk += t.chk; r.tasks += t.tasks; r.steal_att += t.sa; r.steals += t.ss;
  }
  r.lambda = sum > 0 ? mx / (sum / P) : 1.0;  // lambda = max(Ti) / avg(Ti), Ti = thread finish time
  r.probe_s = wall;
  return r;
}

struct Task { u32 part, begin, end; double cost; };  // [begin,end) are absolute indices into S.t

// ------------------------------------------------------------------ (a) static coarse baseline
static RunStats run_static(const HTs& H, const Part& S, int P) {
  vector<TS> ts(P); double t0 = omp_get_wtime();
  #pragma omp parallel num_threads(P)
  {
    int t = omp_get_thread_num(); u64 m = 0, chk = 0, tk = 0;
    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NP; i++) { probe_task(H, i, S.data() + S.off[i], S.sz(i), m, chk); tk++; }
    ts[t].fin = omp_get_wtime() - t0; ts[t].m = m; ts[t].chk = chk; ts[t].tasks = tk;
  }
  return finalize(ts, omp_get_wtime() - t0);
}

// ------------------------------------------------------------------ shared dynamic task list
static RunStats run_queue(const vector<Task>& T, const HTs& H, const Part& S, int P) {
  vector<TS> ts(P); atomic<size_t> next{0}; double t0 = omp_get_wtime();
  #pragma omp parallel num_threads(P)
  {
    int t = omp_get_thread_num(); u64 m = 0, chk = 0, tk = 0;
    for (;;) {
      size_t i = next.fetch_add(1, memory_order_relaxed);
      if (i >= T.size()) break;
      const Task& x = T[i];
      probe_task(H, x.part, S.data() + x.begin, x.end - x.begin, m, chk); tk++;
    }
    ts[t].fin = omp_get_wtime() - t0; ts[t].m = m; ts[t].chk = chk; ts[t].tasks = tk;
  }
  return finalize(ts, omp_get_wtime() - t0);
}

// ------------------------------------------------------------------ (b) paper-inspired fine-grained dynamic
static vector<Task> make_paper_tasks(const Part& S, const Cfg& c) {
  vector<Task> T; double avg = (double)S.n / NP;
  for (int i = 0; i < NP; i++) {
    size_t len = S.sz(i); if (!len) continue;
    if (len > c.paper_factor * avg) {  // oversized -> fixed-size chunks
      for (size_t b = 0; b < len; b += c.paper_chunk)
        T.push_back({(u32)i, (u32)(S.off[i] + b), (u32)(S.off[i] + min(len, b + c.paper_chunk)), 0});
    } else T.push_back({(u32)i, S.off[i], S.off[i + 1], 0});
  }
  return T;  // partition order, no cost information
}

// ------------------------------------------------------------------ (c) adaptive cost-aware task formation
// C_i = [alpha|R_i|] + beta|S_i|; split S_i into k chunks if C_i (1-1/k) > H(k), H(k) = k * task_overhead.
static vector<Task> make_adaptive_tasks(const Part& R, const Part& S, const Cfg& c, double alpha, double beta,
                                        double gamma, const vector<double>* est, int P, u64& nsplit) {
  vector<double> C(NP); double sumC = 0;
  for (int i = 0; i < NP; i++) {
    C[i] = (c.include_build ? alpha * R.sz(i) : 0.0) + beta * S.sz(i) + (est ? gamma * (*est)[i] : 0.0); sumC += C[i];
  }
  double share = sumC / (P * c.granularity), ct = c.task_overhead_ns * 1e-9;
  vector<int> k(NP, 1); long extra = 0;
  for (int i = 0; i < NP; i++) {
    size_t len = S.sz(i);
    if (share <= 0 || len < 2ull * c.min_chunk) continue;
    int kmax = (int)(len / c.min_chunk);
    int ki = min(kmax, (int)ceil(C[i] / share));
    while (ki > 1 && !(C[i] * (1.0 - 1.0 / ki) > ki * ct)) ki--;  // split only if saving > H(k)
    k[i] = ki; extra += ki - 1;
  }
  long B = (long)c.budget_per_thread * P;  // bounded task budget
  if (extra > B) for (int i = 0; i < NP; i++) if (k[i] > 1) k[i] = 1 + (int)((long)(k[i] - 1) * B / extra);
  vector<Task> T; nsplit = 0;
  for (int i = 0; i < NP; i++) {
    size_t len = S.sz(i); if (!len) continue;
    if (k[i] > 1) nsplit++;
    for (int j = 0; j < k[i]; j++) {
      size_t b = len * j / k[i], e = len * (j + 1) / k[i];
      if (e > b) T.push_back({(u32)i, (u32)(S.off[i] + b), (u32)(S.off[i] + e), C[i] * (double)(e - b) / (double)len});
    }
  }
  sort(T.begin(), T.end(), [](const Task& a, const Task& b) { return a.cost > b.cost; });  // largest first
  return T;
}

// ------------------------------------------------------------------ adaptive2: Chase-Lev deques + gated stealing
struct alignas(64) Deque {
  atomic<long> top{0}, bottom{0}; atomic<long long> rem{0};  // rem = queued work (ns)
  vector<Task> buf; long mask = 0;
  void init(size_t n) { size_t c = 2; while (c < n + 1) c <<= 1; buf.assign(c, Task{0, 0, 0, 0}); mask = (long)c - 1; }
  void push(const Task& x) {  // only used single-threaded, before the parallel region
    long b = bottom.load(memory_order_relaxed); buf[b & mask] = x; bottom.store(b + 1, memory_order_relaxed);
  }
  bool pop(Task& x) {  // owner: LIFO from bottom
    long b = bottom.load(memory_order_relaxed) - 1; bottom.store(b, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    long t = top.load(memory_order_relaxed);
    if (t <= b) {
      x = buf[b & mask];
      if (t == b) {
        bool ok = top.compare_exchange_strong(t, t + 1, memory_order_seq_cst, memory_order_relaxed);
        bottom.store(b + 1, memory_order_relaxed); return ok;
      }
      return true;
    }
    bottom.store(b + 1, memory_order_relaxed); return false;
  }
  int steal(Task& x) {  // thief: FIFO from top. 0 empty, 1 ok, 2 lost the race
    long t = top.load(memory_order_acquire); atomic_thread_fence(memory_order_seq_cst);
    long b = bottom.load(memory_order_acquire);
    if (t < b) {
      x = buf[t & mask];
      if (!top.compare_exchange_strong(t, t + 1, memory_order_seq_cst, memory_order_relaxed)) return 2;
      return 1;
    }
    return 0;
  }
};

static RunStats run_steal(const vector<Task>& T, const HTs& H, const Part& S, int P, const Cfg& c) {
  unique_ptr<Deque[]> dq(new Deque[P]);
  for (int w = 0; w < P; w++) dq[w].init(T.size());
  // initial assignment: greedy (largest task -> least loaded worker); pushed largest-first, so the
  // owner pops its small tasks first and thieves (FIFO from top) see the large ones.
  vector<double> load(P, 0.0); size_t idx = 0;
  for (const Task& x : T) {
    int w = (c.init == "rr") ? (int)(idx % P)  // round robin: ignores costs, stealing must fix imbalance
                              : (int)(min_element(load.begin(), load.end()) - load.begin());  // cost-balanced (LPT)
    idx++;
    dq[w].push(x); load[w] += x.cost; dq[w].rem.fetch_add((long long)(x.cost * 1e9));
  }
  long long tau = (long long)c.tau_ns;
  vector<TS> ts(P); double t0 = omp_get_wtime();
  #pragma omp parallel num_threads(P)
  {
    int t = omp_get_thread_num(); u64 m = 0, chk = 0, tk = 0, sa = 0, ss = 0; Task x;
    for (;;) {
      bool have = false;
      if (dq[t].pop(x)) { dq[t].rem.fetch_sub((long long)(x.cost * 1e9)); have = true; }
      else {
        for (int attempt = 0; attempt < 8 && !have; attempt++) {
          int v = -1; long long best = 0;
          for (int w = 0; w < P; w++) if (w != t) {
            long long r = dq[w].rem.load(memory_order_relaxed); if (r > best) { best = r; v = w; }
          }
          if (v < 0 || best <= tau) break;  // gate: victim's remaining work must exceed tau_steal
          sa++;
          if (dq[v].steal(x) == 1) { dq[v].rem.fetch_sub((long long)(x.cost * 1e9)); ss++; have = true; }
        }
      }
      if (!have) break;
      probe_task(H, x.part, S.data() + x.begin, x.end - x.begin, m, chk); tk++;
    }
    ts[t].fin = omp_get_wtime() - t0; ts[t].m = m; ts[t].chk = chk; ts[t].tasks = tk; ts[t].sa = sa; ts[t].ss = ss;
  }
  return finalize(ts, omp_get_wtime() - t0);
}

// ------------------------------------------------------------------ calibration of alpha, beta (seconds/tuple)
static void calibrate(const Part& R, const Part& S, const HTs& H, double& alpha, double& beta) {
  double avgS = (double)S.n / NP; vector<int> sample;
  for (int j = 0; j < 16; j++) { int i = (7 + 61 * j) % NP; if (S.sz(i) <= 4 * avgS && R.sz(i) > 0) sample.push_back(i); }
  size_t nr = 0, ns = 0; double tb = 0, tp = 0; volatile u64 sink = 0;
  vector<Slot> scratch;
  for (int rep = 0; rep < 3; rep++) {
    for (int i : sample) {
      scratch.resize(cap_for(R.sz(i)));
      double a = omp_get_wtime(); build_one(R.data() + R.off[i], R.sz(i), scratch.data(), (u32)scratch.size());
      tb += omp_get_wtime() - a; nr += R.sz(i);
      u64 m = 0, chk = 0;
      a = omp_get_wtime(); probe_task(H, i, S.data() + S.off[i], S.sz(i), m, chk);
      tp += omp_get_wtime() - a; ns += S.sz(i); sink = sink + m + chk;
    }
  }
  alpha = nr ? tb / nr : 1e-8; beta = ns ? tp / ns : 1e-8;
}

// seconds per element of a short contiguous payload walk (expand mode, per-match cost)
static double calibrate_gamma() {
  vector<u32> v(4096, 3); volatile u64 sink = 0; const int reps = 20000;
  double a = omp_get_wtime(); u64 sum = 0;
  for (int r = 0; r < reps; r++) { u64 q = 0; for (size_t x = 0; x < v.size(); x++) q += v[x]; sum += q; }
  double d = omp_get_wtime() - a; sink = sink + sum;
  return d / ((double)reps * v.size());
}

// lightweight statistics: estimate the number of matches of every S-partition from a strided sample of its tuples
static void estimate_outputs(const HTs& H, const Part& S, int per, vector<double>& est) {
  est.assign(NP, 0.0);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < NP; i++) {
    size_t len = S.sz(i); if (!len) continue;
    size_t cnt = min<size_t>(per, len), stride = len / cnt; u64 mm = 0, chk = 0;
    for (size_t j = 0; j < cnt; j++) probe_expand(H, i, S.data() + S.off[i] + j * stride, 1, mm, chk);
    est[i] = (double)mm / cnt * len;
  }
}

// Measures the two parts of the steal cost used in tau_steal = Cost(CAS steal) + Cost(cache migration):
//  cas_ns : latency of a contended compare-and-swap with P threads hammering one atomic
//  mig_ns : time to pull one (average-sized) hash table into cache from memory (cache evicted first)
static void calibrate_overheads(const HTs& H, int P, double& cas_ns, double& mig_ns) {
  atomic<long> x{0}; const int K = 100000; double t0 = omp_get_wtime();
  #pragma omp parallel num_threads(P)
  { for (int i = 0; i < K; i++) { long v = x.load(memory_order_relaxed);
      while (!x.compare_exchange_weak(v, v + 1, memory_order_seq_cst, memory_order_relaxed)) {} } }
  cas_ns = (omp_get_wtime() - t0) / K * 1e9;
  vector<char> ev(64u << 20); volatile u64 sink = 0; double tot = 0; int cnt = 0;
  for (int j = 0; j < 8; j++) {
    int i = (100 + 97 * j) % NP;
    for (size_t k = 0; k < ev.size(); k += 64) ev[k] = (char)(k + j);  // evict caches
    sink = sink + ev[(j * 64) % ev.size()];
    const char* b = H.expand ? (const char*)(H.arenaE.get() + H.off[i]) : (const char*)(H.arena.get() + H.off[i]);
    size_t bytes = H.cap[i] * (H.expand ? sizeof(SlotE) : sizeof(Slot));
    double a = omp_get_wtime(); u64 sum = 0; for (size_t k = 0; k < bytes; k += 64) sum += b[k];
    tot += omp_get_wtime() - a; sink = sink + sum; cnt++;
  }
  mig_ns = tot / cnt * 1e9;
}

// ------------------------------------------------------------------ main
static double median(vector<double> v) {
  if (v.empty()) return 0;
  sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; i++) {
    string a = argv[i]; auto val = [&]() { return string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--n") c.n = stoull(val());
    else if (a == "--key_range") c.key_range = stoul(val());
    else if (a == "--zs") { c.zs.clear(); for (auto& s : split(val())) c.zs.push_back(stod(s)); }
    else if (a == "--threads") { c.threads.clear(); for (auto& s : split(val())) c.threads.push_back(stoi(s)); }
    else if (a == "--reps") c.reps = stoi(val());
    else if (a == "--warmup") c.warmup = stoi(val());
    else if (a == "--methods") c.methods = split(val());
    else if (a == "--out") c.out = val();
    else if (a == "--paper_factor") c.paper_factor = stod(val());
    else if (a == "--paper_chunk") c.paper_chunk = stoul(val());
    else if (a == "--granularity") c.granularity = stod(val());
    else if (a == "--min_chunk") c.min_chunk = stoul(val());
    else if (a == "--budget_per_thread") c.budget_per_thread = stoi(val());
    else if (a == "--task_overhead_ns") { string v = val(); if (v == "auto") c.ovh_auto = true; else { c.ovh_auto = false; c.task_overhead_ns = stod(v); } }
    else if (a == "--ns") c.ns = stoull(val());
    else if (a == "--expand") c.expand = stoi(val()) != 0;
    else if (a == "--r_z") c.r_z = stod(val());
    else if (a == "--samples") c.samples = stoi(val());
    else if (a == "--plan_gate") c.plan_gate = stod(val());
    else if (a == "--gamma_ns") { string v = val(); if (v == "auto") c.gamma_auto = true; else { c.gamma_auto = false; c.gamma_ns = stod(v); } }
    else if (a == "--init") c.init = val();
    else if (a == "--include_build") c.include_build = stoi(val());
    else if (a == "--tau_ns") { string v = val(); if (v == "auto") c.tau_auto = true; else { c.tau_auto = false; c.tau_ns = stod(v); } }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  omp_set_dynamic(0);
  FILE* f = fopen(c.out.c_str(), "w");
  if (!f) { perror("open output"); return 2; }
  fprintf(f, "z,threads,method,rep,warmup,n,ns,partition_s,build_s,probe_s,total_s,seq_s,matches,correct,"
             "lambda,max_avg_part,tasks,steal_attempts,steals,alpha,beta,tau_ns,overhead_ns,plan_s,expand,r_z\n");
  int failures = 0;
  printf("|R|=%zu |S|=%zu key_range=%u reps=%d (+%d warmup) init=%s workload=%s r_z=%.2f\n", c.n, c.ns ? c.ns : c.n, c.key_range, c.reps, c.warmup, c.init.c_str(), c.expand ? "expand" : "count", c.r_z);

  for (double z : c.zs) {
    size_t nS = c.ns ? c.ns : c.n;
    vector<Tuple> Rr = gen_keys(c.n, c.key_range, c.r_z, c.seedR);
    for (size_t i = 0; i < Rr.size(); i++) Rr[i].payload = (u32)i;
    vector<Tuple> Sr = gen_keys(nS, c.key_range, z, c.seedS);
    u64 ref = 0, refchk = 0; vector<double> seqt;
    for (int r = 0; r < c.warmup + c.reps; r++) {
      double a = omp_get_wtime(); seq_join(Rr, Sr, c.expand, ref, refchk); double d = omp_get_wtime() - a;
      if (r >= c.warmup) seqt.push_back(d);
    }
    double seq_s = median(seqt);
    printf("\n== z=%.2f  sequential: %.3fs  matches=%llu\n", z, seq_s, (unsigned long long)ref);

    for (int p : c.threads) {
      omp_set_num_threads(p);
      Part R, S; HTs H; double cas_ns = 0, mig_ns = 0, gamma = 0; bool have_ovh = false;
      for (int rep = 0; rep < c.warmup + c.reps; rep++) {
        double a = omp_get_wtime(); partition2(Rr, R); partition2(Sr, S); double part_s = omp_get_wtime() - a;
        a = omp_get_wtime(); build_all(R, H, c.expand); double build_s = omp_get_wtime() - a;
        double alpha, beta; calibrate(R, S, H, alpha, beta);
        size_t mx = 0; for (int i = 0; i < NP; i++) mx = max(mx, S.sz(i));
        double maxavg = mx / ((double)S.n / NP);
        Cfg cc = c;  // per-run copy with measured thresholds
        if (!have_ovh) {
          gamma = c.gamma_auto ? calibrate_gamma() : c.gamma_ns * 1e-9;
          if (c.tau_auto || c.ovh_auto) calibrate_overheads(H, p, cas_ns, mig_ns);
          have_ovh = true;
        }
        if (c.tau_auto || c.ovh_auto) {
          if (c.tau_auto) cc.tau_ns = cas_ns + mig_ns;
          if (c.ovh_auto) cc.task_overhead_ns = cas_ns + mig_ns;
        }
        // task lists are built once per rep, outside the timed probe phase (planning cost is tiny)
        a = omp_get_wtime(); vector<Task> paperT = make_paper_tasks(S, c); double plan_paper = omp_get_wtime() - a;
        a = omp_get_wtime(); u64 nsplit = 0; vector<double> est;
        bool do_sample = c.expand && maxavg >= c.plan_gate;  // overhead-aware: no sampling when the data is nearly uniform
        if (do_sample) estimate_outputs(H, S, c.samples, est);
        vector<Task> adaptT = make_adaptive_tasks(R, S, cc, alpha, beta, gamma, do_sample ? &est : nullptr, p, nsplit);
        double plan_adapt = omp_get_wtime() - a;
        for (size_t mi = 0; mi < c.methods.size(); mi++) {
          const string& m = c.methods[(mi + rep) % c.methods.size()];  // rotate order to avoid bias
          RunStats st;
          if (m == "static") st = run_static(H, S, p);
          else if (m == "paper") st = run_queue(paperT, H, S, p);
          else if (m == "adaptive1") st = run_queue(adaptT, H, S, p);
          else if (m == "adaptive2") st = run_steal(adaptT, H, S, p, cc);
          else { fprintf(stderr, "unknown method %s\n", m.c_str()); return 2; }
          bool ok = st.matches == ref && (!c.expand || st.chk == refchk); if (!ok) failures++;
          double plan = (m == "static") ? 0.0 : (m == "paper" ? plan_paper : plan_adapt);  // planning is charged to the method
          double total = part_s + build_s + plan + st.probe_s;
          fprintf(f, "%.2f,%d,%s,%d,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%d,%.4f,%.2f,%llu,%llu,%llu,%.3e,%.3e,%.0f,%.0f,%.6f,%d,%.2f\n",
                  z, p, m.c_str(), rep, rep < c.warmup, c.n, S.n, part_s, build_s, st.probe_s, total, seq_s,
                  (unsigned long long)st.matches, ok, st.lambda, maxavg, (unsigned long long)st.tasks,
                  (unsigned long long)st.steal_att, (unsigned long long)st.steals, alpha, beta, cc.tau_ns, cc.task_overhead_ns, plan, (int)c.expand, c.r_z);
          printf("  p=%d rep=%d %-9s total=%.1fms probe=%.2fms lambda=%.2f tasks=%llu steals=%llu %s\n", p, rep,
                 m.c_str(), total * 1e3, st.probe_s * 1e3, st.lambda, (unsigned long long)st.tasks,
                 (unsigned long long)st.steals, ok ? "PASS" : "FAIL");
        }
        fflush(f);
      }
    }
  }
  fclose(f);
  printf("\n%s (%d incorrect runs). Results in %s\n", failures ? "FAILED" : "ALL CORRECT", failures, c.out.c_str());
  return failures ? 1 : 0;
}