#pragma once

#include <algorithm>
#include <chrono>

class ScopedTimer {
public:
    explicit ScopedTimer(double& outMs) : outMs_(outMs) {
        start_ = std::chrono::steady_clock::now();
    }

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        outMs_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    double& outMs_;
    std::chrono::steady_clock::time_point start_;
};

struct TimingStats {
    double minMs = 0.0;
    double avgMs = 0.0;
    double maxMs = 0.0;
};

template <typename Fn>
inline void timeOnce(Fn&& fn, double& outMs) {
    ScopedTimer timer(outMs);
    fn();
}

template <typename Fn>
inline TimingStats measureMs(int warmup, int runs, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    TimingStats stats{};
    double total = 0.0;
    stats.minMs = 1e18;
    stats.maxMs = 0.0;

    for (int i = 0; i < runs; ++i) {
        double ms = 0.0;
        timeOnce(fn, ms);
        total += ms;
        stats.minMs = std::min(stats.minMs, ms);
        stats.maxMs = std::max(stats.maxMs, ms);
    }

    stats.avgMs = total / runs;
    return stats;
}
