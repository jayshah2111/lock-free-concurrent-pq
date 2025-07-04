#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <limits>
#include <cstdlib>

#include "lockfree_pq.hpp"

using namespace lf;
using hr_clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

int main(int argc, char* argv[]) {
    size_t num_producers = 4;
    size_t num_consumers = 4;
    size_t iterations = 100000;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--producers") == 0 && i + 1 < argc) {
            num_producers = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--consumers") == 0 && i + 1 < argc) {
            num_consumers = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        }
    }

    LockFreePQ<int> pq;
    std::atomic<bool> producers_done(false);
    std::atomic<size_t> total_pushes(0);
    std::atomic<size_t> total_pops(0);

    // Per-thread latency storage
    std::vector<std::vector<long long>> push_latencies(num_producers);
    std::vector<std::vector<long long>> pop_latencies(num_consumers);

    // Launch producer threads
    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for (size_t i = 0; i < num_producers; ++i) {
        producers.emplace_back([&, i]() {
            std::mt19937_64 rng(std::random_device{}());
            std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
            auto& lat = push_latencies[i];
            lat.reserve(iterations);
            for (size_t j = 0; j < iterations; ++j) {
                int value = dist(rng);
                auto t1 = hr_clock::now();
                pq.push(value);
                auto t2 = hr_clock::now();
                lat.push_back(std::chrono::duration_cast<ns>(t2 - t1).count());
                total_pushes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Launch consumer threads
    std::vector<std::thread> consumers;
    consumers.reserve(num_consumers);
    for (size_t i = 0; i < num_consumers; ++i) {
        consumers.emplace_back([&, i]() {
            auto& lat = pop_latencies[i];
            lat.reserve((iterations * num_producers) / num_consumers + 1);
            int last_value = std::numeric_limits<int>::min();
            while (!producers_done.load(std::memory_order_acquire) || pq.size() > 0) {
                auto t1 = hr_clock::now();
                int out;
                if (pq.pop(out)) {
                    auto t2 = hr_clock::now();
                    lat.push_back(std::chrono::duration_cast<ns>(t2 - t1).count());
                    total_pops.fetch_add(1, std::memory_order_relaxed);
                    // Validate monotonicity
                    if (out < last_value) {
                        std::cerr << "Monotonicity violated: " << out << " after " << last_value << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    last_value = out;
                }
            }
        });
    }

    // Benchmark window
    auto bench_start = hr_clock::now();
    for (auto& t : producers) t.join();
    producers_done.store(true, std::memory_order_release);
    for (auto& t : consumers) t.join();
    auto bench_end = hr_clock::now();

    // Compute overall throughput
    double total_seconds = std::chrono::duration<double>(bench_end - bench_start).count();
    size_t ops_count = total_pushes.load() + total_pops.load();
    double throughput = ops_count / total_seconds;

    std::cout << "Throughput: " << throughput << " ops/sec" << std::endl;

    // Merge and analyze pop latencies
    std::vector<long long> pops;
    for (auto& v : pop_latencies) pops.insert(pops.end(), v.begin(), v.end());
    std::sort(pops.begin(), pops.end());

    auto percentile = [&](double p) {
        size_t idx = std::min(static_cast<size_t>((p / 100.0) * pops.size()), pops.size() - 1);
        return pops[idx];
    };

    std::cout << "Latency percentiles (pop) [ns]: p50=" << percentile(50)
              << ", p99=" << percentile(99)
              << ", p999=" << percentile(99.9) << std::endl;

    // ASCII histogram for pop latencies
    const int bins = 10;
    if (!pops.empty()) {
        long long min_lat = pops.front();
        long long max_lat = pops.back();
        long long range = max_lat - min_lat + 1;
        std::vector<size_t> counts(bins, 0);
        for (auto v : pops) {
            int bin = static_cast<int>((v - min_lat) * bins / range);
            if (bin >= bins) bin = bins - 1;
            ++counts[bin];
        }
        std::cout << "Latency histogram (pop) [ns]:" << std::endl;
        for (int i = 0; i < bins; ++i) {
            long long start = min_lat + range * i / bins;
            long long end   = min_lat + range * (i + 1) / bins;
            std::cout << "[" << start << ".." << end << ") : ";
            int bar = static_cast<int>(50.0 * counts[i] / pops.size());
            for (int j = 0; j < bar; ++j) std::cout << '#';
            std::cout << std::endl;
        }
    }

    return 0;
}
