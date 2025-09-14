#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace cuda_utils {

// Simple CUDA timer using events
class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_time;
    bool is_started;

public:
    CudaTimer() : elapsed_time(0.0f), is_started(false) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
        is_started = true;
    }

    void stop() {
        if (!is_started) {
            std::cerr << "Timer was not started!" << std::endl;
            return;
        }
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        is_started = false;
    }

    float getElapsedTime() const {
        return elapsed_time;
    }

    void reset() {
        elapsed_time = 0.0f;
        is_started = false;
    }
};

// Advanced timer for profiling multiple kernels
class CudaProfiler {
private:
    std::map<std::string, std::vector<float>> timings;
    std::map<std::string, CudaTimer*> timers;

public:
    ~CudaProfiler() {
        for (auto& pair : timers) {
            delete pair.second;
        }
    }

    void startTimer(const std::string& name) {
        if (timers.find(name) == timers.end()) {
            timers[name] = new CudaTimer();
        }
        timers[name]->start();
    }

    void stopTimer(const std::string& name) {
        if (timers.find(name) == timers.end()) {
            std::cerr << "Timer '" << name << "' not found!" << std::endl;
            return;
        }
        timers[name]->stop();
        timings[name].push_back(timers[name]->getElapsedTime());
    }

    float getAverageTime(const std::string& name) const {
        auto it = timings.find(name);
        if (it == timings.end() || it->second.empty()) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (float time : it->second) {
            sum += time;
        }
        return sum / it->second.size();
    }

    float getMinTime(const std::string& name) const {
        auto it = timings.find(name);
        if (it == timings.end() || it->second.empty()) {
            return 0.0f;
        }

        float min_time = it->second[0];
        for (float time : it->second) {
            min_time = std::min(min_time, time);
        }
        return min_time;
    }

    float getMaxTime(const std::string& name) const {
        auto it = timings.find(name);
        if (it == timings.end() || it->second.empty()) {
            return 0.0f;
        }

        float max_time = it->second[0];
        for (float time : it->second) {
            max_time = std::max(max_time, time);
        }
        return max_time;
    }

    void printSummary() const {
        std::cout << "\n=== CUDA Profiling Summary ===" << std::endl;
        for (const auto& pair : timings) {
            const std::string& name = pair.first;
            const std::vector<float>& times = pair.second;

            if (!times.empty()) {
                std::cout << name << ":" << std::endl;
                std::cout << "  Calls: " << times.size() << std::endl;
                std::cout << "  Average: " << getAverageTime(name) << " ms" << std::endl;
                std::cout << "  Min: " << getMinTime(name) << " ms" << std::endl;
                std::cout << "  Max: " << getMaxTime(name) << " ms" << std::endl;
            }
        }
        std::cout << "==============================\n" << std::endl;
    }

    void reset() {
        timings.clear();
        for (auto& pair : timers) {
            pair.second->reset();
        }
    }
};

// Helper function to compute GFLOPS for GEMM
inline double computeGFLOPS(int M, int N, int K, float time_ms) {
    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (time_ms * 1e-3);
    return gflops;
}

// Helper function to compute bandwidth in GB/s
inline double computeBandwidth(size_t bytes, float time_ms) {
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    double seconds = time_ms / 1000.0;
    return gb / seconds;
}

} // namespace cuda_utils