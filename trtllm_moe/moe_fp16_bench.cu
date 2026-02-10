/*
 * TensorRT-LLM MoE FP16 Kernel Benchmark
 * Standalone benchmark for Hopper FP16 MoE GEMM
 *
 * Usage:
 *   ./moe_fp16_bench --num_experts=8 --tokens=1024 --hidden=4096 --inter=11008
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include MoE kernel headers
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"

// Include the implementation - this will instantiate the template
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu"

using namespace tensorrt_llm::kernels::cutlass_kernels;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

struct Options {
    int num_experts = 8;
    int tokens_per_expert = 128;
    int hidden_size = 4096;
    int inter_size = 11008;
    int warmup = 10;
    int iterations = 100;
    bool help = false;

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.find("--num_experts=") == 0) num_experts = std::stoi(arg.substr(14));
            else if (arg.find("--tokens=") == 0) tokens_per_expert = std::stoi(arg.substr(9));
            else if (arg.find("--hidden=") == 0) hidden_size = std::stoi(arg.substr(9));
            else if (arg.find("--inter=") == 0) inter_size = std::stoi(arg.substr(8));
            else if (arg.find("--warmup=") == 0) warmup = std::stoi(arg.substr(9));
            else if (arg.find("--iterations=") == 0) iterations = std::stoi(arg.substr(13));
            else if (arg == "--help" || arg == "-h") help = true;
        }
    }

    void print_usage() {
        std::cout << "MoE FP16 Benchmark\n"
                  << "Options:\n"
                  << "  --num_experts=N    Number of experts (default: 8)\n"
                  << "  --tokens=N         Tokens per expert (default: 128)\n"
                  << "  --hidden=N         Hidden size (default: 4096)\n"
                  << "  --inter=N          Intermediate size (default: 11008)\n"
                  << "  --warmup=N         Warmup iterations (default: 10)\n"
                  << "  --iterations=N     Benchmark iterations (default: 100)\n";
    }
};

int main(int argc, char** argv) {
    Options opts;
    opts.parse(argc, argv);

    if (opts.help) {
        opts.print_usage();
        return 0;
    }

    std::cout << "=== TensorRT-LLM MoE FP16 Benchmark ===" << std::endl;

    // Check CUDA device
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "SM: " << props.major << "." << props.minor << std::endl;

    if (props.major < 9) {
        std::cout << "This test requires Hopper (SM90) or newer" << std::endl;
        return 0;
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Experts: " << opts.num_experts << std::endl;
    std::cout << "  Tokens/expert: " << opts.tokens_per_expert << std::endl;
    std::cout << "  Hidden: " << opts.hidden_size << std::endl;
    std::cout << "  Intermediate: " << opts.inter_size << std::endl;

    // Create MoE GEMM runner
    MoeGemmRunner<half, half, half> runner;

    std::cout << "\nMoeGemmRunner created successfully" << std::endl;
    std::cout << "SM version: " << runner.getSM() << std::endl;
    std::cout << "TMA Warp Specialized support: " << (runner.supportsTmaWarpSpecialized() ? "Yes" : "No") << std::endl;

    // Get available configs
    auto configs = runner.getConfigs(false);
    std::cout << "Available configs: " << configs.size() << std::endl;

    // TODO: Add actual benchmark with memory allocation and kernel launch

    return 0;
}
